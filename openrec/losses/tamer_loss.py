import torch
import torch.nn as nn
import torch.nn.functional as F


class TAMERLoss(nn.Module):
    """联合序列 CE 与结构父指针 CE 的损失。

    约定：
    - 目标标签 batch[1] 形状 [B, 2 + max_text_len]，内容为 [<sos>] + content + [<eos>] + <pad>...
    - 序列长度 batch[2] 为内容长度（不含 <sos>/<eos>）。
    - 树父指针 batch[*]（通过 KeepKeys 传入）与标签等长，PAD 行会在外部 sim 上被 mask；
      但为稳健起见，这里额外用 child_valid_mask 过滤。
    - 模型输出 pred 为 (logits, sim)：
      logits [B, L, V]，与标签对齐；sim [B, L, L]，已对 PAD 行/列做 -inf 掩蔽。
    """

    def __init__(self, ignore_index: int = 0, lambda_struct: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index  # <pad>
        self.lambda_struct = lambda_struct
        self.seq_ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, pred, batch):
        assert isinstance(pred, (tuple, list)) and len(pred) == 2, \
            "TAMERLoss expects model output (logits, sim)"
        logits, sim = pred
        labels = batch[1]
        lengths = batch[2]

        B, L, V = logits.shape
        max_len = int(lengths.max().item())

        # 序列 CE：预测对齐下一位（logits[:-1] 对齐 labels[1:1+max_len]）
        pred_seq = logits[:, :-1, :].contiguous()              # [B, L-1, V]
        tgt_seq = labels[:, 1:2 + max_len].contiguous()        # [B, 1+max_len]
        if pred_seq.size(1) != tgt_seq.size(1):
            pred_seq = pred_seq[:, :tgt_seq.size(1), :]
        seq_loss = self.seq_ce(pred_seq.view(-1, V), tgt_seq.view(-1))
        valid_mask = (tgt_seq != self.ignore_index).view(-1)
        seq_loss = seq_loss.masked_select(valid_mask).mean() if valid_mask.any() else seq_loss.mean()

        # 结构 CE：与 decoder 输出对齐（child=输出位置，parent=输出位置）
        # 1) 找到与 logits 长度匹配的父指针张量
        parent_tgt_full = None
        for t in batch:
            if not (isinstance(t, torch.Tensor) and t.dim() == 2 and t.size(0) == B):
                continue
            if t.dtype not in (torch.int32, torch.int64):
                continue
            if t.size(1) < L:
                continue
            # tree_parents 含 -1，优先选择；否则退化为前 L 列（兼容固定长度输入）
            parent_tgt_full = t[:, :L]
            if (t < 0).any():
                break
        if parent_tgt_full is None:
            return {'loss': seq_loss, 'seq_loss': seq_loss}

        # 2) 裁剪到与输出对齐的区间：内容+<eos>（长度 1+max_len）
        parent = parent_tgt_full[:, 1:2 + max_len].clone()  # label 坐标

        # 3) 将 label 坐标映射到输出坐标（输出位置 = label位置-1），并处理 -1/0 根
        parent[parent <= 0] = -1            # 0 或 -1 视为无父节点（ignore）
        parent[parent > 0] -= 1             # 其余左移一位，与输出对齐

        # 4) 为样本内的 <eos> 指定父指针：指向最后一个内容 token（或 -1 若无内容）
        for b in range(B):
            l = int(lengths[b].item())
            if l > 0 and l <= max_len:
                parent[b, l] = l - 1  # eos 的列索引（输出坐标）
            else:
                parent[b, l if l <= max_len else max_len] = -1

        # 5) 对超出实际长度的 padding 行强制 ignore
        row_mask = torch.arange(1 + max_len, device=parent.device)[None, :] >= (lengths[:, None] + 1)
        parent = parent.masked_fill(row_mask, -1)

        # 6) 裁剪 sim 到匹配的子/父范围
        sim_use = sim[:, : (1 + max_len), : (1 + max_len)]

        # 7) 保护：若某些子行的所有父列均被掩蔽为 -inf，则该行无法参与有效 CE，改为忽略
        flat_logits = sim_use.reshape(-1, sim_use.size(-1))
        flat_targets = parent.reshape(-1)
        row_has_valid_parent = torch.isfinite(flat_logits).any(dim=1)
        # 保护A：该行所有父列均无效（-inf/NaN），则忽略该样本
        need_ignore_row = (~row_has_valid_parent) & (flat_targets != -1)
        # 保护B：目标索引指向的父列本身是无效的（被屏蔽或NaN），则忽略该样本
        idx = torch.arange(flat_logits.size(0), device=flat_logits.device)
        valid_target_mask = flat_targets >= 0
        target_logits = flat_logits[idx[valid_target_mask], flat_targets[valid_target_mask]]
        need_ignore_target = torch.zeros_like(valid_target_mask, dtype=torch.bool)
        need_ignore_target[valid_target_mask] = ~torch.isfinite(target_logits)
        need_ignore = need_ignore_row | need_ignore_target
        if need_ignore.any():
            flat_targets = flat_targets.clone()
            flat_targets[need_ignore] = -1

        struct_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=-1,
            reduction='mean',
        )

        loss = seq_loss + self.lambda_struct * struct_loss
        return {'loss': loss, 'seq_loss': seq_loss, 'struct_loss': struct_loss}
