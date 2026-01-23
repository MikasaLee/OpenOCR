import torch
import torch.nn as nn
import torch.nn.functional as F


class TextIDSTreeLoss(nn.Module):
    """Dual-branch CE (text + IDS) plus IDS parent-pointer CE."""

    def __init__(self, ignore_index: int = 0, lambda_struct: float = 1.0, lambda_text: float = 1.0, lambda_ids: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_struct = lambda_struct
        self.lambda_text = lambda_text
        self.lambda_ids = lambda_ids
        self.seq_ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def _seq_ce(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
        # logits: [B, L, V], labels: [B, 2+max_len]
        B, L, V = logits.shape
        max_len = int(lengths.max().item())
        pred_seq = logits[:, :-1, :].contiguous()    # align to target next token
        tgt_seq = labels[:, 1:2 + max_len].contiguous()
        if pred_seq.size(1) != tgt_seq.size(1):
            pred_seq = pred_seq[:, :tgt_seq.size(1), :]
        loss = self.seq_ce(pred_seq.reshape(-1, V), tgt_seq.view(-1))
        valid = (tgt_seq != self.ignore_index).view(-1)
        loss = loss.masked_select(valid).mean() if valid.any() else loss.mean()
        return loss, max_len

    def _struct_ce(self, sim: torch.Tensor, parents_full: torch.Tensor, lengths: torch.Tensor, max_len: int):
        # sim: [B, L, L], parents_full: [B, L], lengths: content lengths (no sos/eos)
        # slice to content+eos length (1+max_len)
        parent = parents_full[:, 1:2 + max_len].clone()
        # shift indices to drop BOS position
        parent[parent <= 0] = -1
        parent[parent > 0] -= 1
        # set eos parent to last content token if available
        for b in range(parent.size(0)):
            l = int(lengths[b].item())
            if l > 0 and l <= max_len:
                parent[b, l] = l - 1
            else:
                parent[b, l if l <= max_len else max_len] = -1
        # mask rows beyond actual length
        row_mask = torch.arange(1 + max_len, device=parent.device)[None, :] >= (lengths[:, None] + 1)
        parent = parent.masked_fill(row_mask, -1)
        sim_use = sim[:, : (1 + max_len), : (1 + max_len)]

        flat_logits = sim_use.reshape(-1, sim_use.size(-1))
        flat_targets = parent.reshape(-1)
        row_has_valid_parent = torch.isfinite(flat_logits).any(dim=1)
        need_ignore_row = (~row_has_valid_parent) & (flat_targets != -1)
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
        return struct_loss

    def forward(self, pred, batch):
        # pred: (logits_text, logits_ids, sim_ids)
        assert isinstance(pred, (tuple, list)) and len(pred) == 3, "TextIDSTreeLoss expects (logits_text, logits_ids, sim_ids)"
        logits_text, logits_ids, sim_ids = pred
        # batch: [image, label, length, ids_label, ids_length, tree_parents_label]
        text_labels = batch[1]
        text_lengths = batch[2]
        ids_labels = batch[3]
        ids_lengths = batch[4]
        tree_parents = batch[5]

        # 保险起见，
        if self.lambda_text != 0:
            text_loss, _ = self._seq_ce(logits_text, text_labels, text_lengths)
        else: 
            text_loss = torch.tensor(0.0, device=logits_text.device)

        if self.lambda_ids != 0:
            ids_loss, max_ids = self._seq_ce(logits_ids, ids_labels, ids_lengths)
        else:
            ids_loss = torch.tensor(0.0, device=logits_ids.device)
            max_ids = 0

        if self.lambda_struct != 0:
            struct_loss = self._struct_ce(sim_ids, tree_parents, ids_lengths, max_ids)
        else:
            struct_loss = torch.tensor(0.0, device=sim_ids.device)

        loss = self.lambda_text * text_loss + self.lambda_ids * ids_loss + self.lambda_struct * struct_loss
        return {
            'loss': loss,
            'text_loss': text_loss,
            'ids_loss': ids_loss,
            'struct_loss': struct_loss,
        }

# python -m openrec.losses.text_ids_tree_loss
if __name__ == "__main__":
    import torch
    import sys
    import os

    print("Initializing TextIDSTreeLoss Check...")
    
    # Try to locate project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/lirunrui/OpenOCR"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root: {project_root}")
    
    # Use real vocab files
    text_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    
    if not os.path.exists(text_vocab_path):
        print(f"Error: Text vocab not found at {text_vocab_path}")
        sys.exit(1)
    if not os.path.exists(ids_vocab_path):
        print(f"Error: IDS vocab not found at {ids_vocab_path}")
        sys.exit(1)
        
    def get_vocab_size(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
        return len(lines) + 4
        
    real_text_vocab_size = get_vocab_size(text_vocab_path)
    real_ids_vocab_size = get_vocab_size(ids_vocab_path)
    print(f"Verified Real Text Vocab Size: {real_text_vocab_size}")
    print(f"Verified Real IDS Vocab Size: {real_ids_vocab_size}")

    try:
        criterion = TextIDSTreeLoss(ignore_index=0, lambda_struct=1.0, lambda_text=1.0, lambda_ids=1.0)
        
        bs = 2
        
        # Simulated batch max lengths
        b_max_text = 15 
        b_max_ids = 100 
        
        # pred: (logits_text, logits_ids, sim_ids)
        # using shape [B, 2 + max_len, V]
        # Use REAL vocab sizes
        logits_text = torch.randn(bs, 2 + b_max_text, real_text_vocab_size, requires_grad=True)
        logits_ids = torch.randn(bs, 2 + b_max_ids, real_ids_vocab_size, requires_grad=True)
        sim_ids = torch.randn(bs, 2 + b_max_ids, 2 + b_max_ids, requires_grad=True)
        
        pred = (logits_text, logits_ids, sim_ids)
        
        # batch inputs
        dummy_img = torch.zeros(1)
        
        text_lens = torch.tensor([5, 8])
        text_labels = torch.randint(1, real_text_vocab_size, (bs, 2 + b_max_text))
        # ensure pad
        text_labels[:, 0] = 1 # sos
        for i in range(bs):
            text_labels[i, 1 + text_lens[i]] = 2 # eos
            if 1 + text_lens[i] + 1 < text_labels.size(1):
                text_labels[i, 2 + text_lens[i]:] = 0
        
        ids_lens = torch.tensor([50, 95])
        ids_labels = torch.randint(1, real_ids_vocab_size, (bs, 2 + b_max_ids))
        
        # tree_parents: -1 or index (independent of vocab size, depends on sequence length)
        # Note: Valid parent indices are within [0, current_seq_len] usually.
        # But during debug we just need them to be < calculated max_len.
        # Max valid parent index in simulation is b_max_ids.
        tree_parents = torch.randint(-1, 95, (bs, 2 + b_max_ids)) # keep max index small to avoid out of bounds in small pred
        
        # ensure pad for ids and parents
        ids_labels[:, 0] = 1 # sos
        for i in range(bs):
            ids_labels[i, 1 + ids_lens[i]] = 2 # eos
            if 1 + ids_lens[i] + 1 < ids_labels.size(1):
                ids_labels[i, 2 + ids_lens[i]:] = 0
                tree_parents[i, 2 + ids_lens[i]:] = -1
        
        batch = [
            dummy_img,
            text_labels,
            text_lens,
            ids_labels,
            ids_lens,
            tree_parents
        ]
        
        print("Calculating Loss...")
        loss_dict = criterion(pred, batch)
        
        print("Loss Output:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
            
        total_loss = loss_dict['loss']
        total_loss.backward()
        print("Backward pass successful.")
        
        assert total_loss > 0, "Total loss should be positive"
        assert loss_dict['text_loss'] > 0, "Text loss should be positive"
        assert loss_dict['ids_loss'] > 0, "IDS loss should be positive"
        
        print("TextIDSTreeLoss Check Passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"TextIDSTreeLoss Check Failed: {e}")
