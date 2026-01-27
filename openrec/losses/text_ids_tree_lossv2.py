import torch
import torch.nn as nn
import torch.nn.functional as F

class TextIDSTreeLossv2(nn.Module):
    """
    text: CE (seq2seq next-token)
    ids:  CTC
    tree: parent-pointer CE (aux)
    """

    def __init__(self, ignore_index: int = 0, lambda_struct: float = 1.0, lambda_text: float = 1.0, lambda_ids: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index  # also used as CTC blank
        self.lambda_struct = lambda_struct
        self.lambda_text = lambda_text
        self.lambda_ids = lambda_ids

        self.seq_ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.ctc = nn.CTCLoss(blank=ignore_index, reduction="mean", zero_infinity=True)

    def _seq_ce(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
        # logits: [B, L, V], labels: [B, 2+max_len]
        B, L, V = logits.shape
        max_len = int(lengths.max().item())

        pred_seq = logits[:, :-1, :].contiguous()
        tgt_seq = labels[:, 1:2 + max_len].contiguous()

        if pred_seq.size(1) != tgt_seq.size(1):
            pred_seq = pred_seq[:, :tgt_seq.size(1), :]

        loss = self.seq_ce(pred_seq.reshape(-1, V), tgt_seq.reshape(-1))
        valid = (tgt_seq != self.ignore_index).reshape(-1)
        loss = loss.masked_select(valid).mean() if valid.any() else loss.mean()
        return loss

    def _ctc_loss(self, logits_ids_ctc: torch.Tensor, ids_ctc_labels: torch.Tensor, ids_ctc_lengths: torch.Tensor):
        # logits_ids_ctc: [B, T, V]
        B, T, V = logits_ids_ctc.shape
        device = logits_ids_ctc.device

        log_probs = F.log_softmax(logits_ids_ctc, dim=-1)          # [B,T,V]
        log_probs = log_probs.transpose(0, 1).contiguous()         # [T,B,V]
        log_probs = log_probs.float()  # amp-safe

        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        target_lengths = ids_ctc_lengths.to(device=device, dtype=torch.long)

        # targets can be [B, S] padded, CTCLoss will read only first target_lengths[b]
        targets = ids_ctc_labels.to(device=device, dtype=torch.long)

        return self.ctc(log_probs, targets, input_lengths, target_lengths)

    def _struct_ce_ctc(self, sim, parents_ctc, lengths):
        device = sim.device
        B, L, _ = sim.shape
        parent = parents_ctc[:, :L].to(device=device).clone().long()

        # ✅ 只把 <0 当 ignore；0 是合法 parent
        parent[parent < 0] = -1

        # ✅ 越界丢掉
        parent[parent >= L] = -1

        # ✅ mask 掉超过各自 length 的行
        row_mask = torch.arange(L, device=device)[None, :] >= lengths[:, None].to(device)
        parent[row_mask] = -1

        logits = sim[:, :L, :L].float()

        # （可选）列也 mask：不允许指向 pad 列
        neg = -1e9
        logits = logits.masked_fill(row_mask[:, None, :], neg)

        flat_logits = logits.reshape(-1, L)
        flat_targets = parent.reshape(-1)

        valid = flat_targets >= 0
        if not valid.any():
            return logits.sum() * 0.0

        v_logits = torch.nan_to_num(flat_logits[valid], nan=neg, posinf=1e4, neginf=neg)
        v_targets = flat_targets[valid]
        return F.cross_entropy(v_logits, v_targets, reduction="mean")

    def forward(self, pred, batch):
        # pred: (logits_text, logits_ids_ctc, sim_ids)
        assert isinstance(pred, (tuple, list)) and len(pred) == 3
        logits_text, logits_ids_ctc, sim_ids = pred

        # batch order per keep_keys:
        # [image, label, length, ids_ctc_label, ids_ctc_length, tree_parents_label]
        text_labels = batch[1]
        text_lengths = batch[2]
        ids_ctc_labels = batch[3]
        ids_ctc_lengths = batch[4]
        tree_parents = batch[5]

        if self.lambda_text != 0:
            text_loss = self._seq_ce(logits_text, text_labels, text_lengths)
        else:
            text_loss = torch.tensor(0.0, device=logits_text.device)

        if self.lambda_ids != 0:
            ids_loss = self._ctc_loss(logits_ids_ctc, ids_ctc_labels, ids_ctc_lengths)
        else:
            ids_loss = torch.tensor(0.0, device=logits_ids_ctc.device)

        if self.lambda_struct != 0:
            struct_loss = self._struct_ce_ctc(sim_ids, tree_parents, ids_ctc_lengths)
        else:
            struct_loss = torch.tensor(0.0, device=sim_ids.device)


        loss = self.lambda_text * text_loss + self.lambda_ids * ids_loss + self.lambda_struct * struct_loss
        return {"loss": loss, "text_loss": text_loss, "ids_loss": ids_loss, "struct_loss": struct_loss}


# python -m openrec.losses.text_ids_tree_lossv2
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
        criterion = TextIDSTreeLossv2(ignore_index=0, lambda_struct=1.0, lambda_text=1.0, lambda_ids=1.0)
        
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
