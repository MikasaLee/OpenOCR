"""
CharWiseVerifyLoss:
  - Text CE: standard next-token cross entropy for AR text branch
  - Per-char IDS CTC loss: decoder returns pre-computed scalar CTC loss
  - Grammar penalty: CTC frame-level grammar penalty from decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharWiseVerifyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = 0,
        lambda_text: float = 1.0,
        lambda_ids: float = 1.0,
        lambda_illegal: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_text = lambda_text
        self.lambda_ids = lambda_ids
        self.lambda_illegal = lambda_illegal

        self.seq_ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def _text_ce(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
        """Standard seq2seq text CE loss."""
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

    def forward(self, pred, batch):
        """
        pred: (logits_text, (ctc_loss, decoded_ids), char_feat, grammar_penalty, max_text)
        batch: [image, label, length, per_char_ids_labels, per_char_ids_lengths]
        """
        logits_text, logits_ids, char_feat, grammar_penalty, max_text = pred

        text_labels = batch[1]
        text_lengths = batch[2]

        # Text CE loss
        text_loss = self._text_ce(logits_text, text_labels, text_lengths) if self.lambda_text != 0 \
            else torch.tensor(0.0, device=logits_text.device)

        # Per-char IDS CTC loss (pre-computed in decoder)
        if self.lambda_ids != 0:
            ids_loss = logits_ids[0]  # scalar CTC loss from decoder
        else:
            ids_loss = torch.tensor(0.0, device=logits_text.device)

        # Grammar penalty
        illegal_loss = grammar_penalty if isinstance(grammar_penalty, torch.Tensor) \
            else torch.tensor(0.0, device=logits_text.device)

        loss = (
            self.lambda_text * text_loss
            + self.lambda_ids * ids_loss
            + self.lambda_illegal * illegal_loss
        )

        return {
            "loss": loss,
            "text_loss": text_loss,
            "ids_loss": ids_loss,
            "illegal_loss": illegal_loss,
        }


# python -m openrec.losses.charwise_verify_loss
if __name__ == "__main__":
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing CharWiseVerifyLoss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bs = 2
    max_text_len = 15
    text_vocab_size = 3024

    try:
        loss_fn = CharWiseVerifyLoss(
            ignore_index=0,
            lambda_text=1.0,
            lambda_ids=1.0,
            lambda_illegal=0.1,
        ).to(device)
        print("[OK] Loss function initialized.")

        text_lens_val = [3, 5]
        max_text = max(text_lens_val)

        logits_text = torch.randn(bs, 2 + max_text, text_vocab_size, device=device, requires_grad=True)

        # Simulate CTC loss output from decoder: (scalar_loss, decoded_ids_list)
        ctc_loss_scalar = torch.tensor(5.0, device=device, requires_grad=True)
        decoded_ids_dummy = [[[] for _ in range(n)] for n in text_lens_val]
        logits_ids = (ctc_loss_scalar, decoded_ids_dummy)

        char_feat = torch.randn(bs, max_text, 256, device=device)
        grammar_penalty = torch.tensor(0.05, device=device, requires_grad=True)

        text_labels = torch.randint(4, text_vocab_size, (bs, 2 + max_text_len), device=device)
        text_labels[:, 0] = 1
        for b in range(bs):
            text_labels[b, 1 + text_lens_val[b]] = 2
            text_labels[b, 2 + text_lens_val[b]:] = 0
        text_lengths = torch.tensor(text_lens_val, dtype=torch.long, device=device)

        per_char_ids_labels = torch.zeros(bs, max_text_len, 17, dtype=torch.long, device=device)
        per_char_ids_lengths = torch.zeros(bs, max_text_len, dtype=torch.long, device=device)

        image_placeholder = torch.zeros(bs, 3, 32, 256, device=device)
        batch = [image_placeholder, text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths]
        pred = (logits_text, logits_ids, char_feat, grammar_penalty, max_text)

        print("\nRunning loss forward...")
        result = loss_fn(pred, batch)

        for k, v in result.items():
            print(f"  {k}: {v.item():.4f}")

        assert "loss" in result
        assert "text_loss" in result
        assert "ids_loss" in result
        assert "illegal_loss" in result
        assert result["loss"].requires_grad
        print("  [OK] Loss values computed.")

        result["loss"].backward()
        assert logits_text.grad is not None
        assert ctc_loss_scalar.grad is not None
        assert grammar_penalty.grad is not None
        print("  [OK] Gradients flow through all losses.")

        print("\n" + "=" * 60)
        print("[PASS] All CharWiseVerifyLoss tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")
