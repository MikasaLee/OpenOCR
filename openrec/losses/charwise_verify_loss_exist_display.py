"""
CharWiseVerifyExistDisplayLoss:
  - Text CE: standard next-token cross entropy for AR text branch
  - Per-char IDS CTC loss: decoder returns pre-computed scalar CTC loss
    - Illegal loss: grammar penalty from decoder (independent)
        - Valid loss: query-valid prefix BCE loss from valid head (independent)
"""

import torch
import torch.nn as nn


class CharWiseVerifyExistDisplayLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = 0,
        lambda_text: float = 1.0,
        lambda_ids: float = 1.0,
        lambda_illegal: float = 0.1,
        lambda_valid: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_text = lambda_text
        self.lambda_ids = lambda_ids
        self.lambda_illegal = lambda_illegal
        self.lambda_valid = lambda_valid

        self.seq_ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def _text_ce(self, logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
        """Standard seq2seq text CE loss."""
        _, _, v = logits.shape
        max_len = int(lengths.max().item())

        pred_seq = logits[:, :-1, :].contiguous()
        tgt_seq = labels[:, 1:2 + max_len].contiguous()

        if pred_seq.size(1) != tgt_seq.size(1):
            pred_seq = pred_seq[:, :tgt_seq.size(1), :]

        loss = self.seq_ce(pred_seq.reshape(-1, v), tgt_seq.reshape(-1))
        valid = (tgt_seq != self.ignore_index).reshape(-1)
        loss = loss.masked_select(valid).mean() if valid.any() else loss.mean()
        return loss

    def forward(self, pred, batch):
        """
        pred: (logits_text, (ctc_loss, decoded_ids, valid_loss), char_feat, illegal_penalty, max_text)
        batch: [image, label, length, per_char_ids_labels, per_char_ids_lengths]
        """
        logits_text, logits_ids, _char_feat, illegal_penalty, _max_text = pred

        text_labels = batch[1]
        text_lengths = batch[2]

        text_loss = self._text_ce(logits_text, text_labels, text_lengths) if self.lambda_text != 0 else \
            torch.tensor(0.0, device=logits_text.device)

        if self.lambda_ids != 0:
            ids_loss = logits_ids[0]
        else:
            ids_loss = torch.tensor(0.0, device=logits_text.device)

        if len(logits_ids) >= 3 and isinstance(logits_ids[2], torch.Tensor):
            valid_loss = logits_ids[2]
        else:
            valid_loss = torch.tensor(0.0, device=logits_text.device)

        illegal_loss = illegal_penalty if isinstance(illegal_penalty, torch.Tensor) else torch.tensor(0.0, device=logits_text.device)

        loss = (
            self.lambda_text * text_loss
            + self.lambda_ids * ids_loss
            + self.lambda_illegal * illegal_loss
            + self.lambda_valid * valid_loss
        )

        return {
            "loss": loss,
            "text_loss": text_loss,
            "ids_loss": ids_loss,
            "illegal_loss": illegal_loss,
            "valid_loss": valid_loss,
        }
