"""
RTDLoss: Joint recognition (CE) + Replaced Token Detection (BCE) loss.

Batch layout:  [image, label, length, rtd_label]
  - batch[0] = image        (not used by loss)
  - batch[1] = label        [BOS, ỹ₁, ..., ỹ_T, EOS, PAD, ...]
  - batch[2] = length       scalar, actual text length T
  - batch[3] = rtd_label    [e₁, ..., e_T, 0, 0, ...]  (1=replaced, 0=original)

Preds layout (dict from NRTRRTDDecoder):
  - preds['rec_pred']  (B, T+1, vocab-2)   — recognition logits
  - preds['rtd_pred']  (B, T+1)            — RTD logits (raw, before sigmoid)
"""

import torch
import torch.nn.functional as F
from torch import nn


class RTDLoss(nn.Module):

    def __init__(
        self,
        label_smoothing=0.1,
        rtd_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.rtd_weight = rtd_weight

    def forward(self, preds, batch):
        # ---------- recognition CE loss (same as ARLoss) ----------
        rec_pred = preds['rec_pred']              # (B, T+1, vocab-2)
        max_len = batch[2].max()
        # target: [ỹ₁, ..., ỹ_T, EOS]  (skip BOS at position 0)
        rec_tgt = batch[1][:, 1:2 + max_len]     # (B, T+1)

        rec_loss = F.cross_entropy(
            rec_pred.flatten(0, 1),
            rec_tgt.reshape(-1),
            reduction='mean',
            label_smoothing=self.label_smoothing,
            ignore_index=rec_pred.shape[-1] + 1,  # PAD index
        )

        # ---------- RTD BCE loss ----------
        rtd_pred = preds['rtd_pred']              # (B, T+1) raw logits
        rtd_label = batch[3]                      # (B, max_text_length)

        # RTD targets correspond to positions 1..T of decoder output
        # (position 0 = BOS input, skip it)
        rtd_pred_trim = rtd_pred[:, 1:1 + max_len]   # (B, T)
        rtd_tgt_trim = rtd_label[:, :max_len].float() # (B, T)

        # mask out PAD positions (where length < T)
        lengths = batch[2]  # (B,)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.float()  # (B, T)

        rtd_loss_raw = F.binary_cross_entropy_with_logits(
            rtd_pred_trim, rtd_tgt_trim, reduction='none',
        )
        rtd_loss = (rtd_loss_raw * mask).sum() / mask.sum().clamp(min=1)

        # ---------- combine ----------
        total_loss = rec_loss + self.rtd_weight * rtd_loss

        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'rtd_loss': rtd_loss,
        }


if __name__ == '__main__':
    print('=' * 60)
    print('TEST RTDLoss')
    print('=' * 60)

    B, T, max_text_len, vocab = 4, 6, 15, 3024
    BOS_IDX = vocab - 2
    EOS_IDX = 0
    PAD_IDX = vocab - 1

    loss_fn = RTDLoss(label_smoothing=0.1, rtd_weight=1.0)

    label = torch.full((B, max_text_len + 2), PAD_IDX, dtype=torch.long)
    label[:, 0] = BOS_IDX
    for b in range(B):
        seq_len = T - b
        for t in range(seq_len):
            label[b, 1 + t] = torch.randint(1, vocab - 2, (1,)).item()
        label[b, 1 + seq_len] = EOS_IDX

    lengths = torch.tensor([T, T-1, T-2, T-3], dtype=torch.long)
    rtd_label = torch.zeros(B, max_text_len, dtype=torch.long)
    for b in range(B):
        seq_len = int(lengths[b])
        n_replace = max(1, int(seq_len * 0.3))
        positions = torch.randperm(seq_len)[:n_replace]
        rtd_label[b, positions] = 1

    batch = [None, label, lengths, rtd_label]
    max_len = int(lengths.max())
    rec_pred = torch.randn(B, max_len + 1, vocab - 2, requires_grad=True)
    rtd_pred = torch.randn(B, max_len + 1, requires_grad=True)
    preds = {'rec_pred': rec_pred, 'rtd_pred': rtd_pred}

    result = loss_fn(preds, batch)
    print(f'  total={result["loss"].item():.4f}, rec={result["rec_loss"].item():.4f}, rtd={result["rtd_loss"].item():.4f}')
    assert not torch.isnan(result['loss']), 'Loss is NaN!'
    assert result['loss'].requires_grad, 'Loss should require grad'

    result['loss'].backward()
    assert rec_pred.grad is not None, 'rec_pred should have grad'
    assert rtd_pred.grad is not None, 'rtd_pred should have grad'
    print(f'  rec_pred grad norm: {rec_pred.grad.norm().item():.4f}')
    print(f'  rtd_pred grad norm: {rtd_pred.grad.norm().item():.4f}')
    print('  [PASS] Forward + Backward OK\n')

    print('  Edge case: lengths all zero ...')
    lengths_zero = torch.zeros(B, dtype=torch.long)
    batch_zero = [None, label, lengths_zero, rtd_label]
    rec_pred_z = torch.randn(B, 1, vocab - 2, requires_grad=True)
    rtd_pred_z = torch.randn(B, 1, requires_grad=True)
    preds_zero = {'rec_pred': rec_pred_z, 'rtd_pred': rtd_pred_z}
    result_zero = loss_fn(preds_zero, batch_zero)
    assert not torch.isnan(result_zero['loss']), 'Loss is NaN for zero lengths'
    print(f'  loss with zero lengths: {result_zero["loss"].item():.4f}')
    print('  [PASS] Edge case OK\n')

    print('=' * 60)
    print('ALL RTDLoss TESTS PASSED')
    print('=' * 60)
