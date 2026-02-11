"""
NRTRRTDDecoder: Standard Transformer Decoder with an auxiliary RTD
(Replaced Token Detection) binary classification head.

Single-pass design:
  - Teacher-forcing input is the (possibly corrupted) label sequence.
  - Recognition head predicts the next token  (CE loss).
  - RTD head predicts whether the *current input* token was replaced (BCE).

Data layout expected by forward():
  data = [label, length]          # training (rtd_label is in batch for loss)
         or None                  # inference
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openrec.modeling.common import Mlp

# ---------- reuse building-blocks from nrtr_decoder ----------
from .nrtr_decoder import (
    Embeddings,
    MultiheadAttention,
    PositionalEncoding,
    TransformerBlock,
)


class NRTRRTDDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        num_encoder_layers=-1,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1  # PAD index
        self.bos = out_channels - 2
        self.eos = 0
        self.max_len = max_len

        d_model = in_channels
        dim_feedforward = d_model * 4
        nhead = nhead if nhead is not None else d_model // 32

        # ---------- token embedding + positional encoding ----------
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model,
        )

        # ---------- optional encoder layers (set -1/0 to skip) ----------
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                TransformerBlock(
                    d_model, nhead, dim_feedforward,
                    attention_dropout_rate, residual_dropout_rate,
                    with_self_attn=True, with_cross_attn=False,
                ) for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder = None

        # ---------- decoder layers ----------
        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model, nhead, dim_feedforward,
                attention_dropout_rate, residual_dropout_rate,
                with_self_attn=True, with_cross_attn=True,
            ) for _ in range(num_decoder_layers)
        ])

        # ---------- recognition projection head ----------
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels - 2,
                                      bias=False)
        w0 = np.random.normal(
            0.0, d_model ** -0.5,
            (d_model, self.out_channels - 2),
        ).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())

        # ---------- RTD binary classification head ----------
        self.rtd_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def _encode_memory(self, src):
        """Run optional encoder layers; returns memory (B, N, C)."""
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for layer in self.encoder:
                src = layer(src)
        return src

    # ------------------------------------------------------------------ #
    def forward_train(self, src, tgt):
        """
        Args:
            src : (B, N, C)  — image encoder output
            tgt : (B, L)     — token ids [BOS, ỹ₁, ..., ỹ_T, EOS, PAD...]
                               (already sliced to 2+max_len by forward())
        Returns:
            dict with 'rec_pred' (B, T+1, vocab-2) and 'rtd_pred' (B, T+1)
        """
        tgt = tgt[:, :-1]  # remove last → [BOS, ỹ₁, ..., ỹ_T]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[1], device=src.device)

        memory = self._encode_memory(src)

        for layer in self.decoder:
            tgt = layer(tgt, memory, self_mask=tgt_mask)

        hidden = tgt  # (B, T+1, d_model)

        # recognition logits
        rec_logit = self.tgt_word_prj(hidden)  # (B, T+1, vocab-2)

        # RTD scores  (position 0 = BOS, positions 1..T correspond to ỹ₁..ỹ_T)
        rtd_logit = self.rtd_head(hidden).squeeze(-1)  # (B, T+1)

        return {'rec_pred': rec_logit, 'rtd_pred': rtd_logit}

    # ------------------------------------------------------------------ #
    def forward_test(self, src):
        """Greedy AR decoding; collect RTD scores along the way."""
        bs = src.shape[0]
        memory = self._encode_memory(src)

        dec_seq = torch.full(
            (bs, self.max_len + 1), self.ignore_index,
            dtype=torch.int64, device=src.device,
        )
        dec_seq[:, 0] = self.bos

        logits = []
        rtd_scores = []
        self.attn_maps = []

        for step in range(self.max_len):
            dec_seq_embed = self.embedding(dec_seq[:, :step + 1])
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(
                dec_seq_embed.shape[1], src.device)

            tgt = dec_seq_embed
            for layer in self.decoder:
                tgt = layer(tgt, memory, self_mask=tgt_mask)

            self.attn_maps.append(
                self.decoder[-1].cross_attn.attn_map[0][:, -1:, :])

            last_hidden = tgt[:, -1:, :]  # (B, 1, d)

            # recognition
            word_prob = F.softmax(self.tgt_word_prj(last_hidden), dim=-1)
            logits.append(word_prob)

            # RTD score for the input token at this step
            rtd_score = torch.sigmoid(
                self.rtd_head(last_hidden).squeeze(-1))  # (B, 1)
            rtd_scores.append(rtd_score)

            if step < self.max_len:
                dec_seq[:, step + 1] = word_prob.squeeze(1).argmax(-1)
                if (dec_seq == self.eos).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)          # (B, T', vocab-2)
        rtd_scores = torch.cat(rtd_scores, dim=1)  # (B, T')

        return {'rec_pred': logits, 'rtd_pred': rtd_scores}

    # ------------------------------------------------------------------ #
    def forward(self, src, data=None):
        """
        Args:
            src  : (B, N, C) encoder features
            data : [label, length, ...] during training, None during inference
        """
        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            return self.forward_train(src, tgt)
        else:
            return self.forward_test(src)

    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(
            torch.full((sz, sz), dtype=torch.float32, fill_value=-torch.inf),
            diagonal=1,
        )
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0).to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print('=' * 60)
    print('TEST NRTRRTDDecoder')
    print('=' * 60)

    B, N, C = 4, 64, 256
    vocab = 3024
    max_len = 15

    decoder = NRTRRTDDecoder(
        in_channels=C, out_channels=vocab, nhead=8,
        num_encoder_layers=-1, num_decoder_layers=6,
        max_len=max_len, beam_size=0,
    ).to(device)

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f'  Params: {num_params:,}')

    # TEST 1: forward_train
    print('\nTEST 1: forward_train')
    decoder.train()
    src = torch.randn(B, N, C, device=device)
    BOS_IDX, EOS_IDX, PAD_IDX = vocab - 2, 0, vocab - 1
    lengths = torch.tensor([6, 5, 4, 3], dtype=torch.long, device=device)
    label = torch.full((B, max_len + 2), PAD_IDX, dtype=torch.long, device=device)
    label[:, 0] = BOS_IDX
    for b in range(B):
        sl = int(lengths[b])
        for t in range(sl):
            label[b, 1+t] = torch.randint(1, vocab-2, (1,)).item()
        label[b, 1+sl] = EOS_IDX

    data = [label, lengths]
    out = decoder(src, data=data)
    max_t = int(lengths.max())
    assert out['rec_pred'].shape == (B, max_t+1, vocab-2), f'{out["rec_pred"].shape}'
    assert out['rtd_pred'].shape == (B, max_t+1), f'{out["rtd_pred"].shape}'
    loss = out['rec_pred'].sum() + out['rtd_pred'].sum()
    loss.backward()
    print(f'  rec_pred: {out["rec_pred"].shape}, rtd_pred: {out["rtd_pred"].shape}')
    print('  [PASS] forward_train + backward OK\n')

    # TEST 2: forward_test
    print('TEST 2: forward_test')
    decoder.eval()
    with torch.no_grad():
        out_t = decoder(torch.randn(B, N, C, device=device), data=None)
    assert out_t['rec_pred'].shape[0] == B
    assert out_t['rec_pred'].shape[1] == out_t['rtd_pred'].shape[1]
    assert out_t['rec_pred'].shape[1] <= max_len
    assert (out_t['rec_pred'] >= 0).all()
    assert (out_t['rtd_pred'] >= 0).all() and (out_t['rtd_pred'] <= 1).all()
    print(f'  rec_pred: {out_t["rec_pred"].shape}, rtd_pred: {out_t["rtd_pred"].shape}')
    print('  [PASS] forward_test OK\n')

    # TEST 3: causal mask
    print('TEST 3: causal mask')
    mask = decoder.generate_square_subsequent_mask(5, device)
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0,0,0,1] == float('-inf')
    assert mask[0,0,0,0] == 0.0
    assert mask[0,0,4,0] == 0.0
    print('  [PASS] Causal mask OK\n')

    print('=' * 60)
    print('ALL NRTRRTDDecoder TESTS PASSED')
    print('=' * 60)
