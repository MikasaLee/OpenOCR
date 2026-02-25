import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):

    def __init__(self, dropout: float, dim: int, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe1d', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        pe = self.pe1d[:steps].unsqueeze(0).expand(bsz, -1, -1)
        return self.dropout(x + pe)


class SSMDecoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 max_len: int = 25,
                 nhead: int = 12,
                 num_layers: int = 1,
                 pred_dropout: float = 0.1,
                 max_pe2d: int = 5000,
                 num_decoder_layers: int = None,
                 **kwargs):
        super().__init__()

        if num_decoder_layers is not None:
            num_layers = num_decoder_layers

        self.start_idx = out_channels - 2
        self.eos_idx = 0
        self.pad_idx = out_channels - 1
        self.num_classes = out_channels - 2
        self.max_seq_len = max_len

        d_model = in_channels
        dim_feedforward = d_model * 4
        nhead = nhead if nhead is not None else d_model // 32

        self.embedding = nn.Embedding(out_channels,
                                      d_model,
                                      padding_idx=0)

        self.pos_enc_1d = PositionalEncoding1D(pred_dropout, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=
                                                   dim_feedforward,
                                                   dropout=pred_dropout,
                                                   batch_first=True,
                                                   activation='relu')
        self.transformer_dec = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Linear(d_model, self.num_classes, bias=False)

        w0 = np.random.normal(0.0, d_model**-0.5,
                              (d_model, self.num_classes)).astype(np.float32)
        self.classifier.weight.data = torch.from_numpy(w0.transpose())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, data=None) -> torch.Tensor:
        if feat.dim() == 4:
            feat = feat.flatten(2).transpose(1, 2)

        bsz, _, _ = feat.shape
        device = feat.device

        if self.training:
            labels, lengths = data[:2]
            steps = lengths.max() + 1
            tgt = labels[:, :steps]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                steps).to(device).bool()
            tgt_key_padding_mask = (tgt == self.pad_idx)

            tgt = self.embedding(tgt)
            tgt = self.pos_enc_1d(tgt)

            dec_out = self.transformer_dec(
                tgt=tgt,
                memory=feat,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            logits = self.classifier(dec_out)
            return logits

        outputs = []
        dec_seq = torch.full((bsz, self.max_seq_len + 1),
                             self.pad_idx,
                             dtype=torch.int64,
                             device=device)
        dec_seq[:, 0] = self.start_idx
        for len_dec_seq in range(0, self.max_seq_len):
            dec_seq_embed = self.embedding(dec_seq[:, :len_dec_seq + 1])
            dec_seq_embed = self.pos_enc_1d(dec_seq_embed)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_seq_embed.shape[1]).to(device)

            tgt = self.transformer_dec(tgt=dec_seq_embed,
                                       memory=feat,
                                       tgt_mask=tgt_mask)
            dec_output = tgt[:, -1:, :]
            word_prob = F.softmax(self.classifier(dec_output), dim=-1)
            outputs.append(word_prob)
            if len_dec_seq < self.max_seq_len:
                dec_seq[:, len_dec_seq + 1] = word_prob.squeeze().argmax(-1)
                if (dec_seq == self.eos_idx).any(dim=-1).all():
                    break
        return torch.cat(outputs, dim=1)
