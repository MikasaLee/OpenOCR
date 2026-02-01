import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openrec.modeling.common import Mlp


class NRTRDecoderTest(nn.Module):
    """
    NRTR-style Transformer decoder (AR decoding).
    Added: layer-wise cross-attn map capture + simple layer focus analysis.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        num_encoder_layers=6,
        beam_size=0,  # keep for compatibility (not used here)
        num_decoder_layers=6,
        max_len=25,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super(NRTRDecoderTest, self).__init__()

        self.out_channels = out_channels

        # NOTE: keep your original conventions
        self.ignore_index = out_channels - 1
        self.bos = out_channels - 2
        self.eos = 0

        self.max_len = max_len
        d_model = in_channels
        dim_feedforward = d_model * 4
        nhead = nhead if nhead is not None else max(1, d_model // 32)

        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model
        )

        # Encoder (optional)
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dim_feedforward,
                        attention_dropout_rate,
                        residual_dropout_rate,
                        with_self_attn=True,
                        with_cross_attn=False,
                    )
                    for _ in range(num_encoder_layers)
                ]
            )
        else:
            self.encoder = None

        # Decoder (self-attn + cross-attn)
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead

        # projection
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels - 2, bias=False)
        self.apply(self._init_weights)

        # (optional) your original random init style for proj weight
        w0 = np.random.normal(0.0, d_model**-0.5, (d_model, self.out_channels - 2)).astype(
            np.float32
        )
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())

        # attn buffers (filled only in eval+collect_attn)
        self.cross_attn_maps = None  # list[L] of list[step] tensors [B,H,(q or 1),kN]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, src, tgt):
        # tgt: [B, T] with BOS ... EOS/PAD etc
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], device=src.device)

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src  # [B, sN, C]
        else:
            memory = src

        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)

        logit = self.tgt_word_prj(tgt)
        return logit

    def forward(self, src, data=None):
        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        else:
            return self.forward_test(src)

    @torch.no_grad()
    def forward_test(
        self,
        src,
        collect_attn: bool = False,
        attn_capture: str = "last",   # "last" (only last query token) or "all" (all qN)
        attn_store_cpu: bool = True,  # store attn on CPU to save VRAM
    ):
        """
        AR greedy decode.
        If collect_attn=True, will store cross-attn maps for each decoder layer at each step.

        Stored format:
          self.cross_attn_maps[layer_idx][step] = attn  [B, H, q?, kN]
          - q? = 1 if attn_capture="last", else full qN
        """
        bs = src.shape[0]
        device = src.device

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src
        else:
            memory = src

        # init decode seq
        dec_seq = torch.full(
            (bs, self.max_len + 1),
            self.ignore_index,
            dtype=torch.int64,
            device=device,
        )
        dec_seq[:, 0] = self.bos

        logits = []

        # init attn buffer
        if collect_attn:
            self.cross_attn_maps = [[] for _ in range(len(self.decoder))]
        else:
            self.cross_attn_maps = None

        for len_dec_seq in range(0, self.max_len):
            dec_seq_embed = self.embedding(dec_seq[:, : len_dec_seq + 1])
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1], device=device)

            tgt = dec_seq_embed  # [B, qN, C]
            for li, decoder_layer in enumerate(self.decoder):
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)

                if collect_attn and getattr(decoder_layer, "with_cross_attn", False):
                    # decoder_layer.cross_attn.attn_map: [B, H, qN, kN] (saved in eval mode)
                    attn = decoder_layer.cross_attn.attn_map  # may be fp16/fp32
                    if attn is None:
                        continue

                    if attn_capture == "last":
                        attn = attn[:, :, -1:, :]  # only last query token
                    elif attn_capture == "all":
                        pass
                    else:
                        raise ValueError(f"attn_capture must be 'last' or 'all', got {attn_capture}")

                    attn = attn.detach()
                    if attn_store_cpu:
                        attn = attn.float().cpu()
                    self.cross_attn_maps[li].append(attn)

            dec_output = tgt[:, -1:, :]  # last token hidden
            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            logits.append(word_prob)

            # greedy next
            dec_seq[:, len_dec_seq + 1] = word_prob.squeeze(1).argmax(-1)
            if (dec_seq == self.eos).any(dim=-1).all():
                break

        logits = torch.cat(logits, dim=1)  # [B, T, V]
        return logits

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.zeros([sz, sz], dtype=torch.float32, device=device)
        mask_inf = torch.triu(
            torch.full((sz, sz), dtype=torch.float32, fill_value=-torch.inf, device=device),
            diagonal=1,
        )
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,sz,sz]

    # ---------------------- analysis helpers ----------------------
    @torch.no_grad()
    def layer_focus_scores(self, eps: float = 1e-9) -> list:
        """
        Return per-layer "focus" score in [~0, 1]:
          focus = 1 - H(p)/log(kN)
        where p is mean over heads and (captured) queries/steps.
        Higher => attention more concentrated on fewer visual tokens.

        Requires: you ran forward_test(..., collect_attn=True)
        """
        if not self.cross_attn_maps:
            return []

        scores = []
        for li, steps in enumerate(self.cross_attn_maps):
            if not steps:
                scores.append(None)
                continue

            # steps: list of [B,H,q?,kN]
            A = torch.cat(steps, dim=2)  # concat on q-dim => [B,H, q_total, kN]
            p = A.mean(dim=1)            # avg heads => [B,q_total,kN]
            p = p / (p.sum(dim=-1, keepdim=True) + eps)

            ent = -(p * (p + eps).log()).sum(dim=-1)  # [B,q_total]
            ent_norm = ent / math.log(p.shape[-1] + eps)
            focus = 1.0 - ent_norm  # higher => more peaky

            scores.append(float(focus.mean().item()))
        return scores

    @torch.no_grad()
    def layer_token_cv(self, eps: float = 1e-9) -> torch.Tensor:
        """
        Compute per-visual-token coefficient of variation (CV) across layers:
          CV(n) = std_l a_l(n) / (mean_l a_l(n) + eps)

        a_l(n) is attention mass on visual token n, averaged over heads and queries/steps.

        Returns:
          cv: [kN] tensor (CPU)
        """
        if not self.cross_attn_maps:
            return torch.empty(0)

        layer_vecs = []
        for steps in self.cross_attn_maps:
            if not steps:
                continue
            A = torch.cat(steps, dim=2)   # [B,H,q_total,kN]
            a = A.mean(dim=1).mean(dim=1) # avg heads & queries => [B,kN]
            a = a.mean(dim=0)             # avg batch => [kN]
            a = a / (a.sum() + eps)
            layer_vecs.append(a)

        if len(layer_vecs) == 0:
            return torch.empty(0)

        M = torch.stack(layer_vecs, dim=0)     # [L,kN]
        mean = M.mean(dim=0)
        std = M.std(dim=0, unbiased=False)
        cv = std / (mean + eps)
        return cv.cpu()


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim**-0.5
        self.self_attn = self_attn

        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # will be filled in eval mode
        self.attn_map = None  # [B,H,qN,kN]

    def forward(self, query, key=None, attn_mask=None):
        B, qN = query.shape[:2]

        if self.self_attn:
            qkv = self.qkv(query)
            qkv = qkv.reshape(B, qN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            assert key is not None
            kN = key.shape[1]
            q = self.q(query).reshape(B, qN, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,qN,hd]
            kv = self.kv(key).reshape(B, kN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)  # [B,H,kN,hd]

        attn = (q.matmul(k.transpose(2, 3))) * self.scale  # [B,H,qN,kN]
        if attn_mask is not None:
            attn = attn + attn_mask  # broadcast ok: [1,1,qN,qN] for self-attn

        attn = F.softmax(attn, dim=-1)

        if not self.training:
            self.attn_map = attn  # store raw weights (before dropout)

        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose(1, 2).reshape(B, qN, self.embed_dim)
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate, self_attn=True
            )
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)

        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=False)
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            assert memory is not None
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)
