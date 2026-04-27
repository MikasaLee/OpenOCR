"""
DualLevelVerifyDecoder — standalone dual-level decoder with two optional lines.

Design principle:
    Keep one single DualLevel implementation in this file and control optional
    enhancements by switches only:

    1) IDS local-fusion line (frame-level local residual for CTC)
    2) IDS->Text hidden-level fusion line (use_ids2text_adapter)

When switches are OFF:
    the corresponding enhancement line naturally deactivates in-place.
"""

from typing import Optional, List, Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)


class SinusoidalPosEnc2D(nn.Module):
    """2D sinusoidal positional encoding for flattened spatial features."""

    def __init__(self, d_model, temperature=10000.0):
        super().__init__()
        self.d_model = d_model
        if self.d_model % 2 != 0:
            raise ValueError(f"SinusoidalPosEnc2D requires even d_model, got {self.d_model}")
        half = d_model // 2
        if half % 2 != 0:
            raise ValueError(
                f"SinusoidalPosEnc2D requires (d_model//2) to be even for sin/cos interleave, got {half}"
            )
        dim_t = torch.arange(0, half, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / half))
        self.register_buffer('div_term', div_term)

    def forward(self, feat_local, local_sz):
        H, W = local_sz
        if feat_local.size(1) != H * W:
            raise ValueError(f"feat_local length {feat_local.size(1)} != H*W {H * W}")

        device = feat_local.device
        half = self.d_model // 2

        row_pos = torch.arange(H, dtype=torch.float, device=device)
        col_pos = torch.arange(W, dtype=torch.float, device=device)

        row_enc = torch.einsum('i,j->ij', row_pos, self.div_term)
        col_enc = torch.einsum('i,j->ij', col_pos, self.div_term)

        pe_row = torch.zeros(H, half, device=device)
        pe_row[:, 0::2] = row_enc.sin()
        pe_row[:, 1::2] = row_enc.cos()

        pe_col = torch.zeros(W, half, device=device)
        pe_col[:, 0::2] = col_enc.sin()
        pe_col[:, 1::2] = col_enc.cos()

        pe = torch.cat([
            pe_row.unsqueeze(1).expand(-1, W, -1),
            pe_col.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)
        pe = pe.reshape(H * W, self.d_model).unsqueeze(0)
        return feat_local + pe


class DualLevelVerifyDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        text_vocab_path: Optional[str] = None,
        ids_vocab_path: Optional[str] = None,
        char_to_ids_path: Optional[str] = None,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 64,
        cross_coverage: bool = False,
        self_coverage: bool = False,
        max_text_length: int = 25,
        max_single_char_ids_len: int = 15,
        use_space_char: bool = False,
        constrained_ctc_decode: bool = True,
        ids_syntax_max_need: int = 64,
        grammar_penalty_weight: float = 0.1,
        feat_local_channels: int = 256,
        use_ids_local_fusion: bool = True,
        use_ids2text_adapter: bool = True,
        ids2text_dropout: float = 0.1,
        detach_local_for_ids: bool = False,
        detach_ids_for_text: bool = False,
        **kwargs,
    ):
        super().__init__()

        def _infer_vocab_size(path: Optional[str]) -> int:
            if path is None:
                raise ValueError("vocab_path is required to infer vocab size.")
            with open(path, "rb") as fin:
                count = len(fin.readlines())
            if use_space_char:
                count += 1
            return count + 4

        text_vocab_size = out_channels if out_channels is not None else _infer_vocab_size(text_vocab_path)
        ids_vocab_size = _infer_vocab_size(ids_vocab_path)

        self.ignore_index = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.max_text_len = int(max_text_length)
        self.max_single_char_ids_len = int(max_single_char_ids_len)
        self.use_space_char = bool(use_space_char)
        self.text_vocab_size = text_vocab_size
        self.ids_vocab_size = ids_vocab_size

        self.constrained_ctc_decode = bool(constrained_ctc_decode)
        self.ids_syntax_max_need = int(ids_syntax_max_need)
        self.grammar_penalty_weight = float(grammar_penalty_weight)

        self.use_ids_local_fusion = bool(use_ids_local_fusion)
        self.use_ids2text_adapter = bool(use_ids2text_adapter)
        self.detach_local_for_ids = bool(detach_local_for_ids)
        self.detach_ids_for_text = bool(detach_ids_for_text)

        d_model = in_channels

        self.text_embed = nn.Sequential(
            nn.Embedding(text_vocab_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.text_pos_enc = WordPosEnc(d_model)
        self.text_norm = nn.LayerNorm(d_model)

        arm_factory = (
            lambda: AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
            if (cross_coverage or self_coverage)
            else None
        )
        self.text_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers,
            arm=arm_factory(),
        )
        self.proj_text = nn.Linear(d_model, text_vocab_size)

        num_char_locator_layers = int(kwargs.get('num_char_locator_layers', 2))
        self.char_pos_queries = nn.Parameter(torch.randn(max_text_length, d_model) * 0.02)
        char_loc_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.char_locator = nn.TransformerDecoder(char_loc_layer, num_layers=num_char_locator_layers)
        self.char_visual_ln = nn.LayerNorm(d_model)

        ctc_frames = int(kwargs.get('ctc_frames_per_char', 32))
        num_ctc_encoder_layers = int(kwargs.get('num_ctc_encoder_layers', 1))
        self.ctc_frames = ctc_frames
        self.ctc_blank_id = 0

        self.char_frame_expand = nn.Linear(d_model, ctc_frames * d_model)
        self.char_frame_pos = nn.Parameter(torch.randn(ctc_frames, d_model) * 0.02)
        ctc_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.ctc_frame_encoder = nn.TransformerEncoder(ctc_enc_layer, num_layers=num_ctc_encoder_layers)
        self.ctc_proj = nn.Linear(d_model, ids_vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.ctc_blank_id, reduction='mean', zero_infinity=True)

        if self.use_ids_local_fusion:
            self.local_pos_enc = SinusoidalPosEnc2D(feat_local_channels)
            self.local_pool_h = int(kwargs.get('local_pool_h', 2))
            self.local_frame_proj = nn.Linear(
                feat_local_channels * self.local_pool_h,
                d_model,
            )
            self.local_frame_ln = nn.LayerNorm(d_model)
            self.ids_local_alpha = nn.Parameter(torch.tensor(0.02))
        else:
            self.local_pos_enc = None
            self.local_frame_proj = None
            self.local_frame_ln = None
            self.ids_local_alpha = None

        if self.use_ids2text_adapter:
            self.ids_to_text_proj = nn.Linear(d_model, d_model)
            self.ids_to_text_ln = nn.LayerNorm(d_model)
            self.ids_to_text_drop = nn.Dropout(ids2text_dropout)
            self.ids_to_text_alpha = nn.Parameter(torch.tensor(0.05))
        else:
            self.ids_to_text_proj = None
            self.ids_to_text_ln = None
            self.ids_to_text_drop = None
            self.ids_to_text_alpha = None

        self.ids_tokens: Optional[List[str]] = None
        self.ids_token2id: Optional[Dict[str, int]] = None
        self.space_id: Optional[int] = None

        if ids_vocab_path is not None:
            ids_chars = []
            with open(ids_vocab_path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip("\n\r")
                    if s:
                        ids_chars.append(s)
            if self.use_space_char and " " not in ids_chars:
                ids_chars.append(" ")
            self.ids_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"] + ids_chars
            self.ids_token2id = {t: i for i, t in enumerate(self.ids_tokens)}
            self.space_id = self.ids_token2id.get(" ", None)

        self._ids_unary_ops: Set[int] = set()
        self._ids_binary_ops: Set[int] = set()
        self._ids_trinary_ops: Set[int] = set()
        self._ids_leaf_ids: Set[int] = set()
        self._ids_operator_ids: Set[int] = set()
        self._init_ids_operator_sets()

        self.char_to_ids_map: Optional[Dict[str, str]] = None
        if char_to_ids_path is not None:
            self.char_to_ids_map = {}
            with open(char_to_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        self.char_to_ids_map[parts[0]] = parts[1].strip()

        self.debug_print_alpha = True
        self.debug_print_interval = 200
        self._debug_step = 0

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), True, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_memory(self, x) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        if isinstance(x, dict):
            feat_global = x['feat_global']
            mem_mask = x.get('mem_mask', None)
            if mem_mask is not None:
                mem_mask = mem_mask.bool()

            if feat_global.dim() == 4:
                b, h, w, c = feat_global.shape
                mem = feat_global.view(b, h * w, c)
                if mem_mask is not None and mem_mask.dim() == 3:
                    mem_mask = mem_mask.reshape(b, h * w).bool()
                return mem, int(h), mem_mask

            h_from_meta = x.get('height', x.get('h', None))
            if h_from_meta is None:
                hw = x.get('global_hw', None)
                if hw is not None and len(hw) >= 1:
                    h_from_meta = hw[0]

            if h_from_meta is not None:
                return feat_global, int(h_from_meta), mem_mask

            inferred_h = 1 if feat_global.dim() == 3 else 2
            return feat_global, inferred_h, mem_mask

        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                feat2d, mask2d = x
                b, h, w, c = feat2d.shape
                mem = feat2d.view(b, h * w, c)
                mem_mask = mask2d.reshape(b, h * w).bool()
                return mem, h, mem_mask
            elif len(x) == 3:
                mem, hw, mem_mask = x
                h, _ = hw
                return mem, h, mem_mask.bool() if mem_mask is not None else None

        return x, 2, None

    def _prep_feat_local(self, x):
        if (not self.use_ids_local_fusion) or (not isinstance(x, dict)):
            return None, None
        feat_local = x.get('feat_local', None)
        local_sz = x.get('local_sz', None)
        if feat_local is None:
            return None, None
        feat_local_ids = feat_local.detach() if self.detach_local_for_ids else feat_local
        if (local_sz is not None) and (self.local_pos_enc is not None):
            feat_local_ids = self.local_pos_enc(feat_local_ids, local_sz)
        return feat_local_ids, local_sz

    def _init_ids_operator_sets(self):
        if self.ids_token2id is None:
            return
        unary = ["⿾", "⿿"]
        trinary = ["⿲", "⿳"]
        binary = ["⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻", "⿼", "⿽", "㇯"]

        specials = {self.ignore_index, self.bos_id, self.eos_id, self.unk_id}
        if self.space_id is not None:
            specials.add(self.space_id)

        for s in unary:
            if s in self.ids_token2id:
                self._ids_unary_ops.add(self.ids_token2id[s])
        for s in trinary:
            if s in self.ids_token2id:
                self._ids_trinary_ops.add(self.ids_token2id[s])
        for s in binary:
            if s in self.ids_token2id:
                self._ids_binary_ops.add(self.ids_token2id[s])

        self._ids_operator_ids = self._ids_unary_ops | self._ids_binary_ops | self._ids_trinary_ops
        for tok_id in range(self.ids_vocab_size):
            if tok_id not in specials and tok_id not in self._ids_operator_ids:
                self._ids_leaf_ids.add(tok_id)

    def _extract_char_features(self, memory, mem_mask, num_chars, lengths=None):
        B = memory.size(0)
        queries = self.char_pos_queries[:num_chars].unsqueeze(0).expand(B, -1, -1)

        tgt_kpm = None
        if lengths is not None:
            idx = torch.arange(num_chars, device=memory.device).unsqueeze(0)
            tgt_kpm = idx >= lengths.unsqueeze(1)

        char_feat = self.char_locator(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_mask,
        )
        char_feat = self.char_visual_ln(char_feat)
        return char_feat.unsqueeze(2)

    def _extract_char_slots(self, memory, mem_mask, num_chars, lengths=None):
        char_feat = self._extract_char_features(memory, mem_mask, num_chars, lengths=lengths)
        return char_feat[:, :, 0, :]

    def _reshape_local_2d(self, feat_local_ids, local_sz):
        if feat_local_ids is None or local_sz is None:
            return None

        B, L, C = feat_local_ids.shape
        H, W = local_sz
        if L != H * W:
            raise ValueError(f"feat_local_ids length {L} != H*W {H * W}")

        feat2d = feat_local_ids.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return feat2d

    def _build_char_width_bins(self, num_chars, width, device):
        if num_chars <= 0:
            return None, None

        edges = torch.linspace(0, width, steps=num_chars + 1, device=device)
        left = torch.floor(edges[:-1]).long()
        right = torch.ceil(edges[1:]).long()

        right = torch.maximum(right, left + 1)
        left = torch.clamp(left, min=0, max=max(width - 1, 0))
        right = torch.clamp(right, min=1, max=width)
        return left, right

    def _build_local_ctc_frames(self, feat_local_ids, local_sz, num_chars, lengths=None):
        if (not self.use_ids_local_fusion) or (feat_local_ids is None):
            return None
        if local_sz is None:
            return None
        if self.local_frame_proj is None:
            return None
        if num_chars <= 0:
            return None

        feat2d = self._reshape_local_2d(feat_local_ids, local_sz)
        B, C, _, W = feat2d.shape
        K = self.ctc_frames

        left, right = self._build_char_width_bins(num_chars, W, feat2d.device)

        local_frames_all = []
        for t in range(num_chars):
            l = int(left[t].item())
            r = int(right[t].item())
            window = feat2d[:, :, :, l:r]
            pooled = F.adaptive_avg_pool2d(window, (self.local_pool_h, K))
            pooled = pooled.permute(0, 3, 2, 1).contiguous()
            pooled = pooled.view(B, K, self.local_pool_h * C)
            frames_t = self.local_frame_proj(pooled)
            frames_t = self.local_frame_ln(frames_t)
            local_frames_all.append(frames_t)

        local_frames = torch.stack(local_frames_all, dim=1)

        if lengths is not None:
            valid_mask = (
                torch.arange(num_chars, device=local_frames.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            local_frames = local_frames * valid_mask[:, :, None, None].type_as(local_frames)

        return local_frames

    def _build_ids_states(self, memory, mem_mask, lengths):
        max_chars = int(lengths.max().item())
        if max_chars <= 0:
            return None, None, None, 0

        char_slots_global = self._extract_char_slots(memory, mem_mask, max_chars, lengths=lengths)
        valid_char_mask = torch.arange(max_chars, device=memory.device).unsqueeze(0) < lengths.unsqueeze(1)

        h_ids = F.layer_norm(char_slots_global, (char_slots_global.size(-1),))
        h_ids = h_ids * valid_char_mask.unsqueeze(-1).type_as(h_ids)
        return char_slots_global, h_ids, valid_char_mask, max_chars

    def _ctc_forward(self, char_feat, local_frame_feat=None):
        N, d = char_feat.shape
        K = self.ctc_frames
        frames = self.char_frame_expand(char_feat).view(N, K, d)

        if (
            self.use_ids_local_fusion
            and (local_frame_feat is not None)
            and (self.ids_local_alpha is not None)
        ):
            frames = frames + self.ids_local_alpha * local_frame_feat

        frames = frames + self.char_frame_pos.unsqueeze(0)
        frames = self.ctc_frame_encoder(frames)
        return self.ctc_proj(frames)

    def _ctc_loss(self, ctc_logits, ids_targets, ids_lengths):
        N, K, _ = ctc_logits.shape
        device = ctc_logits.device
        if N == 0:
            return torch.tensor(0.0, device=device)

        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        max_tgt_len = int(ids_lengths.max().item())
        targets = ids_targets[:, 1:1 + max_tgt_len].contiguous()
        input_lengths = torch.full((N,), K, dtype=torch.long, device=device)
        return self.ctc_loss_fn(log_probs, targets, input_lengths, ids_lengths.long())

    @torch.no_grad()
    def _ctc_greedy_decode(self, ctc_logits):
        pred_ids = ctc_logits.argmax(dim=-1)
        results = []
        drop_tokens = (0, 1, 2)
        for i in range(pred_ids.size(0)):
            tokens = pred_ids[i].tolist()
            collapsed = []
            prev = None
            for t in tokens:
                if t != prev:
                    collapsed.append(t)
                prev = t
            results.append([t for t in collapsed if t not in drop_tokens])
        return results

    def _is_ids_token_legal(self, token_id: int, need: int) -> bool:
        if token_id in (0, 1, 2, 3):
            return False
        if need <= 0:
            return False
        if token_id in self._ids_binary_ops:
            return (need + 1) <= self.ids_syntax_max_need
        if token_id in self._ids_trinary_ops:
            return (need + 2) <= self.ids_syntax_max_need
        return True

    @torch.no_grad()
    def _ctc_constrained_decode(self, ctc_logits):
        N, K, _ = ctc_logits.shape
        results = []

        for i in range(N):
            need = 1
            prev = -1
            decoded = []

            for k in range(K):
                scores = ctc_logits[i, k]
                top = scores.argmax().item()

                if top == 0 or top == prev:
                    prev = top
                    continue
                if need <= 0:
                    prev = 0
                    continue

                if not self._is_ids_token_legal(top, need):
                    sorted_indices = scores.argsort(descending=True)
                    found = False
                    for idx_t in sorted_indices:
                        idx = idx_t.item()
                        if idx == 0 or idx == prev:
                            continue
                        if self._is_ids_token_legal(idx, need):
                            top = idx
                            found = True
                            break
                    if not found:
                        prev = 0
                        continue

                decoded.append(top)
                prev = top

                if top in self._ids_binary_ops:
                    need += 1
                elif top in self._ids_trinary_ops:
                    need += 2
                elif top in self._ids_unary_ops:
                    pass
                elif top not in (0, 1, 2, 3):
                    need -= 1

            results.append(decoded)

        return results

    def _ctc_grammar_penalty(self, ctc_logits):
        N, K, _ = ctc_logits.shape
        device = ctc_logits.device
        if N == 0:
            return torch.tensor(0.0, device=device)

        pred_ids = ctc_logits.detach().argmax(dim=-1)
        is_blank = (pred_ids == 0)
        is_dup = torch.zeros_like(pred_ids, dtype=torch.bool)
        is_dup[:, 1:] = (pred_ids[:, 1:] == pred_ids[:, :-1])
        is_emission = ~is_blank & ~is_dup

        delta = torch.zeros(N, K, device=device)
        for op_id in self._ids_binary_ops:
            delta += (pred_ids == op_id).float()
        for op_id in self._ids_trinary_ops:
            delta += (pred_ids == op_id).float() * 2
        is_special = (pred_ids <= 3)
        is_op = torch.zeros(N, K, dtype=torch.bool, device=device)
        for op_id in self._ids_operator_ids:
            is_op = is_op | (pred_ids == op_id)
        is_leaf_emission = is_emission & ~is_special & ~is_op
        delta = delta - is_leaf_emission.float()
        delta = delta * is_emission.float()

        cum_delta = delta.cumsum(dim=1)
        need = torch.ones(N, K, device=device)
        need[:, 1:] = need[:, 1:] + cum_delta[:, :-1]

        probs = F.softmax(ctc_logits, dim=-1)
        emission_need_pos = is_emission & (need > 0)
        emission_need_done = is_emission & (need <= 0)

        penalty = torch.zeros(N, K, device=device)
        penalty += (1.0 - probs[:, :, 0]) * emission_need_done.float()
        penalty += probs[:, :, 1] * emission_need_pos.float()
        penalty += probs[:, :, 2] * emission_need_pos.float()
        penalty += probs[:, :, 3] * emission_need_pos.float()

        binary_overflow = emission_need_pos & (need + 1 > self.ids_syntax_max_need)
        for op_id in self._ids_binary_ops:
            penalty += probs[:, :, op_id] * binary_overflow.float()

        trinary_overflow = emission_need_pos & (need + 2 > self.ids_syntax_max_need)
        for op_id in self._ids_trinary_ops:
            penalty += probs[:, :, op_id] * trinary_overflow.float()

        n_emissions = is_emission.float().sum().clamp_min(1.0)
        frame_penalty = penalty.sum() / n_emissions

        return frame_penalty

    def _fuse_text_hidden_with_ids(self, text_hidden, h_ids):
        if (not self.use_ids2text_adapter) or (h_ids is None):
            return text_hidden
        if (self.ids_to_text_proj is None) or (self.ids_to_text_ln is None) or (self.ids_to_text_alpha is None):
            return text_hidden

        z = h_ids.detach() if self.detach_ids_for_text else h_ids
        z = self.ids_to_text_proj(z)

        B, L, _ = text_hidden.shape
        T = z.size(1)
        fuse_len = min(T, max(L - 1, 0))
        if fuse_len <= 0:
            return text_hidden

        text_core = text_hidden[:, 1:1 + fuse_len, :]
        ids_core = z[:, :fuse_len, :]
        text_core = self.ids_to_text_ln(
            text_core + self.ids_to_text_drop(self.ids_to_text_alpha * ids_core)
        )

        if fuse_len == (L - 1):
            text_hidden = torch.cat([text_hidden[:, :1, :], text_core], dim=1)
        else:
            text_hidden = torch.cat(
                [text_hidden[:, :1, :], text_core, text_hidden[:, 1 + fuse_len:, :]], dim=1,
            )
        return text_hidden

    def _decode_text_hidden(self, memory, h, mem_mask, tgt_text):
        device = memory.device
        L_text = tgt_text.size(1)

        pad_mask = (tgt_text == self.ignore_index)
        tgt_mask = self._causal_mask(L_text, device=device)

        tgt_emb = self.text_embed(tgt_text)
        tgt_emb = self.text_pos_enc(tgt_emb)
        tgt_emb = self.text_norm(tgt_emb)
        text_hidden = self.text_decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory.transpose(0, 1),
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        ).transpose(0, 1)
        return text_hidden

    def _decode_text_logits_train(self, memory, h, mem_mask, tgt_text, h_ids=None):
        text_hidden = self._decode_text_hidden(memory, h, mem_mask, tgt_text)
        if self.use_ids2text_adapter and (h_ids is not None):
            text_hidden = self._fuse_text_hidden_with_ids(text_hidden, h_ids)
        return self.proj_text(text_hidden)

    def _greedy_text_decode(self, memory, h, mem_mask, h_ids=None):
        B = memory.size(0)
        device = memory.device
        tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        probs_text_steps = []

        for i in range(self.max_text_len + 1):
            pad_mask = (tgt == self.ignore_index)
            L = tgt.size(1)
            tgt_mask = self._causal_mask(L, device=device)

            tgt_emb = self.text_embed(tgt)
            tgt_emb = self.text_pos_enc(tgt_emb)
            tgt_emb = self.text_norm(tgt_emb)

            hidden = self.text_decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=memory.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
            ).transpose(0, 1)

            hidden_i = hidden[:, -1:, :]
            if self.use_ids2text_adapter and (h_ids is not None) and (i < h_ids.size(1)):
                ids_i = h_ids[:, i:i + 1, :]
                ids_i = ids_i.detach() if self.detach_ids_for_text else ids_i
                ids_i = self.ids_to_text_proj(ids_i)
                hidden_i = self.ids_to_text_ln(
                    hidden_i + self.ids_to_text_drop(self.ids_to_text_alpha * ids_i)
                )

            logits_i = self.proj_text(hidden_i)

            probs_text_steps.append(F.softmax(logits_i, dim=-1))

            if i < self.max_text_len:
                nxt = logits_i.squeeze(1).argmax(-1)
                tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                if (tgt == self.eos_id).any(dim=-1).all():
                    break

        probs_text = torch.cat(probs_text_steps, dim=1)

        pred_tokens = tgt[:, 1:]
        eos_mask = (pred_tokens == self.eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.zeros((B,), dtype=torch.long, device=device)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)
        text_len_pred = torch.where(
            has_eos, first_eos,
            torch.full((B,), pred_tokens.size(1), device=device, dtype=torch.long),
        )
        text_len_pred = torch.clamp(text_len_pred, min=0)
        return probs_text, tgt, text_len_pred

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def _debug_log_alphas(self):
        if not getattr(self, "debug_print_alpha", False):
            return

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        self._debug_step += 1
        interval = int(getattr(self, "debug_print_interval", 200))
        if interval <= 0 or (self._debug_step % interval != 0):
            return

        ids_local_alpha = None
        if self.ids_local_alpha is not None:
            ids_local_alpha = float(self.ids_local_alpha.detach().cpu().item())

        ids_to_text_alpha = None
        if self.ids_to_text_alpha is not None:
            ids_to_text_alpha = float(self.ids_to_text_alpha.detach().cpu().item())

        msg = f"[AlphaDebug] step={self._debug_step}"
        if ids_local_alpha is not None:
            msg += f" ids_local_alpha={ids_local_alpha:.6f}"
        if ids_to_text_alpha is not None:
            msg += f" ids_to_text_alpha={ids_to_text_alpha:.6f}"

        if (self.ids_local_alpha is not None) and (self.ids_local_alpha.grad is not None):
            msg += f" ids_local_grad={float(self.ids_local_alpha.grad.detach().cpu().item()):.6f}"
        if (self.ids_to_text_alpha is not None) and (self.ids_to_text_alpha.grad is not None):
            msg += f" ids_to_text_grad={float(self.ids_to_text_alpha.grad.detach().cpu().item()):.6f}"

        print(msg, flush=True)

    def forward_train(self, x, data):
        self._debug_log_alphas()
        memory, h, mem_mask = self._prep_memory(x)
        feat_local_ids, local_sz = self._prep_feat_local(x)
        if self.use_ids_local_fusion:
            assert feat_local_ids is not None and local_sz is not None, (
                "use_ids_local_fusion=True but feat_local/local_sz is missing from encoder output"
            )
        text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths = data[:4]
        B = memory.size(0)

        _, h_ids_for_ids, valid_char_mask_for_ids, max_chars_for_ids = self._build_ids_states(
            memory, mem_mask, text_lengths,
        )

        local_ctc_frames = None
        if self.use_ids_local_fusion and (max_chars_for_ids > 0):
            local_ctc_frames = self._build_local_ctc_frames(
                feat_local_ids=feat_local_ids,
                local_sz=local_sz,
                num_chars=max_chars_for_ids,
                lengths=text_lengths,
            )

        h_ids_for_text = None
        if self.use_ids2text_adapter:
            _, h_ids_for_text, _, _ = self._build_ids_states(
                memory, mem_mask, text_lengths,
            )

        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, :2 + max_text]
        logits_text = self._decode_text_logits_train(
            memory,
            h,
            mem_mask,
            tgt_text,
            h_ids=h_ids_for_text,
        )

        device = memory.device
        if max_chars_for_ids <= 0 or h_ids_for_ids is None:
            ids_ctc_loss = torch.tensor(0.0, device=device)
            grammar_penalty = torch.tensor(0.0, device=device)
            ids_decoded_valid = []
        else:
            ids_label_2d = per_char_ids_labels[:, :max_chars_for_ids, :]
            ids_len_2d = per_char_ids_lengths[:, :max_chars_for_ids]

            h_ids_valid = h_ids_for_ids[valid_char_mask_for_ids]
            ids_valid = ids_label_2d[valid_char_mask_for_ids]
            ids_len_valid = ids_len_2d[valid_char_mask_for_ids]

            local_frames_valid = None
            if local_ctc_frames is not None:
                local_frames_valid = local_ctc_frames[valid_char_mask_for_ids]

            ctc_logits = self._ctc_forward(
                h_ids_valid,
                local_frame_feat=local_frames_valid,
            )
            ids_ctc_loss = self._ctc_loss(ctc_logits, ids_valid, ids_len_valid)

            if self.grammar_penalty_weight > 0:
                grammar_penalty = self._ctc_grammar_penalty(ctc_logits)
            else:
                grammar_penalty = torch.tensor(0.0, device=device)

            with torch.no_grad():
                ids_decoded_valid = self._ctc_greedy_decode(ctc_logits)

        all_char_ids_train: List[List[List[int]]] = []
        idx = 0
        for b in range(B):
            n = min(int(text_lengths[b].item()), max_chars_for_ids)
            all_char_ids_train.append(ids_decoded_valid[idx:idx + n])
            idx += n

        if h_ids_for_ids is None:
            char_feat_out = memory.new_zeros((B, 0, 1, memory.size(-1)))
        else:
            char_feat_out = h_ids_for_ids.unsqueeze(2)

        return logits_text, (ids_ctc_loss, all_char_ids_train), char_feat_out, grammar_penalty, max_text

    def forward_test(self, x):
        memory, h, mem_mask = self._prep_memory(x)
        feat_local_ids, local_sz = self._prep_feat_local(x)
        if self.use_ids_local_fusion:
            assert feat_local_ids is not None and local_sz is not None, (
                "use_ids_local_fusion=True but feat_local/local_sz is missing from encoder output"
            )
        B = memory.size(0)
        device = memory.device

        if self.use_ids2text_adapter:
            lengths_prior = torch.full((B,), self.max_text_len, dtype=torch.long, device=device)
            _, h_ids_text, _, _ = self._build_ids_states(
                memory, mem_mask, lengths_prior,
            )
            probs_text, tgt, text_len_pred = self._greedy_text_decode(
                memory,
                h,
                mem_mask,
                h_ids=h_ids_text,
            )
        else:
            probs_text, tgt, text_len_pred = self._greedy_text_decode(
                memory, h, mem_mask, h_ids=None,
            )

        char_slots, h_ids, valid_char_mask, max_chars = self._build_ids_states(
            memory, mem_mask, text_len_pred,
        )

        local_ctc_frames = None
        if self.use_ids_local_fusion and (max_chars > 0):
            local_ctc_frames = self._build_local_ctc_frames(
                feat_local_ids=feat_local_ids,
                local_sz=local_sz,
                num_chars=max_chars,
                lengths=text_len_pred,
            )

        if max_chars > 0 and h_ids is not None:
            h_ids_valid = h_ids[valid_char_mask]
            local_frames_valid = None
            if local_ctc_frames is not None:
                local_frames_valid = local_ctc_frames[valid_char_mask]

            ctc_logits = self._ctc_forward(
                h_ids_valid,
                local_frame_feat=local_frames_valid,
            )
            if self.constrained_ctc_decode:
                all_ids_valid = self._ctc_constrained_decode(ctc_logits)
            else:
                all_ids_valid = self._ctc_greedy_decode(ctc_logits)

            all_char_ids = []
            idx = 0
            for b in range(B):
                n = min(int(text_len_pred[b].item()), max_chars)
                all_char_ids.append(all_ids_valid[idx:idx + n])
                idx += n
        else:
            all_char_ids = [[] for _ in range(B)]

        return probs_text, all_char_ids, text_len_pred

    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        return self.forward_test(x)
