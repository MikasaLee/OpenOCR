"""
LocalGlobalVerifyDecoder — Local-Global IDS Verifier.

Core architecture:

1. **Dual-view encoder input**: IDS branch accesses pre-collapse 2D local
   features for stroke/structural detail, while text branch uses height-pooled
   1D memory (as before).

2. **Soft-span char locator**: Each character query predicts a soft spatial
   span (center + width) along the width axis, then performs differentiable
   RoI attention pooling on 2D features to get truly local character features.

3. **Real local-token CTC**: IDS CTC operates on real local tokens extracted
   from the 2D feature map within each character's soft span.

4. **Global line-context fusion**: Each character's local features are fused
   with global 1D line context via gated cross-attention, providing the key
   "line-aware" signal for IDS verification.
"""

from typing import Optional, List, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)


class SoftSpanCharLocator(nn.Module):
    """Character locator that predicts soft spatial spans on 2D feature map.

    Each learnable position query attends to 1D memory to produce a character
    feature, center position, and width. The center/width define a soft span
    used for RoI attention pooling on 2D features.
    """

    def __init__(self, d_model, max_text_length, nhead, dim_feedforward,
                 dropout, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.max_text_length = max_text_length

        self.pos_queries = nn.Parameter(
            torch.randn(max_text_length, d_model) * 0.02
        )

        loc_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.locator = nn.TransformerDecoder(loc_layer, num_layers=num_layers)
        self.loc_norm = nn.LayerNorm(d_model)

        # Span prediction: center (sigmoid) and width (softplus)
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),  # [center, width]
        )

    def forward(self, memory_1d, mem_mask, num_chars, lengths=None):
        """
        Args:
            memory_1d: [B, T, C] 1D encoder memory
            mem_mask: [B, T] bool padding mask or None
            num_chars: int, max number of characters
            lengths: [B] character counts for masking
        Returns:
            char_feat: [B, num_chars, C] per-char features from 1D memory
            spans: [B, num_chars, 2] (center, width) normalized to [0, 1]
        """
        B = memory_1d.size(0)
        queries = self.pos_queries[:num_chars].unsqueeze(0).expand(B, -1, -1)

        tgt_kpm = None
        if lengths is not None:
            idx = torch.arange(num_chars, device=memory_1d.device).unsqueeze(0)
            tgt_kpm = idx >= lengths.unsqueeze(1)  # [B, num_chars] True=pad

        char_feat = self.locator(
            tgt=queries,
            memory=memory_1d,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_mask,
        )
        char_feat = self.loc_norm(char_feat)  # [B, num_chars, C]

        span_raw = self.span_head(char_feat)  # [B, num_chars, 2]
        center = torch.sigmoid(span_raw[..., 0])  # (0, 1)
        width = F.softplus(span_raw[..., 1]) * 0.3 + 0.05  # min ~0.05

        spans = torch.stack([center, width], dim=-1)  # [B, num_chars, 2]
        return char_feat, spans


class SoftRoIPool2D(nn.Module):
    """Differentiable RoI pooling using 2D Gaussian attention over H*W grid.

    Given a 2D feature map [B, H, W, C] and per-char soft spans (center, width)
    along the width axis, produces a fixed-length local token sequence per
    character. Each of the K sample points attends to the full H*W grid with a
    2D Gaussian kernel, preserving vertical structural information.

    The height dimension uses a learned per-head vertical attention profile,
    allowing the model to focus on top/middle/bottom of each character span.
    """

    def __init__(self, d_model, num_local_tokens=8, num_h_heads=4):
        super().__init__()
        self.num_local_tokens = num_local_tokens
        self.num_h_heads = num_h_heads
        self.proj = nn.Linear(d_model, d_model)

        # Learnable vertical attention: num_h_heads profiles over H
        self.h_profile = nn.Parameter(torch.randn(num_h_heads, 1) * 0.1)
        self.h_width = nn.Parameter(torch.ones(num_h_heads, 1) * 0.3)
        self.k_to_h = nn.Linear(d_model, num_h_heads)

    def forward(self, feat_2d, spans, char_feat_1d=None):
        """
        Args:
            feat_2d: [B, H, W, C] pre-collapse 2D features (already projected)
            spans: [B, N, 2] (center, width) for N character slots
            char_feat_1d: [B, N, C] char features for vertical head selection (optional)
        Returns:
            local_tokens: [B, N, K, C] local token sequences per character
        """
        B, H, W, C = feat_2d.shape
        N = spans.size(1)
        K = self.num_local_tokens
        device = feat_2d.device

        centers = spans[:, :, 0]  # [B, N]
        widths = spans[:, :, 1]   # [B, N]

        # === Width attention (Gaussian over W) ===
        offsets = torch.linspace(-0.5, 0.5, K, device=device)  # [K]
        sample_pos = (
            centers.unsqueeze(-1)
            + offsets.unsqueeze(0).unsqueeze(0) * widths.unsqueeze(-1)
        )  # [B, N, K]
        sample_pos = sample_pos.clamp(0.0, 1.0)
        w_coords = sample_pos * (W - 1)  # [B, N, K]

        w_grid = torch.arange(W, device=device, dtype=torch.float)  # [W]
        sigma_w = (widths / K * (W - 1)).clamp(min=0.5)  # [B, N]

        dist_w = (w_coords.unsqueeze(-1) - w_grid.view(1, 1, 1, W)) ** 2  # [B, N, K, W]
        attn_w = torch.exp(-dist_w / (2 * sigma_w.unsqueeze(-1).unsqueeze(-1) ** 2 + 1e-8))
        attn_w = attn_w / (attn_w.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N, K, W]

        # === Height attention via learnable vertical profiles ===
        h_grid = torch.linspace(0, 1, H, device=device)  # [H]
        h_centers = torch.sigmoid(self.h_profile)  # [num_h_heads, 1]
        h_spreads = F.softplus(self.h_width).clamp(min=0.1)  # [num_h_heads, 1]

        dist_h = (h_grid.unsqueeze(0) - h_centers) ** 2  # [num_h_heads, H]
        attn_h_heads = torch.exp(-dist_h / (2 * h_spreads ** 2 + 1e-8))
        attn_h_heads = attn_h_heads / (attn_h_heads.sum(dim=-1, keepdim=True) + 1e-8)
        # [num_h_heads, H]

        # Mix vertical heads per K token
        if char_feat_1d is not None:
            h_weights = F.softmax(self.k_to_h(char_feat_1d), dim=-1)  # [B, N, num_h_heads]
            h_weights = h_weights.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, num_h_heads]
        else:
            h_weights = torch.ones(B, N, K, self.num_h_heads, device=device) / self.num_h_heads

        # attn_h: [B, N, K, H]
        attn_h = torch.einsum('bnkm,mh->bnkh', h_weights, attn_h_heads)
        attn_h = attn_h / (attn_h.sum(dim=-1, keepdim=True) + 1e-8)

        # === 2D attention: outer product of H and W attentions ===
        attn_2d = attn_h.unsqueeze(-1) * attn_w.unsqueeze(-2)  # [B, N, K, H, W]

        # === Pool from 2D features ===
        feat_flat = feat_2d.reshape(B, H * W, C)
        attn_flat = attn_2d.reshape(B, N, K, H * W)

        local_tokens = torch.einsum('bnkp,bpc->bnkc', attn_flat, feat_flat)
        local_tokens = self.proj(local_tokens)

        return local_tokens  # [B, N, K, C]


class GatedCrossAttentionFusion(nn.Module):
    """Gated cross-attention to fuse global line context into local features."""

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, local_feat, global_context, global_mask=None):
        """
        Args:
            local_feat: [N, K, C] local token sequences per valid char
            global_context: [N, T, C] global 1D line context (expanded per char)
            global_mask: [N, T] padding mask for global context or None
        Returns:
            fused: [N, K, C]
        """
        attended, _ = self.cross_attn(
            query=local_feat,
            key=global_context,
            value=global_context,
            key_padding_mask=global_mask,
        )
        attended = self.norm(attended + local_feat)

        gate_input = torch.cat([local_feat, attended], dim=-1)
        g = self.gate(gate_input)
        return g * attended + (1 - g) * local_feat


class LocalGlobalVerifyDecoder(nn.Module):

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
        # Grammar constraints
        constrained_ctc_decode: bool = True,
        ids_syntax_max_need: int = 64,
        grammar_penalty_weight: float = 0.1,
        # 2D feature channels from encoder
        feat_2d_channels: int = 384,
        # Soft span and local tokens
        num_local_tokens: int = 8,
        num_char_locator_layers: int = 2,
        # CTC frame expansion and encoder
        ctc_frames_per_char: int = 32,
        num_ctc_encoder_layers: int = 1,
        # Global context fusion
        global_fusion_nhead: int = 4,
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
            return count + 4  # <pad><sos><eos><unk>

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

        self.num_local_tokens = num_local_tokens
        self.ctc_frames = int(ctc_frames_per_char)

        d_model = in_channels

        # ===== 2D -> d_model projection =====
        self.feat_2d_proj = nn.Linear(feat_2d_channels, d_model)
        self.feat_2d_proj_norm = nn.LayerNorm(d_model)

        # ===== Text branch (AR) =====
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

        # ===== Soft-span char locator =====
        self.char_locator = SoftSpanCharLocator(
            d_model, max_text_length, nhead, dim_feedforward, dropout,
            num_layers=num_char_locator_layers,
        )

        # ===== Soft RoI pooling on 2D features =====
        self.soft_roi_pool = SoftRoIPool2D(d_model, num_local_tokens, num_h_heads=4)

        # ===== Global line-context fusion =====
        self.global_fusion = GatedCrossAttentionFusion(
            d_model, nhead=global_fusion_nhead, dropout=dropout,
        )

        # ===== CTC: pool K local tokens → 1 vector, then expand to ctc_frames =====
        self.ctc_blank_id = 0
        ctc_frames = self.ctc_frames
        # Attention pooling: learnable query aggregates K tokens into 1 vector
        self.ctc_pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ctc_pool_attn = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.ctc_pool_norm = nn.LayerNorm(d_model)
        # 1 vector → ctc_frames frames (same as SOTA)
        self.frame_expand = nn.Linear(d_model, ctc_frames * d_model)
        self.frame_pos = nn.Parameter(torch.randn(ctc_frames, d_model) * 0.02)

        ctc_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.ctc_frame_encoder = nn.TransformerEncoder(
            ctc_enc_layer, num_layers=num_ctc_encoder_layers,
        )
        self.ctc_proj = nn.Linear(d_model, ids_vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.ctc_blank_id, reduction='mean', zero_infinity=True,
        )

        # ===== IDS vocab =====
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

        # ===== IDC operator sets =====
        self._ids_unary_ops: Set[int] = set()
        self._ids_binary_ops: Set[int] = set()
        self._ids_trinary_ops: Set[int] = set()
        self._ids_leaf_ids: Set[int] = set()
        self._ids_operator_ids: Set[int] = set()
        self._init_ids_operator_sets()

        # ===== char_to_ids mapping =====
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

    # -------------------- Helpers --------------------
    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), True, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_dual_memory(self, x):
        """Extract dual-view features from encoder output dict."""
        if isinstance(x, dict):
            feat_1d = x['feat_1d']   # [B, W, C_out]
            feat_2d = x['feat_2d']   # [B, H, W, C_2d]
            sz_2d = x['sz_2d']       # (H, W)
            return feat_1d, feat_2d, sz_2d
        # Fallback: plain tensor -> treat as 1D only (backward compat)
        return x, None, None

    def _build_global_context(self, memory_1d, valid_mask):
        """Expand global context so each valid char has access to full line.

        Args:
            memory_1d: [B, T, C]
            valid_mask: [B, max_chars] bool, True=valid
        Returns:
            global_ctx: [N_valid, T, C]
        """
        counts = valid_mask.sum(dim=1)  # [B]
        global_ctx = torch.repeat_interleave(memory_1d, counts, dim=0)
        return global_ctx  # [N_valid, T, C]

    # -------------------- IDC operator sets --------------------
    def _init_ids_operator_sets(self):
        if self.ids_token2id is None:
            return
        unary = ["\u2FBE", "\u2FBF"]
        trinary = ["\u2FF2", "\u2FF3"]
        binary = ["\u2FF0", "\u2FF1", "\u2FF4", "\u2FF5", "\u2FF6",
                  "\u2FF7", "\u2FF8", "\u2FF9", "\u2FFA", "\u2FFB",
                  "\u2FFC", "\u2FFD", "\u33EF"]

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

    # -------------------- Per-char CTC --------------------
    def _ctc_forward(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """Pool K local tokens into 1 vector, expand to ctc_frames, encode.

        Args: local_tokens [N, K, d_model]
        Returns: logits [N, ctc_frames, V_ids]
        """
        N, K, d = local_tokens.shape
        # Attention pooling: [N, 1, d] query attends to [N, K, d] tokens
        query = self.ctc_pool_query.expand(N, -1, -1)       # [N, 1, d]
        pooled, _ = self.ctc_pool_attn(query, local_tokens, local_tokens)
        pooled = self.ctc_pool_norm(pooled.squeeze(1))       # [N, d]
        # Expand 1 vector → ctc_frames frames (full freedom, same as SOTA)
        frames = self.frame_expand(pooled)                   # [N, ctc_frames * d]
        frames = frames.view(N, self.ctc_frames, d)          # [N, ctc_frames, d]
        frames = frames + self.frame_pos.unsqueeze(0)
        frames = self.ctc_frame_encoder(frames)
        return self.ctc_proj(frames)

    def _ctc_loss(self, ctc_logits, ids_targets, ids_lengths):
        """CTC loss for N valid characters."""
        N, K, V = ctc_logits.shape
        device = ctc_logits.device
        if N == 0:
            return torch.tensor(0.0, device=device)

        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)  # [K, N, V]
        max_tgt_len = int(ids_lengths.max().item())
        targets = ids_targets[:, 1:1 + max_tgt_len].contiguous()  # skip BOS
        input_lengths = torch.full((N,), K, dtype=torch.long, device=device)
        return self.ctc_loss_fn(log_probs, targets, input_lengths, ids_lengths.long())

    @torch.no_grad()
    def _ctc_greedy_decode(self, ctc_logits):
        """CTC greedy decode: argmax -> collapse duplicates -> remove blank."""
        pred_ids = ctc_logits.argmax(dim=-1)
        results = []
        for i in range(pred_ids.size(0)):
            tokens = pred_ids[i].tolist()
            collapsed = []
            prev = None
            for t in tokens:
                if t != prev:
                    collapsed.append(t)
                prev = t
            results.append([t for t in collapsed if t not in (0, 1, 2)])
        return results

    def _is_ids_token_legal(self, token_id, need):
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
        """Grammar-constrained CTC greedy decode with terminal legality check."""
        N, K, V = ctc_logits.shape
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

            # Terminal legality: trim trailing tokens until need == 0
            while need != 0 and decoded:
                last = decoded[-1]
                decoded.pop()
                if last in self._ids_binary_ops:
                    need -= 1
                elif last in self._ids_trinary_ops:
                    need -= 2
                elif last in self._ids_unary_ops:
                    pass
                elif last not in (0, 1, 2, 3):
                    need += 1

            results.append(decoded)
        return results

    def _ctc_grammar_penalty(self, ctc_logits):
        """Differentiable grammar penalty on CTC frame logits (training).

        Includes terminal legality: penalizes sequences that end with need != 0.
        """
        N, K, V = ctc_logits.shape
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

        # Terminal legality penalty (differentiable via soft probabilities).
        # Compute expected need-delta per frame from probs so gradient flows.
        soft_delta = torch.zeros(N, K, device=device)
        for op_id in self._ids_binary_ops:
            soft_delta = soft_delta + probs[:, :, op_id]       # binary: +1
        for op_id in self._ids_trinary_ops:
            soft_delta = soft_delta + probs[:, :, op_id] * 2   # trinary: +2
        # Leaf prob = 1 - P(special) - P(all_operators)
        special_prob = probs[:, :, 0] + probs[:, :, 1] + probs[:, :, 2] + probs[:, :, 3]
        op_prob = torch.zeros(N, K, device=device)
        for op_id in self._ids_operator_ids:
            op_prob = op_prob + probs[:, :, op_id]
        leaf_prob = (1.0 - special_prob - op_prob).clamp(min=0)
        soft_delta = soft_delta - leaf_prob                     # leaf: -1

        soft_final_need = 1.0 + soft_delta.sum(dim=1)  # [N]
        terminal_penalty = soft_final_need.abs().mean()

        return frame_penalty + terminal_penalty

    # -------------------- Forward (train) --------------------
    def forward_train(self, x, data):
        memory_1d, feat_2d, sz_2d = self._prep_dual_memory(x)
        text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths = data[:4]

        B = memory_1d.size(0)
        device = memory_1d.device
        h_1d = sz_2d[0] if sz_2d is not None else 2
        mem_mask = None

        # Project 2D features to d_model
        if feat_2d is not None:
            feat_2d_proj = self.feat_2d_proj_norm(self.feat_2d_proj(feat_2d))
        else:
            feat_2d_proj = None

        # ---- 1. Text AR (teacher forcing) ----
        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, :2 + max_text]
        pad_mask = (tgt_text == self.ignore_index)
        L_text = tgt_text.size(1)
        tgt_mask = self._causal_mask(L_text, device=device)
        tgt_emb = self.text_embed(tgt_text)
        tgt_emb = self.text_pos_enc(tgt_emb)
        tgt_emb = self.text_norm(tgt_emb)
        text_hidden = self.text_decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory_1d.transpose(0, 1),
            height=h_1d,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        ).transpose(0, 1)  # [B, 2+max_text, C]
        logits_text = self.proj_text(text_hidden)

        # ---- 2. Soft-span char locator ----
        max_chars = int(text_lengths.max().item())
        max_chars = min(max_chars, self.max_text_len)

        char_feat_1d, spans = self.char_locator(
            memory_1d, mem_mask, max_chars, lengths=text_lengths,
        )  # char_feat_1d: [B, max_chars, C], spans: [B, max_chars, 2]

        # ---- 3. Soft RoI pooling on 2D features ----
        valid_char_mask = torch.arange(max_chars, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
        # [B, max_chars] bool

        if feat_2d_proj is not None:
            all_local_tokens = self.soft_roi_pool(feat_2d_proj, spans, char_feat_1d=char_feat_1d)
            # [B, max_chars, K, C]
            local_tokens_valid = all_local_tokens[valid_char_mask]  # [N_valid, K, C]
        else:
            local_tokens_valid = char_feat_1d[valid_char_mask].unsqueeze(1).expand(
                -1, self.num_local_tokens, -1
            )  # [N_valid, K, C]

        # ---- 4. Global line-context fusion ----
        global_context = self._build_global_context(memory_1d, valid_char_mask)
        # [N_valid, T, C]
        fused_tokens = self.global_fusion(local_tokens_valid, global_context)
        # [N_valid, K, C]

        # ---- 5. Per-char CTC on fused local tokens ----
        ctc_logits = self._ctc_forward(fused_tokens)  # [N_valid, K, V_ids]

        # CTC loss uses GT IDS targets
        ids_label_2d = per_char_ids_labels[:, :max_chars, :]
        ids_len_2d = per_char_ids_lengths[:, :max_chars]
        ids_valid = ids_label_2d[valid_char_mask]
        ids_len_valid = ids_len_2d[valid_char_mask]

        ids_ctc_loss = self._ctc_loss(ctc_logits, ids_valid, ids_len_valid)

        # Grammar penalty
        if self.grammar_penalty_weight > 0:
            grammar_penalty = self._ctc_grammar_penalty(ctc_logits)
        else:
            grammar_penalty = torch.tensor(0.0, device=device)

        # Greedy decode for postprocess
        with torch.no_grad():
            all_ids_decoded = self._ctc_greedy_decode(ctc_logits)

        all_char_ids_train: List[List[List[int]]] = []
        idx = 0
        for b in range(B):
            n = min(int(text_lengths[b].item()), max_chars)
            all_char_ids_train.append(all_ids_decoded[idx:idx + n])
            idx += n

        return (
            logits_text,                            # [0] for postprocess & text CE
            (ids_ctc_loss, all_char_ids_train),     # [1] for postprocess & ids loss
            char_feat_1d,                           # [2] char features (unused by loss)
            grammar_penalty,                        # [3] grammar penalty scalar
            max_text,                               # [4] max text length
        )

    # -------------------- Forward (test) --------------------
    def forward_test(self, x):
        memory_1d, feat_2d, sz_2d = self._prep_dual_memory(x)
        B, Tenc, _ = memory_1d.shape
        device = memory_1d.device
        h_1d = sz_2d[0] if sz_2d is not None else 2
        mem_mask = None

        if feat_2d is not None:
            feat_2d_proj = self.feat_2d_proj_norm(self.feat_2d_proj(feat_2d))
        else:
            feat_2d_proj = None

        # ---- 1. Text greedy AR ----
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
                memory=memory_1d.transpose(0, 1),
                height=h_1d,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
            ).transpose(0, 1)

            logits_i = self.proj_text(hidden[:, -1:, :])
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

        # ---- 2. Char locator + IDS verification ----
        max_chars = min(int(text_len_pred.max().item()), tgt.size(1) - 1)

        if max_chars > 0:
            char_feat_1d, spans = self.char_locator(
                memory_1d, mem_mask, max_chars, lengths=text_len_pred,
            )

            valid_char_mask = torch.arange(max_chars, device=device).unsqueeze(0) < text_len_pred.unsqueeze(1)

            if feat_2d_proj is not None:
                all_local_tokens = self.soft_roi_pool(feat_2d_proj, spans, char_feat_1d=char_feat_1d)
                local_tokens_valid = all_local_tokens[valid_char_mask]
            else:
                local_tokens_valid = char_feat_1d[valid_char_mask].unsqueeze(1).expand(
                    -1, self.num_local_tokens, -1
                )

            global_context = self._build_global_context(memory_1d, valid_char_mask)
            fused_tokens = self.global_fusion(local_tokens_valid, global_context)

            ctc_logits = self._ctc_forward(fused_tokens)
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

    # -------------------- Forward --------------------
    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        return self.forward_test(x)
