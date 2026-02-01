from typing import Optional, Tuple, List, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)


class TextIDSTreeDecoderv3(nn.Module):
    """
    CORE (no beam):
      - Text: autoregressive TransformerDecoder
      - IDS:  CTC head with TransformerEncoder refinement
      - IDS legality (decode-time, optional): HARD IDS syntax constraint (arity/need) in constrained greedy CTC
      - Tree supervision (train-time): sim_ids [B,L,L] from token<->memory cross-attn + simple structural scorer

    Inference IDS supports:
      - "greedy": per-frame argmax CTC (no syntax constraint)
      - "constrained": constrained greedy CTC with IDS syntax legality
    """

    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        text_vocab_path: Optional[str] = None,
        ids_vocab_path: Optional[str] = None,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 64,
        cross_coverage: bool = False,
        self_coverage: bool = False,
        max_text_length: int = 25,
        max_ids_length: int = 50,
        use_space_char: bool = False,

        # IDS decode mode in test
        ids_decode_mode: str = "constrained",  # "greedy" | "constrained"
        # IDS legality, only used when ids_decode_mode="constrained"
        ids_use_syntax_constraint: bool = True,
        ids_syntax_max_need: int = 64,
        ids_use_text_length_constraint: bool = True,  # if use_space_char: align #segments with text length
        ids_greedy_topk: int = 40,
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

        # specials
        self.ignore_index = 0  # also used as CTC blank
        self.blank_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.max_text_len = int(max_text_length)
        self.max_ids_len = int(max_ids_length)
        self.use_space_char = bool(use_space_char)

        # legality
        self.ids_use_syntax_constraint = bool(ids_use_syntax_constraint)
        self.ids_syntax_max_need = int(ids_syntax_max_need)
        self.ids_use_text_length_constraint = bool(ids_use_text_length_constraint)
        self.ids_greedy_topk = int(ids_greedy_topk)

        # default decode mode
        self.ids_decode_mode = str(ids_decode_mode)
        if self.ids_decode_mode not in ("greedy", "constrained"):
            raise ValueError(f"ids_decode_mode must be 'greedy' or 'constrained', got {self.ids_decode_mode}")

        d_model = in_channels

        # ===== text branch =====
        self.text_embed = nn.Sequential(nn.Embedding(text_vocab_size, d_model), nn.LayerNorm(d_model))
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

        # ===== IDS branch: CTC + TransformerEncoder =====
        self.ids_ctc_q_proj = nn.Linear(d_model, d_model)
        self.ids_ctc_pos_enc = WordPosEnc(d_model)
        self.ids_ctc_norm = nn.LayerNorm(d_model)
        self.ids_ctc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )
        self.proj_ids_ctc = nn.Linear(d_model, ids_vocab_size)

        # ===== Tree head (train-time supervision) =====
        self.ids_embed_for_tree = nn.Sequential(nn.Embedding(ids_vocab_size, d_model), nn.LayerNorm(d_model))
        self.ids_pos_enc_for_tree = WordPosEnc(d_model)
        self.ids_norm_for_tree = nn.LayerNorm(d_model)

        # token queries attend to visual memory -> token_feat
        self.tree_xattn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tree_ln = nn.LayerNorm(d_model)

        # simple structural scorer -> sim [B,L,L]
        self.struct_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.child_proj = nn.Linear(d_model, d_model)
        self.parent_proj = nn.Linear(d_model, d_model)
        self.vs = nn.Linear(d_model, 1, bias=True)

        # ===== ids vocab strings (for space/op ids) =====
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

        # ===== IDC operator id sets (for HARD syntax legality) =====
        self._ids_unary_ops: Set[int] = set()
        self._ids_binary_ops: Set[int] = set()
        self._ids_trinary_ops: Set[int] = set()
        self._init_ids_operator_sets()

    # ---------------- basic helpers ----------------
    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), True, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_memory(self, x) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
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

    # ---------------- text ----------------
    def _decode_text_seq(self, memory, h, mem_mask, tgt_ids):
        pad_mask = (tgt_ids == self.ignore_index)
        L = tgt_ids.size(1)
        tgt_mask = self._causal_mask(L, device=memory.device)

        tgt_emb = self.text_embed(tgt_ids)
        tgt_emb = self.text_pos_enc(tgt_emb)
        tgt_emb = self.text_norm(tgt_emb)

        out = self.text_decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory.transpose(0, 1),
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        ).transpose(0, 1)

        return self.proj_text(out)

    # ---------------- IDS CTC encoder ----------------
    def _decode_ids_ctc_seq(self, memory, mem_mask):
        x = self.ids_ctc_q_proj(memory)
        x = self.ids_ctc_pos_enc(x)
        x = self.ids_ctc_norm(x)
        feat = self.ids_ctc_encoder(x, src_key_padding_mask=mem_mask)
        return self.proj_ids_ctc(feat)  # [B,T,V]

    # ---------------- Tree sim (train-time) ----------------
    def _build_parent_allowed_mask(self, tgt_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        allowed[b,i,j]=True if j can be parent of i:
          - same segment (split by SPACE)
          - j < i
          - parent valid
          - segment start allow self-parent (i==j) to avoid empty
        """
        B, L = tgt_ids.shape
        device = tgt_ids.device
        valid = ~pad_mask

        if self.use_space_char and self.space_id is not None:
            is_space = (tgt_ids == self.space_id)
        else:
            is_space = torch.zeros((B, L), dtype=torch.bool, device=device)

        seg_id = torch.cumsum(is_space.long(), dim=1)  # [B,L]

        idx = torch.arange(L, device=device)
        i = idx[None, :, None]
        j = idx[None, None, :]

        lt = j < i
        same_seg = (seg_id[:, :, None] == seg_id[:, None, :])
        valid_parent = valid[:, None, :]

        allowed = lt & same_seg & valid_parent

        seg_start = torch.zeros((B, L), dtype=torch.bool, device=device)
        seg_start[:, 0] = True
        if L > 1:
            seg_start[:, 1:] = is_space[:, :-1]

        diag = torch.eye(L, dtype=torch.bool, device=device)[None, :, :]
        allowed = allowed | (diag & seg_start[:, :, None] & valid[:, :, None])

        allowed = allowed & valid[:, :, None]  # invalid child -> mask row
        return allowed

    def _struct_sim(self, feat: torch.Tensor, pad_mask: torch.Tensor, parent_allowed: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.struct_enc(feat, src_key_padding_mask=pad_mask)
        child = self.child_proj(feat)
        parent = self.parent_proj(feat)
        M = F.relu(child.unsqueeze(2) + parent.unsqueeze(1))
        sim = self.vs(M).squeeze(-1)  # [B,L,L]

        neg = -1e4 if sim.dtype in (torch.float16, torch.bfloat16) else -1e9
        sim = sim.masked_fill(pad_mask[:, None, :], neg)  # invalid parent cols
        sim = sim.masked_fill(pad_mask[:, :, None], neg)  # invalid child rows
        if parent_allowed is not None:
            sim = sim.masked_fill(~parent_allowed, neg)
        return sim

    # ---------------- IDS operator sets ----------------
    def _init_ids_operator_sets(self):
        if self.ids_token2id is None:
            return
        unary = ["⿾", "⿿"]
        trinary = ["⿲", "⿳"]
        binary = ["⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻", "⿼", "⿽", "㇯"]

        for s in unary:
            if s in self.ids_token2id:
                self._ids_unary_ops.add(self.ids_token2id[s])
        for s in trinary:
            if s in self.ids_token2id:
                self._ids_trinary_ops.add(self.ids_token2id[s])
        for s in binary:
            if s in self.ids_token2id:
                self._ids_binary_ops.add(self.ids_token2id[s])

    # ---------------- HARD IDS syntax constraint (arity/need) ----------------
    def _syntax_step(
        self,
        need: int,
        token_id: int,
        last_emit: Optional[int],
        seg_count: int,
        target_segs: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        if token_id in (self.bos_id, self.eos_id):
            return None

        # space
        if self.use_space_char and self.space_id is not None and token_id == self.space_id:
            if last_emit == self.space_id:
                return None
            if self.ids_use_syntax_constraint and need != 0:
                return None
            if target_segs is not None and seg_count >= target_segs:
                return None
            return 1, seg_count + 1

        if not self.ids_use_syntax_constraint:
            return need, seg_count

        if need == 0:
            return None

        if token_id in self._ids_unary_ops:
            need2 = need
        elif token_id in self._ids_binary_ops:
            need2 = need + 1
        elif token_id in self._ids_trinary_ops:
            need2 = need + 2
        else:
            need2 = need - 1

        if need2 < 0 or need2 > self.ids_syntax_max_need:
            return None
        return need2, seg_count

    # ---------------- plain greedy CTC path ----------------
    def _ctc_frame_greedy_path(
        self,
        logits: torch.Tensor,                 # [T,V]
        mem_mask_t: Optional[torch.Tensor],   # [T] bool (True=pad)
    ) -> List[int]:
        path = torch.argmax(logits, dim=-1)  # [T]
        if mem_mask_t is not None:
            path = path.clone()
            path[mem_mask_t.bool()] = int(self.blank_id)
        return path.tolist()

    # ---------------- constrained greedy CTC path ----------------
    def _ctc_constrained_greedy_path(
        self,
        logp: torch.Tensor,                 # [T,V] log-prob
        mem_mask_t: Optional[torch.Tensor], # [T] bool (True=pad)
        target_segs: Optional[int],
    ) -> List[int]:
        T, V = logp.shape
        blank = self.blank_id
        k = max(5, min(self.ids_greedy_topk, V))

        # top-k candidate ids per frame, ordered by prob
        _, topi = torch.topk(logp, k=k, dim=-1)  # [T,k]

        prev_frame: Optional[int] = None
        last_emit: Optional[int] = None
        need, seg = 1, 1

        path: List[int] = []

        for t in range(T):
            # pad frame -> force blank
            if mem_mask_t is not None and bool(mem_mask_t[t].item()):
                path.append(blank)
                prev_frame = blank
                continue

            cand = topi[t].tolist()
            if blank not in cand:
                cand.append(blank)

            picked = blank
            picked_need, picked_seg = need, seg

            for c in cand:
                c = int(c)

                if c == blank:
                    picked = c
                    picked_need, picked_seg = need, seg
                    break

                emit_now = (c != prev_frame)
                if not emit_now:
                    # repeated token frame -> collapses away, safe
                    picked = c
                    picked_need, picked_seg = need, seg
                    break

                step = self._syntax_step(need, c, last_emit, seg, target_segs)
                if step is None:
                    continue

                picked = c
                picked_need, picked_seg = step
                break

            path.append(picked)
            if picked != blank and (picked != prev_frame):
                last_emit = picked
                need, seg = picked_need, picked_seg
            prev_frame = picked

        return path

    # ---------------- forward ----------------
    def forward_train(self, x, data):
        if isinstance(x, dict) and "text" in x and "ids" in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x["text"])
            mem_ids, _, mem_mask_ids = self._prep_memory(x["ids"])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, _, mem_mask_ids = mem_text, h_text, mem_mask_text

        # data: expected [text_labels, text_lengths, ids_ctc_labels, ids_ctc_lengths]
        # optionally an extra `tree_parents_ctc_label` may be provided.
        if len(data) == 5:
            text_labels, text_lengths, ids_labels, ids_lengths, tree_parents_ctc_label = data
        else:
            text_labels, text_lengths, ids_labels, ids_lengths = data
            tree_parents_ctc_label = None  # (kept for compatibility, computed outside)

        # text logits
        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, : 2 + max_text]
        logits_text = self._decode_text_seq(mem_text, h_text, mem_mask_text, tgt_text)

        # ids logits (CTC)
        logits_ids_ctc = self._decode_ids_ctc_seq(mem_ids, mem_mask_ids)

        # ----- tree sim (for tree loss) -----
        max_ids = int(ids_lengths.max().item())
        tgt_ids = ids_labels[:, :max_ids]  # [B,L]

        pad = (tgt_ids == self.ignore_index)
        if self.use_space_char and self.space_id is not None:
            pad = pad | (tgt_ids == self.space_id)

        # token queries (GT ids) -> attend visual memory -> token_feat
        q = self.ids_embed_for_tree(tgt_ids)
        q = self.ids_pos_enc_for_tree(q)
        q = self.ids_norm_for_tree(q)

        mem_fp32 = mem_ids.float()
        q_fp32 = q.float()
        attn_out, _ = self.tree_xattn(q_fp32, mem_fp32, mem_fp32, key_padding_mask=mem_mask_ids)
        token_feat = self.tree_ln(q_fp32 + attn_out)  # [B,L,C]

        parent_allowed = self._build_parent_allowed_mask(tgt_ids, pad)
        sim_ids = self._struct_sim(token_feat, pad, parent_allowed=parent_allowed).float()  # [B,L,L]

        return logits_text, logits_ids_ctc, sim_ids

    def forward_test(self, x, ids_decode_mode: Optional[str] = None):
        if isinstance(x, dict) and "text" in x and "ids" in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x["text"])
            mem_ids, _, mem_mask_ids = self._prep_memory(x["ids"])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, _, mem_mask_ids = mem_text, h_text, mem_mask_text

        B, Tenc, _ = mem_text.shape
        device = mem_text.device

        # ---- text greedy ----
        tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        probs_text_steps = []
        for i in range(self.max_text_len + 1):
            logits = self._decode_text_seq(mem_text, h_text, mem_mask_text, tgt)
            p_i = logits[:, -1:, :]
            probs_text_steps.append(F.softmax(p_i, -1))
            if i < self.max_text_len:
                nxt = p_i.squeeze(1).argmax(-1)
                tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                if (tgt == self.eos_id).any(dim=-1).all():
                    break
        probs_text = torch.cat(probs_text_steps, dim=1)

        # predicted text lengths -> target segments
        pred = tgt[:, 1:]
        Lp = pred.size(1)
        eos_mask = (pred == self.eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.zeros((B,), dtype=torch.long, device=device)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)
        text_len_pred = torch.where(has_eos, first_eos, torch.full((B,), Lp, device=device, dtype=torch.long))
        text_len_pred = torch.clamp(text_len_pred, min=1)

        # ---- IDS decode (greedy or constrained) ----
        logits_ids_ctc = self._decode_ids_ctc_seq(mem_ids, mem_mask_ids)  # [B,T,V]
        V = logits_ids_ctc.size(-1)

        mode = self.ids_decode_mode if ids_decode_mode is None else str(ids_decode_mode)
        if mode not in ("greedy", "constrained"):
            raise ValueError(f"ids_decode_mode must be 'greedy' or 'constrained', got {mode}")

        probs_ids = torch.zeros((B, Tenc, V), dtype=torch.float16, device=device)

        logp = None
        if mode == "constrained":
            logp = F.log_softmax(logits_ids_ctc.float(), dim=-1)  # [B,T,V]

        for b in range(B):
            mem_mask_t = mem_mask_ids[b] if mem_mask_ids is not None else None

            if mode == "greedy":
                path = self._ctc_frame_greedy_path(logits_ids_ctc[b], mem_mask_t)
            else:
                target_segs = None
                if self.use_space_char and self.ids_use_text_length_constraint:
                    target_segs = int(text_len_pred[b].item())
                path = self._ctc_constrained_greedy_path(logp[b], mem_mask_t, target_segs)

            for t_idx in range(min(Tenc, len(path))):
                tok = int(path[t_idx])
                tok = max(0, min(V - 1, tok))
                probs_ids[b, t_idx, tok] = 1.0

        return probs_text, probs_ids

    def forward(self, x, data=None, ids_decode_mode: Optional[str] = None):
        if self.training:
            return self.forward_train(x, data)
        return self.forward_test(x, ids_decode_mode=ids_decode_mode)


# python -m openrec.modeling.decoders.text_ids_tree_decoderv3
if __name__ == "__main__":
    import sys
    import os

    print("Initializing TextIDSTreeDecoder Output Check...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Try to locate project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Project root: {project_root}")

    # Real Configs
    d_model = 512
    nhead = 4  # Reduced for speed
    num_layers = 1
    bs = 2
    max_text_len = 15
    max_ids_len = 100

    # Use real vocab files
    text_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")

    if not os.path.exists(text_vocab_path):
        print(f"Error: Text vocab not found at {text_vocab_path}")
        sys.exit(1)
    if not os.path.exists(ids_vocab_path):
        print(f"Error: IDS vocab not found at {ids_vocab_path}")
        sys.exit(1)

    try:
        decoder = TextIDSTreeDecoderv3(
            in_channels=d_model,
            text_vocab_path=text_vocab_path,
            ids_vocab_path=ids_vocab_path,
            nhead=nhead,
            num_decoder_layers=num_layers,
            dim_feedforward=1024,
            max_text_length=max_text_len,
            max_ids_length=max_ids_len,
            cross_coverage=False,
            self_coverage=False,
            use_space_char=True,
            ids_decode_mode="constrained",  # default test mode
        ).to(device)

        # ---- train forward check ----
        decoder.train()
        print("Decoder initialized successfully (train mode).")

        feat_seq_len = 32
        x = torch.randn(bs, feat_seq_len, d_model).to(device)
        print(f"Input feature shape: {x.shape}")

        text_lens_val = [5, 7]
        ids_lens_val = [10, 15]

        def get_vocab_size(path, use_space):
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if use_space and " " not in lines:
                lines.append(" ")
            return len(lines) + 4

        real_text_vocab_size = get_vocab_size(text_vocab_path, True)
        real_ids_vocab_size = get_vocab_size(ids_vocab_path, True)

        text_labels = torch.randint(4, real_text_vocab_size, (bs, max_text_len + 5)).to(device)
        text_lens = torch.tensor(text_lens_val, dtype=torch.long, device=device)

        ids_labels = torch.randint(4, real_ids_vocab_size, (bs, max_ids_len + 5)).to(device)
        ids_lens = torch.tensor(ids_lens_val, dtype=torch.long, device=device)

        data = [text_labels, text_lens, ids_labels, ids_lens]

        print("Running forward_train...")
        logits_text, logits_ids, sim_ids = decoder(x, data)

        print(f"Text Logits Shape: {logits_text.shape}")  # [B, 2+max(text_len), V_text]
        print(f"IDS Logits Shape:  {logits_ids.shape}")   # [B, Tenc, V_ids]
        print(f"Struct Sim Shape:  {sim_ids.shape}")      # [B, max(ids_len), max(ids_len)]

        assert logits_text.shape[0] == bs
        assert logits_text.shape[1] == 2 + max(text_lens_val)
        assert logits_text.shape[2] == real_text_vocab_size

        assert logits_ids.shape[0] == bs
        assert logits_ids.shape[1] == feat_seq_len
        assert logits_ids.shape[2] == real_ids_vocab_size

        assert sim_ids.shape[0] == bs
        assert sim_ids.shape[1] == max(ids_lens_val)
        assert sim_ids.shape[2] == max(ids_lens_val)

        print("Train shapes verification passed!")

        # ---- eval forward check ----
        decoder.eval()
        with torch.no_grad():
            probs_text, probs_ids_greedy = decoder(x, ids_decode_mode="greedy")
            probs_text2, probs_ids_constrained = decoder(x, ids_decode_mode="constrained")

        print(f"[EVAL] probs_text shape: {probs_text.shape}")
        print(f"[EVAL] probs_ids (greedy) shape: {probs_ids_greedy.shape}")
        print(f"[EVAL] probs_ids (constrained) shape: {probs_ids_constrained.shape}")
        print("Eval verification passed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with error: {e}")
