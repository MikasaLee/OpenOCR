from typing import Optional, Tuple, List, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.utils.ids_to_tree import load_char_to_ids
from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)


class TextIDSTreeDecoderv2(nn.Module):
    """
    Text: autoregressive TransformerDecoder
    IDS:  CTC head (encoder memory -> per-timestep logits)
    Tree: auxiliary structure head (token-level features -> parent sim)

    Inference(IDS) enhanced:
      - CTC prefix beam search
      - HARD IDS syntax constraint (Unicode IDS grammar: unary/binary/trinary IDC arity)
      - optional: align #segments with text length (spaces count)
      - optional: soft-bias Trie from char_to_ids (NO hard reject)
      - optional: tree-head rerank
    """

    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        text_vocab_path: Optional[str] = None,
        ids_vocab_path: Optional[str] = None,

        # lexicon source (for SOFT bias only; NOT hard legality)
        char_to_ids_path: Optional[str] = None,

        # IDS beam/constraint knobs
        ids_use_beam_search: bool = True,
        ids_beam_size: int = 20,
        ids_beam_token_topk: int = 80,          # per frame token top-k (blank handled separately)
        ids_beam_prune_logp: float = 12.0,      # prune beams with score < best - prune_logp
        ids_length_penalty_alpha: float = 0.0,  # 0 => no length penalty

        # HARD legality = IDS syntax (recommended True)
        ids_use_syntax_constraint: bool = True,
        ids_syntax_max_need: int = 64,          # safety cap for "need" (prevents runaway)

        # optional: force #segments to match predicted text length (recommended True if you rely on per-char segments)
        ids_use_text_length_constraint: bool = True,

        # SOFT bias trie (recommended True; does NOT kill错字检测)
        ids_lexicon_soft_bias: bool = True,
        ids_lexicon_prefix_bonus: float = 0.03,   # small bonus if still on trie prefix
        ids_lexicon_term_bonus: float = 0.15,     # bonus when a segment ends exactly on a lexicon entry

        # stronger rerank using tree head
        ids_struct_rerank_weight: float = 0.30, # 0 => disable rerank
        ids_struct_rerank_topk: int = 8,        # rerank only top-k beams per sample

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

        if out_channels is not None:
            text_vocab_size = out_channels
        else:
            text_vocab_size = _infer_vocab_size(text_vocab_path)
        ids_vocab_size = _infer_vocab_size(ids_vocab_path)

        # specials (match your pipeline)
        self.ignore_index = 0  # also used as CTC blank
        self.blank_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.max_text_len = int(max_text_length)
        self.max_ids_len = int(max_ids_length)
        self.use_space_char = bool(use_space_char)

        # beam configs
        self.ids_use_beam_search = bool(ids_use_beam_search)
        self.ids_beam_size = int(ids_beam_size)
        self.ids_beam_token_topk = int(ids_beam_token_topk)
        self.ids_beam_prune_logp = float(ids_beam_prune_logp)
        self.ids_length_penalty_alpha = float(ids_length_penalty_alpha)

        # syntax / segment alignment
        self.ids_use_syntax_constraint = bool(ids_use_syntax_constraint)
        self.ids_syntax_max_need = int(ids_syntax_max_need)
        self.ids_use_text_length_constraint = bool(ids_use_text_length_constraint)

        # soft-bias lexicon trie
        self.ids_lexicon_soft_bias = bool(ids_lexicon_soft_bias)
        self.ids_lexicon_prefix_bonus = float(ids_lexicon_prefix_bonus)
        self.ids_lexicon_term_bonus = float(ids_lexicon_term_bonus)

        # tree rerank
        self.ids_struct_rerank_weight = float(ids_struct_rerank_weight)
        self.ids_struct_rerank_topk = int(ids_struct_rerank_topk)

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

        # ===== ids branch (CTC) =====
        self.ids_ctc_norm = nn.LayerNorm(d_model)
        self.proj_ids_ctc = nn.Linear(d_model, ids_vocab_size)

        # ===== tree branch (aux) =====
        self.ids_embed_for_tree = nn.Sequential(nn.Embedding(ids_vocab_size, d_model), nn.LayerNorm(d_model))
        self.ids_pos_enc_for_tree = WordPosEnc(d_model)
        self.ids_norm_for_tree = nn.LayerNorm(d_model)
        self.tree_xattn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tree_ln = nn.LayerNorm(d_model)

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

        # ===== build IDS vocab strings =====
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

        # ===== build IDC operator id sets (for HARD syntax legality) =====
        self._ids_unary_ops: set[int] = set()
        self._ids_binary_ops: set[int] = set()
        self._ids_trinary_ops: set[int] = set()
        self._ids_all_ops: set[int] = set()
        self._init_ids_operator_sets()

        # ===== build lexicon trie from char_to_ids (SOFT bias only) =====
        self._lex_next: List[Dict[int, int]] = []
        self._lex_term: List[bool] = []
        self._lex_built: bool = False

        if char_to_ids_path is not None and self.ids_token2id is not None:
            self._build_lexicon_trie(char_to_ids_path)

    # ---------------- basic helpers ----------------
    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), fill_value=True, dtype=torch.bool, device=device)
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

    def _struct_sim(self, feat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        feat = self.struct_enc(feat, src_key_padding_mask=pad_mask)

        child = self.child_proj(feat)
        parent = self.parent_proj(feat)
        M = F.relu(child.unsqueeze(2) + parent.unsqueeze(1))
        sim = self.vs(M).squeeze(-1)  # [B,L,L]

        if pad_mask is not None:
            pm = pad_mask.clone()
            neg = -1e4 if sim.dtype in (torch.float16, torch.bfloat16) else -1e9
            sim = sim.masked_fill(pm[:, None, :], neg)
        return sim

    # ---------------- IDS operator sets (Unicode IDS grammar) ----------------
    def _init_ids_operator_sets(self):
        """
        Unicode IDS grammar (Chapter 18):
          Unary:  U+2FFE ⿾, U+2FFF ⿿
          Trinary: U+2FF2 ⿲, U+2FF3 ⿳
          Binary: U+2FF0 ⿰, U+2FF1 ⿱, U+2FF4..U+2FFD, plus U+31EF ㇯
        """
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

        self._ids_all_ops = set().union(self._ids_unary_ops, self._ids_binary_ops, self._ids_trinary_ops)

    # ---------------- Lexicon Trie (SOFT bias only) ----------------
    def _new_lex_node(self) -> int:
        self._lex_next.append({})
        self._lex_term.append(False)
        return len(self._lex_next) - 1

    def _build_lexicon_trie(self, char_to_ids_path: str):
        """
        Build trie from char_to_ids values (ids_str set).
        IMPORTANT: this trie is used for SOFT bias only, NOT for hard legality.
        """
        self._lex_next = []
        self._lex_term = []
        root = self._new_lex_node()

        char2ids = load_char_to_ids(char_to_ids_path)
        uniq = set(char2ids.values())

        for ids_str in uniq:
            if not ids_str:
                continue
            toks = [ids_str] if (ids_str.startswith("<") and ids_str.endswith(">")) else list(ids_str)

            ids = []
            ok = True
            for t in toks:
                if t not in self.ids_token2id:
                    ok = False
                    break
                ids.append(self.ids_token2id[t])
            if not ok:
                continue

            node = root
            for tid in ids:
                nxt = self._lex_next[node].get(tid, None)
                if nxt is None:
                    nxt = self._new_lex_node()
                    self._lex_next[node][tid] = nxt
                node = nxt
            self._lex_term[node] = True

        self._lex_built = True

    def _lex_step_and_bias(
        self,
        lex_node: int,
        token_id: int,
        need_after: int,
        is_space: bool,
        last_token: Optional[int],
    ) -> Tuple[int, float]:
        """
        Return (next_lex_node, bias_bonus).
        - Never blocks decoding.
        - If goes off-trie => next_lex_node = -1 and bonus=0.
        - If still on prefix => prefix_bonus.
        - If a segment completes exactly on terminal node => term_bonus.
        """
        if (not self.ids_lexicon_soft_bias) or (not self._lex_built):
            return -1, 0.0

        # dead state stays dead
        if lex_node == -1:
            return -1, 0.0

        root = 0
        bonus = 0.0

        if is_space:
            # space resets segment; can give terminal bonus if previous node was terminal
            if last_token is not None and self.space_id is not None and last_token == self.space_id:
                return -1, 0.0
            if 0 <= lex_node < len(self._lex_term) and self._lex_term[lex_node]:
                bonus += self.ids_lexicon_term_bonus
                return root, bonus
            return -1, 0.0

        # normal token
        if lex_node < 0 or lex_node >= len(self._lex_next):
            return -1, 0.0
        nxt = self._lex_next[lex_node].get(token_id, None)
        if nxt is None:
            return -1, 0.0

        bonus += self.ids_lexicon_prefix_bonus
        if need_after == 0 and self._lex_term[nxt]:
            bonus += self.ids_lexicon_term_bonus
        return nxt, bonus

    # ---------------- HARD IDS syntax constraint (arity) ----------------
    def _syntax_step(
        self,
        need: int,
        token_id: int,
        last_token: Optional[int],
        seg_count: int,
        target_segs: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """
        Maintain a simple arity-based legality for IDS in prefix notation.

        Interpret "need" as how many IDS nodes are still required to finish the current segment.
          - start segment: need=1
          - emit leaf(component): need -= 1
          - emit unary op: need += (1-1) = 0
          - emit binary op: need += (2-1) = +1
          - emit trinary op: need += (3-1) = +2
        Segment is complete when need == 0.
        Space is allowed only when segment complete, and resets need=1 for next segment.

        Returns (need_after, seg_count_after) or None if illegal.
        """
        # forbid special tokens inside IDS stream
        if token_id in (self.bos_id, self.eos_id):
            return None

        # if syntax constraint disabled: only keep space rules mildly
        if not self.ids_use_syntax_constraint:
            if self.use_space_char and self.space_id is not None and token_id == self.space_id:
                if last_token == self.space_id:
                    return None
                if target_segs is not None and seg_count >= target_segs:
                    return None
                return 1, seg_count + 1
            return need, seg_count

        # space handling
        if self.use_space_char and self.space_id is not None and token_id == self.space_id:
            if last_token == self.space_id:
                return None
            if need != 0:
                return None
            if target_segs is not None and seg_count >= target_segs:
                return None
            return 1, seg_count + 1

        # no token allowed after segment complete unless space
        if need == 0:
            return None

        # operator / leaf
        if token_id in self._ids_unary_ops:
            need_after = need  # +0
        elif token_id in self._ids_binary_ops:
            need_after = need + 1
        elif token_id in self._ids_trinary_ops:
            need_after = need + 2
        else:
            need_after = need - 1

        if need_after < 0:
            return None
        if need_after > self.ids_syntax_max_need:
            return None
        return need_after, seg_count

    # ---------------- CTC prefix beam core ----------------
    @staticmethod
    def _logadd(a: float, b: float) -> float:
        if a < -1e8:
            return b
        if b < -1e8:
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    def _ctc_prefix_beam_single(
        self,
        logp: np.ndarray,  # [T, V]
        target_segs: Optional[int],  # if not None, force seg_count == target_segs at end
    ) -> List[Dict]:
        """
        Returns N-best hyps:
          {"tokens": List[int], "score": float, "pb": float, "pnb": float, "need": int, "seg": int, "lex": int}
        """
        T, V = logp.shape
        blank = self.blank_id
        beam_size = max(1, self.ids_beam_size)
        topk = min(max(5, self.ids_beam_token_topk), V)
        LOG_ZERO = -1e9

        # state key: (prefix_tuple, need, seg_count, lex_node)
        # value: (pb, pnb)
        beams: Dict[Tuple[Tuple[int, ...], int, int, int], Tuple[float, float]] = {
            (tuple(), 1, 1, (0 if self._lex_built else -1)): (0.0, LOG_ZERO)
        }

        op_ids = np.array(sorted(list(self._ids_all_ops)), dtype=np.int64) if len(self._ids_all_ops) > 0 else None

        for t in range(T):
            lp_t = logp[t]
            lp_blank = float(lp_t[blank])

            cand = np.argpartition(lp_t, -topk)[-topk:]
            cand = cand[np.argsort(lp_t[cand])[::-1]]

            extra = []
            if self.use_space_char and self.space_id is not None:
                extra.append(self.space_id)
            if self.unk_id is not None:
                extra.append(self.unk_id)
            if op_ids is not None and op_ids.size > 0:
                extra.extend(op_ids.tolist())

            if len(extra) > 0:
                cand = np.unique(np.concatenate([cand, np.array(extra, dtype=cand.dtype)], axis=0))

            next_beams: Dict[Tuple[Tuple[int, ...], int, int, int], Tuple[float, float]] = {}

            def _get(k):
                return next_beams.get(k, (LOG_ZERO, LOG_ZERO))

            def _set(k, pb, pnb):
                next_beams[k] = (pb, pnb)

            for (pref, need, seg, lex_node), (pb, pnb) in beams.items():
                last = pref[-1] if len(pref) > 0 else None

                # 1) emit blank: prefix unchanged
                k0 = (pref, need, seg, lex_node)
                nb_pb, nb_pnb = _get(k0)
                nb_pb = self._logadd(nb_pb, self._logadd(pb, pnb) + lp_blank)
                _set(k0, nb_pb, nb_pnb)

                # 2) emit non-blank
                for c in cand:
                    c = int(c)
                    if c == blank:
                        continue
                    if c == self.bos_id or c == self.eos_id:
                        continue

                    is_space = (self.use_space_char and self.space_id is not None and c == self.space_id)

                    step = self._syntax_step(need, c, last, seg, target_segs)
                    if step is None:
                        continue
                    need2, seg2 = step

                    # lex soft bias
                    lex2, bias = self._lex_step_and_bias(
                        lex_node=lex_node,
                        token_id=c,
                        need_after=need2,
                        is_space=is_space,
                        last_token=last,
                    )
                    lp_c = float(lp_t[c]) + float(bias)

                    # CTC prefix update rules
                    if last is not None and c == last:
                        # a) stay on same prefix from pnb (repeat w/o blank): state unchanged
                        k_same = (pref, need, seg, lex_node)
                        sb_pb, sb_pnb = _get(k_same)
                        sb_pnb = self._logadd(sb_pnb, pnb + lp_c)
                        _set(k_same, sb_pb, sb_pnb)

                        # b) extend from pb only: prefix+token, state advanced
                        new_pref = pref + (c,)
                        k_ext = (new_pref, need2, seg2, lex2)
                        eb_pb, eb_pnb = _get(k_ext)
                        eb_pnb = self._logadd(eb_pnb, pb + lp_c)
                        _set(k_ext, eb_pb, eb_pnb)
                    else:
                        new_pref = pref + (c,)
                        k_ext = (new_pref, need2, seg2, lex2)
                        eb_pb, eb_pnb = _get(k_ext)
                        eb_pnb = self._logadd(eb_pnb, self._logadd(pb, pnb) + lp_c)
                        _set(k_ext, eb_pb, eb_pnb)

            # prune
            scored = []
            best = -1e9
            for (pref, need, seg, lex_node), (pb, pnb) in next_beams.items():
                sc = self._logadd(pb, pnb)
                if sc > best:
                    best = sc
                scored.append((sc, pref, need, seg, lex_node, pb, pnb))

            if self.ids_beam_prune_logp > 0:
                keep_thr = best - self.ids_beam_prune_logp
                scored = [x for x in scored if x[0] >= keep_thr]

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:beam_size]

            beams = {(p, need, seg, lex): (pb, pnb) for (sc, p, need, seg, lex, pb, pnb) in scored}

        # finalize: require segment closed (need==0), not ending with space, and seg_count==target if set
        hyps = []
        for (pref, need, seg, lex_node), (pb, pnb) in beams.items():
            if self.ids_use_syntax_constraint:
                if need != 0:
                    continue
            if target_segs is not None and seg != target_segs:
                continue
            if self.use_space_char and self.space_id is not None and len(pref) > 0 and pref[-1] == self.space_id:
                continue

            sc = self._logadd(pb, pnb)
            if self.ids_length_penalty_alpha != 0.0:
                L = max(1, len(pref))
                sc = sc / (float(L) ** self.ids_length_penalty_alpha)
            hyps.append({"tokens": list(pref), "score": sc, "pb": pb, "pnb": pnb, "need": need, "seg": seg, "lex": lex_node})

        if len(hyps) == 0:
            # fallback: best unfinished
            for (pref, need, seg, lex_node), (pb, pnb) in beams.items():
                sc = self._logadd(pb, pnb)
                if self.ids_length_penalty_alpha != 0.0:
                    L = max(1, len(pref))
                    sc = sc / (float(L) ** self.ids_length_penalty_alpha)
                hyps.append({"tokens": list(pref), "score": sc, "pb": pb, "pnb": pnb, "need": need, "seg": seg, "lex": lex_node})
            hyps.sort(key=lambda d: d["score"], reverse=True)

        hyps.sort(key=lambda d: d["score"], reverse=True)
        return hyps[: max(1, self.ids_beam_size)]

    def _tree_rerank_single(
        self,
        mem_ids: torch.Tensor,                 # [1, Tenc, C]
        mem_mask_ids: Optional[torch.Tensor],  # [1, Tenc] bool
        hyps: List[Dict],
    ) -> List[int]:
        """
        score = ctc_score + w * tree_score
        tree_score: average max log P(parent) with constraint parent < child
        ignore PAD and SPACE positions
        """
        if len(hyps) == 0:
            return []

        if self.ids_struct_rerank_weight <= 0:
            return hyps[0]["tokens"]

        K = min(len(hyps), max(1, self.ids_struct_rerank_topk))
        cand = hyps[:K]
        seqs = [h["tokens"][: self.max_ids_len] for h in cand]

        maxL = max(1, max(len(s) for s in seqs))
        device = mem_ids.device
        tgt = torch.full((K, maxL), self.ignore_index, dtype=torch.long, device=device)
        lens = []
        for i, s in enumerate(seqs):
            L = min(len(s), maxL)
            lens.append(L)
            if L > 0:
                tgt[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)

        pad_mask = (tgt == self.ignore_index)
        if self.use_space_char and self.space_id is not None:
            pad_mask = pad_mask | (tgt == self.space_id)

        with torch.cuda.amp.autocast(enabled=False):
            q = self.ids_embed_for_tree(tgt).float()
            q = self.ids_pos_enc_for_tree(q)
            q = self.ids_norm_for_tree(q)

            mem_fp32 = mem_ids.float().expand(K, -1, -1)
            mem_mask_k = mem_mask_ids.expand(K, -1) if mem_mask_ids is not None else None

            attn_out, _ = self.tree_xattn(q, mem_fp32, mem_fp32, key_padding_mask=mem_mask_k)
            token_feat = self.tree_ln(q + attn_out)

            sim = self._struct_sim(token_feat, pad_mask).float()  # [K,L,L]
            log_sim = F.log_softmax(sim, dim=-1)

        tree_scores = []
        for i in range(K):
            L = lens[i]
            if L <= 1:
                tree_scores.append(0.0)
                continue
            valid_cols = (~pad_mask[i, :L]).detach().cpu().numpy().astype(bool)

            ssum = 0.0
            cnt = 0
            for r in range(1, L):
                if pad_mask[i, r]:
                    continue
                cols = np.where(valid_cols[:r])[0]
                if cols.size == 0:
                    continue
                vals = log_sim[i, r, :r].detach().cpu().numpy()
                m = float(np.max(vals[cols]))
                ssum += m
                cnt += 1
            tree_scores.append(ssum / max(1, cnt))

        comb = []
        for i in range(K):
            comb.append(float(cand[i]["score"]) + self.ids_struct_rerank_weight * float(tree_scores[i]))
        best_i = int(np.argmax(np.array(comb, dtype=np.float32)))
        return seqs[best_i]

    @staticmethod
    def _tokens_to_pseudo_ctc(seq: List[int], blank_id: int = 0) -> List[int]:
        out = []
        prev = None
        for tok in seq:
            if prev is not None and tok == prev:
                out.append(blank_id)
            out.append(tok)
            prev = tok
        return out

    # ---------------- forward ----------------
    def forward_train(self, x, data):
        if isinstance(x, dict) and "text" in x and "ids" in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x["text"])
            mem_ids, h_ids, mem_mask_ids = self._prep_memory(x["ids"])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, h_ids, mem_mask_ids = mem_text, h_text, mem_mask_text

        # data order:
        text_labels = data[0]
        text_lengths = data[1]
        ids_ctc_labels = data[2]
        ids_ctc_lengths = data[3]

        # text logits
        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, : 2 + max_text]
        logits_text = self._decode_text_seq(mem_text, h_text, mem_mask_text, tgt_text)

        # ids logits (CTC)
        feats_ids = self.ids_ctc_norm(mem_ids)
        logits_ids_ctc = self.proj_ids_ctc(feats_ids)

        # tree sim (aux)
        max_ids = int(ids_ctc_lengths.max().item())
        tgt_ids = ids_ctc_labels[:, :max_ids]
        pad_ids = (tgt_ids == self.ignore_index)

        with torch.cuda.amp.autocast(enabled=False):
            q = self.ids_embed_for_tree(tgt_ids).float()
            q = self.ids_pos_enc_for_tree(q)
            q = self.ids_norm_for_tree(q)

            mem_fp32 = mem_ids.float()
            attn_out, _ = self.tree_xattn(q, mem_fp32, mem_fp32, key_padding_mask=mem_mask_ids)
            token_feat = self.tree_ln(q + attn_out)

            sim_ids = self._struct_sim(token_feat, pad_ids).float()

        return logits_text, logits_ids_ctc, sim_ids

    def forward_test(self, x):
        if isinstance(x, dict) and "text" in x and "ids" in x:
            mem_text, h_text, mem_mask_text = self._prep_memory(x["text"])
            mem_ids, h_ids, mem_mask_ids = self._prep_memory(x["ids"])
        else:
            mem_text, h_text, mem_mask_text = self._prep_memory(x)
            mem_ids, h_ids, mem_mask_ids = mem_text, h_text, mem_mask_text

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

        # predicted text lengths (exclude BOS; stop at first EOS)
        # pred tokens are tgt[:, 1:]
        pred = tgt[:, 1:]
        Lp = pred.size(1)
        eos_mask = (pred == self.eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.zeros((B,), dtype=torch.long, device=device)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)  # only valid for has_eos rows
        text_len_pred = torch.where(has_eos, first_eos, torch.full((B,), Lp, device=device, dtype=torch.long))
        text_len_pred = torch.clamp(text_len_pred, min=1)  # at least 1 segment

        # ---- IDS CTC logits ----
        logits_ids_ctc = self.proj_ids_ctc(self.ids_ctc_norm(mem_ids))  # [B,Tenc,V]
        V = logits_ids_ctc.size(-1)

        # fallback: no beam
        if (not self.ids_use_beam_search) or (self.ids_beam_size <= 1):
            probs_ids = F.softmax(logits_ids_ctc, dim=-1)
            return probs_text, probs_ids

        # ---- beam decode on CPU (log-softmax) ----
        logp = F.log_softmax(logits_ids_ctc.float(), dim=-1).detach().cpu().numpy()

        best_seqs: List[List[int]] = []
        for b in range(B):
            target_segs = int(text_len_pred[b].item()) if (self.use_space_char and self.ids_use_text_length_constraint) else None
            hyps = self._ctc_prefix_beam_single(logp[b], target_segs=target_segs)
            seq = self._tree_rerank_single(
                mem_ids[b:b + 1],
                mem_mask_ids[b:b + 1] if mem_mask_ids is not None else None,
                hyps,
            )
            best_seqs.append(seq)

        # ---- build one-hot probs from best seq ----
        aligns = [self._tokens_to_pseudo_ctc(s, blank_id=self.blank_id) for s in best_seqs]
        lens = [len(a) for a in aligns]
        maxT = max(1, max(lens))

        probs_ids = torch.zeros((B, maxT, V), dtype=torch.float16, device=device)
        for b in range(B):
            a = aligns[b]
            L = len(a)
            if L == 0:
                probs_ids[b, :, self.blank_id] = 1.0
                continue
            for t, tok in enumerate(a):
                probs_ids[b, t, int(tok)] = 1.0
            if L < maxT:
                probs_ids[b, L:, self.blank_id] = 1.0

        return probs_text, probs_ids

    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        else:
            return self.forward_test(x)




# python -m openrec.modeling.decoders.text_ids_tree_decoderv2
if __name__ == "__main__":
    import torch
    import sys
    import os
    import shutil

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
    nhead = 4 # Reduced for speed
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
        decoder = TextIDSTreeDecoderv2(
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
            use_space_char=True
        ).to(device)
        
        decoder.train()
        print("Decoder initialized successfully.")
        
        # Dummy Input
        # x: [B, Seq, C]
        feat_seq_len = 32
        x = torch.randn(bs, feat_seq_len, d_model).to(device)   # [2,2*16,512]
        print(f"Input feature shape: {x.shape}")
        
        # Dummy Targets
        # Labels include special tokens: 0:pad, 1:sos, 2:eos, 3:unk. + vocab
        # text_labels: [B, L]
        # Make sure lengths imply valid range
        text_lens_val = [5, 7, 9]
        ids_lens_val = [10, 15, 25]
        
        # Determine strict vocab sizes for random labels
        def get_vocab_size(path, use_space):
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
            if use_space and " " not in lines:
                lines.append(" ")
            return len(lines) + 4
            
        real_text_vocab_size = get_vocab_size(text_vocab_path, True)
        real_ids_vocab_size = get_vocab_size(ids_vocab_path, True)
        
        text_labels = torch.randint(4, real_text_vocab_size, (bs, max_text_len + 5)).to(device)
        text_lens = torch.tensor(text_lens_val).to(device)
        
        ids_labels = torch.randint(4, real_ids_vocab_size, (bs, max_ids_len + 5)).to(device)
        ids_lens = torch.tensor(ids_lens_val).to(device)
        
        data = [text_labels, text_lens, ids_labels, ids_lens]
        
        print("Running forward_train...")
        logits_text, logits_ids, sim_ids = decoder(x, data)
        
        print(f"Text Logits Shape: {logits_text.shape}") 
        # Expected: [B, 2 + max(text_lens_val), VocabSize]
        
        print(f"IDS Logits Shape:  {logits_ids.shape}")  
        # Expected: [B, 2 + max(ids_lens_val), VocabSize]
        
        print(f"Struct Sim Shape:  {sim_ids.shape}")    
        # Expected: [B, 2 + max(ids_lens_val), 2 + max(ids_lens_val)]
        
        assert logits_text.shape[0] == bs
        assert logits_text.shape[1] == 2 + max(text_lens_val)
        assert logits_text.shape[2] == real_text_vocab_size, f"Text vocab size mismatch. Expected {real_text_vocab_size}, got {logits_text.shape[2]}"
        
        assert logits_ids.shape[0] == bs
        assert logits_ids.shape[1] == 2 + max(ids_lens_val)
        assert logits_ids.shape[2] == real_ids_vocab_size, f"IDS vocab size mismatch. Expected {real_ids_vocab_size}, got {logits_ids.shape[2]}"
        
        assert sim_ids.shape[0] == bs
        assert sim_ids.shape[1] == 2 + max(ids_lens_val)
        assert sim_ids.shape[2] == 2 + max(ids_lens_val)
        
        print("Shapes verification passed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with error: {e}")