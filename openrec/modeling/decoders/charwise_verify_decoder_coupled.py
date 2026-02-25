"""
Coupled CharWiseVerifyDecoder:
  保留原始解码器结构，但可选地用文本分支的 hidden states
  直接作为每字符的特征（`coupling_mode='text_hidden'`），以增强
  文本→IDS 的信息流动。

用途：用于对比实验，验证“分支耦合”对IDS查错/合法性的影响。
"""
from typing import Optional, Tuple, List, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charwise_verify_decoder import CharWiseVerifyDecoder


class CharWiseVerifyDecoderCoupled(CharWiseVerifyDecoder):
    """耦合版：当 `coupling_mode=='text_hidden'` 时，使用 text_hidden
    的 token-level features 作为 per-char 特征（直接投影），否则
    回退到父类的独立定位器。
    """

    def __init__(
        self,
        in_channels: int,
        coupling_mode: str = "text_hidden",
        coupling_alpha: float = 0.2,
        detach_text_for_ids: bool = False,
        stopgrad_attn: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, **kwargs)
        self.coupling_mode = str(coupling_mode).lower()
        valid_modes = {"text_hidden", "attn_crop", "independent", "hybrid"}
        if self.coupling_mode not in valid_modes:
            raise ValueError(
                f"Unsupported coupling_mode: {coupling_mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        d_model = in_channels
        if self.coupling_mode in ("text_hidden", "hybrid"):
            self.text_to_char_proj = nn.Linear(d_model, d_model)
            # Independent locator branch is bypassed only in text_hidden mode.
            if self.coupling_mode == "text_hidden":
                self._freeze_independent_locator(freeze_ln=True)
        elif self.coupling_mode == "attn_crop":
            # attn_crop uses char_visual_ln but not the independent locator.
            self._freeze_independent_locator(freeze_ln=False)

        self.coupling_alpha = float(coupling_alpha)
        if not (0.0 <= self.coupling_alpha <= 1.0):
            raise ValueError(f"coupling_alpha must be in [0, 1], got {self.coupling_alpha}")
        self.detach_text_for_ids = bool(detach_text_for_ids)
        # attn crop options
        self.stopgrad_attn = bool(stopgrad_attn)

    def _freeze_independent_locator(self, freeze_ln: bool = True):
        # Mark bypassed branch parameters as non-trainable to avoid DDP
        # "unused parameter" reduction errors.
        self.char_pos_queries.requires_grad = False
        for p in self.char_locator.parameters():
            p.requires_grad = False
        if freeze_ln:
            for p in self.char_visual_ln.parameters():
                p.requires_grad = False

    def forward_train(self, x, data):
        # 基于父类实现，区别在于 char_feat 的获取
        memory, h, mem_mask = self._prep_memory(x)
        text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths = data[:4]

        # 1. Text AR
        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, :2 + max_text]
        pad_mask = (tgt_text == self.ignore_index)
        L_text = tgt_text.size(1)
        tgt_mask = self._causal_mask(L_text, device=memory.device)
        tgt_emb = self.text_embed(tgt_text)
        tgt_emb = self.text_pos_enc(tgt_emb)
        tgt_emb = self.text_norm(tgt_emb)
        # 如果使用 attn_crop，则请求 cross-attn 权重
        if self.coupling_mode == "attn_crop":
            text_hidden, cross_attn = self.text_decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=memory.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
                return_attn=True,
            )
            text_hidden = text_hidden.transpose(0, 1)
        else:
            text_hidden = self.text_decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=memory.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
            ).transpose(0, 1)
        logits_text = self.proj_text(text_hidden)

        # 2. Char features (耦合/回退)
        device = memory.device
        max_chars = max_text
        if self.coupling_mode == "text_hidden":
            # text_hidden: [B, 2+max_text, d]
            # 取 [1:1+max_text] 作为每字符表示（对应每个字符预测位置）
            char_hidden = text_hidden[:, 1:1 + max_chars, :].contiguous()
            char_feat = self.text_to_char_proj(char_hidden).unsqueeze(2)  # [B, max_chars, 1, d]
        elif self.coupling_mode == "hybrid":
            # Keep independent IDS evidence while injecting a controlled
            # amount of text-branch representation.
            char_feat_ind = self._extract_char_features(memory, mem_mask, max_chars, lengths=text_lengths)
            char_hidden = text_hidden[:, 1:1 + max_chars, :].contiguous()
            if self.detach_text_for_ids:
                char_hidden = char_hidden.detach()
            char_hidden = self.text_to_char_proj(char_hidden)
            a = self.coupling_alpha
            char_vec = (1.0 - a) * char_feat_ind[:, :max_chars, 0, :] + a * char_hidden
            char_vec = self.char_visual_ln(char_vec)
            char_feat = char_vec.unsqueeze(2)
        elif self.coupling_mode == "attn_crop":
            # cross_attn: [B, nhead, Tq, Sk]  (Tq = 2+max_text)
            # aggregate heads -> [B, Tq, Sk]
            attn = cross_attn.mean(dim=1)  # head mean
            # select per-char rows (skip BOS at pos 0)
            A = attn[:, 1:1 + max_chars, :].contiguous()
            # mask padding positions beyond text_lengths
            if text_lengths is not None:
                idx = torch.arange(max_chars, device=A.device).unsqueeze(0)
                pad = idx >= text_lengths.unsqueeze(1)
                A = A.masked_fill(pad.unsqueeze(-1), 0.0)
            if self.stopgrad_attn:
                A = A.detach()
            # memory: [B, Tenc, d]
            char_vec = torch.bmm(A, memory)  # [B, max_chars, d]
            char_vec = self.char_visual_ln(char_vec)
            char_feat = char_vec.unsqueeze(2)
        else:
            char_feat = self._extract_char_features(memory, mem_mask, max_chars, lengths=text_lengths)

        # 3. Per-char CTC（与父类一致）
        ids_label_2d = per_char_ids_labels[:, :max_chars, :]
        ids_len_2d = per_char_ids_lengths[:, :max_chars]
        char_feat_2d = char_feat[:, :max_chars, 0, :]

        valid_char_mask = torch.arange(max_chars, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
        char_feat_valid = char_feat_2d[valid_char_mask]
        ids_valid = ids_label_2d[valid_char_mask]
        ids_len_valid = ids_len_2d[valid_char_mask]

        ctc_logits = self._ctc_forward(char_feat_valid)
        ids_ctc_loss = self._ctc_loss(ctc_logits, ids_valid, ids_len_valid)

        if self.grammar_penalty_weight > 0:
            grammar_penalty = self._ctc_grammar_penalty(ctc_logits)
        else:
            grammar_penalty = torch.tensor(0.0, device=device)

        with torch.no_grad():
            ids_decoded_valid = self._ctc_greedy_decode(ctc_logits)

        # Unflatten
        all_char_ids_train: List[List[List[int]]] = []
        idx = 0
        B = memory.size(0)
        for b in range(B):
            n = min(int(text_lengths[b].item()), max_chars)
            all_char_ids_train.append(ids_decoded_valid[idx:idx + n])
            idx += n

        return logits_text, (ids_ctc_loss, all_char_ids_train), char_feat, grammar_penalty, max_text

    def forward_test(self, x):
        # 基于父类 forward_test，但在完成 text 解码后重新计算 text_hidden
        memory, h, mem_mask = self._prep_memory(x)
        B, Tenc, _ = memory.shape
        device = memory.device

        # 1. Text greedy AR（与父类相同）
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

        # 为了得到 text_hidden（token-level features），对完整 tgt 再运行一次 decoder
        pad_mask = (tgt == self.ignore_index)
        L_text = tgt.size(1)
        tgt_mask = self._causal_mask(L_text, device=memory.device)
        tgt_emb = self.text_embed(tgt)
        tgt_emb = self.text_pos_enc(tgt_emb)
        tgt_emb = self.text_norm(tgt_emb)
        # 如果使用 attn_crop，则请求 cross-attn 权重
        if self.coupling_mode == "attn_crop":
            text_hidden, cross_attn = self.text_decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=memory.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
                return_attn=True,
            )
            text_hidden = text_hidden.transpose(0, 1)
        else:
            text_hidden = self.text_decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=memory.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
            ).transpose(0, 1)  # [B, L, d]

        # 2. Char features
        max_chars = min(int(text_len_pred.max().item()), tgt.size(1) - 1)
        if max_chars > 0:
            if self.coupling_mode == "text_hidden":
                char_hidden = text_hidden[:, 1:1 + max_chars, :].contiguous()
                char_feat_all = self.text_to_char_proj(char_hidden).unsqueeze(2)
            elif self.coupling_mode == "hybrid":
                char_feat_ind = self._extract_char_features(memory, mem_mask, max_chars, lengths=text_len_pred)
                char_hidden = text_hidden[:, 1:1 + max_chars, :].contiguous()
                if self.detach_text_for_ids:
                    char_hidden = char_hidden.detach()
                char_hidden = self.text_to_char_proj(char_hidden)
                a = self.coupling_alpha
                char_vec = (1.0 - a) * char_feat_ind[:, :max_chars, 0, :] + a * char_hidden
                char_vec = self.char_visual_ln(char_vec)
                char_feat_all = char_vec.unsqueeze(2)
            elif self.coupling_mode == "attn_crop":
                # cross_attn: [B, nhead, Tq, Sk]
                attn = cross_attn.mean(dim=1)
                A = attn[:, 1:1 + max_chars, :].contiguous()
                if text_len_pred is not None:
                    idx = torch.arange(max_chars, device=A.device).unsqueeze(0)
                    pad = idx >= text_len_pred.unsqueeze(1)
                    A = A.masked_fill(pad.unsqueeze(-1), 0.0)
                if self.stopgrad_attn:
                    A = A.detach()
                char_vec = torch.bmm(A, memory)
                char_vec = self.char_visual_ln(char_vec)
                char_feat_all = char_vec.unsqueeze(2)
            else:
                char_feat_all = self._extract_char_features(memory, mem_mask, max_chars, lengths=text_len_pred)

            valid_char_mask = torch.arange(max_chars, device=device).unsqueeze(0) < text_len_pred.unsqueeze(1)
            char_feat_valid = char_feat_all[valid_char_mask][:, 0, :]

            ctc_logits = self._ctc_forward(char_feat_valid)
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


if __name__ == "__main__":
    # 简单自测：复用父类的测试逻辑
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing CharWiseVerifyDecoderCoupled")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d_model = 256
    nhead = 4
    bs = 2
    max_text_len = 15
    max_single_char_ids_len = 15
    feat_seq_len = 32

    text_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")

    for p in [text_vocab_path, ids_vocab_path, char_to_ids_path]:
        assert os.path.exists(p), f"Not found: {p}"

    try:
        decoder = CharWiseVerifyDecoderCoupled(
            in_channels=d_model,
            text_vocab_path=text_vocab_path,
            ids_vocab_path=ids_vocab_path,
            char_to_ids_path=char_to_ids_path,
            nhead=nhead,
            num_decoder_layers=1,
            dim_feedforward=512,
            max_text_length=max_text_len,
            max_single_char_ids_len=max_single_char_ids_len,
            cross_coverage=False,
            self_coverage=False,
            use_space_char=True,
            constrained_ctc_decode=True,
            grammar_penalty_weight=0.1,
            ctc_frames_per_char=32,
            num_ctc_encoder_layers=1,
            coupling_mode="text_hidden",
        ).to(device)
        print(f"[OK] Coupled Decoder initialized. text_vocab={decoder.text_vocab_size}, ids_vocab={decoder.ids_vocab_size}")

        # ---- train forward ----
        decoder.train()
        x = torch.randn(bs, feat_seq_len, d_model, device=device)
        text_lens_val = [3, 5]
        ids_label_len = max_single_char_ids_len + 2

        text_labels = torch.randint(4, decoder.text_vocab_size, (bs, 2 + max_text_len), device=device)
        text_labels[:, 0] = 1
        for b in range(bs):
            text_labels[b, 1 + text_lens_val[b]] = 2
            text_labels[b, 2 + text_lens_val[b]:] = 0
        text_lengths = torch.tensor(text_lens_val, dtype=torch.long, device=device)

        per_char_ids_labels = torch.randint(4, decoder.ids_vocab_size, (bs, max_text_len, ids_label_len), device=device)
        per_char_ids_lengths = torch.zeros(bs, max_text_len, dtype=torch.long, device=device)
        for b in range(bs):
            for ci in range(text_lens_val[b]):
                ids_len = 3
                per_char_ids_labels[b, ci, 0] = 1
                per_char_ids_labels[b, ci, 1 + ids_len] = 2
                per_char_ids_labels[b, ci, 2 + ids_len:] = 0
                per_char_ids_lengths[b, ci] = ids_len
            per_char_ids_labels[b, text_lens_val[b]:] = 0

        data = [text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths]

        print("\nRunning forward_train...")
        logits_text, ids_output, char_feat, grammar_penalty, max_text = decoder(x, data)
        ids_ctc_loss, all_char_ids_train = ids_output

        print(f"  logits_text shape: {logits_text.shape}")
        print(f"  ids_ctc_loss:      {ids_ctc_loss.item():.4f}")
        print(f"  grammar_penalty:   {grammar_penalty.item():.4f}")
        print(f"  #batches decoded:  {len(all_char_ids_train)}")
        print(f"  char_feat shape:   {char_feat.shape}")

        loss = logits_text.sum() + ids_ctc_loss
        loss.backward()
        print("  [OK] Train backward executed.")

        # ---- eval forward ----
        decoder.eval()
        print("\nRunning forward_test...")
        with torch.no_grad():
            probs_text, all_char_ids, text_len_pred = decoder(x)

        print(f"  probs_text shape:  {probs_text.shape}")
        print(f"  text_len_pred:     {text_len_pred.tolist()}")
        print("  [OK] Eval forward executed.")

        print("\n" + "=" * 60)
        print("[PASS] CharWiseVerifyDecoderCoupled tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")
