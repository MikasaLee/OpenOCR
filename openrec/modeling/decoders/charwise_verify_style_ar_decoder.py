from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charwise_verify_style_decoder import CharWiseVerifyStyleDecoder
from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoder,
    TransformerDecoderLayer,
)


class CharWiseVerifyStyleARDecoder(CharWiseVerifyStyleDecoder):
    """Ablation variant: keep the same pipeline but decode per-char IDS with AR."""

    def _freeze_inherited_ctc_branch(self):
        # This ablation keeps the parent class scaffolding but does not use
        # the inherited per-char CTC head at all. Mark those parameters as
        # non-trainable so DDP does not expect gradients from the bypassed path.
        if isinstance(getattr(self, "char_frame_pos", None), torch.nn.Parameter):
            self.char_frame_pos.requires_grad = False
        if getattr(self, "char_frame_expand", None) is not None:
            for p in self.char_frame_expand.parameters():
                p.requires_grad = False
        if getattr(self, "ctc_frame_encoder", None) is not None:
            for p in self.ctc_frame_encoder.parameters():
                p.requires_grad = False
        if getattr(self, "ctc_proj", None) is not None:
            for p in self.ctc_proj.parameters():
                p.requires_grad = False

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
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            text_vocab_path=text_vocab_path,
            ids_vocab_path=ids_vocab_path,
            char_to_ids_path=char_to_ids_path,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            max_text_length=max_text_length,
            max_single_char_ids_len=max_single_char_ids_len,
            use_space_char=use_space_char,
            constrained_ctc_decode=constrained_ctc_decode,
            ids_syntax_max_need=ids_syntax_max_need,
            grammar_penalty_weight=grammar_penalty_weight,
            **kwargs,
        )

        self.ids_ar_max_len = int(kwargs.get("ids_ar_max_len", self.max_single_char_ids_len))
        self.ids_ar_num_decoder_layers = int(
            kwargs.get("ids_ar_num_decoder_layers", num_decoder_layers)
        )
        self.ids_use_style_memory = bool(kwargs.get("ids_use_style_memory", True))
        self.grammar_penalty_weight = 0.0
        self._freeze_inherited_ctc_branch()

        if self.enable_ids_branch:
            self.ids_embed = nn.Sequential(
                nn.Embedding(self.ids_vocab_size, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            self.ids_pos_enc = WordPosEnc(self.d_model)
            self.ids_norm = nn.LayerNorm(self.d_model)
            self.ids_decoder = TransformerDecoder(
                TransformerDecoderLayer(
                    self.d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                ),
                num_layers=self.ids_ar_num_decoder_layers,
                arm=None,
            )
            self.proj_ids = nn.Linear(self.d_model, self.ids_vocab_size)
            self.ids_memory_norm = nn.LayerNorm(self.d_model)
            self.ids_seq_ce = nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=self.ignore_index,
            )
        else:
            self.ids_embed = None
            self.ids_pos_enc = None
            self.ids_norm = None
            self.ids_decoder = None
            self.proj_ids = None
            self.ids_memory_norm = None
            self.ids_seq_ce = None

    def _build_ids_memory(
        self,
        char_feat_valid: torch.Tensor,
        style_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = char_feat_valid
        if (
            self.ids_use_style_memory
            and self.use_line_style_film
            and style_code is not None
        ):
            if isinstance(self.style_alpha, torch.Tensor):
                memory = memory + self.style_alpha * style_code
            else:
                memory = memory + style_code
        memory = self.ids_memory_norm(memory)
        return memory.unsqueeze(1)

    def _decode_ids_ar_train(
        self,
        ids_memory: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        pad_mask = tgt_ids.eq(self.ignore_index)
        tgt_mask = self._causal_mask(tgt_ids.size(1), device=tgt_ids.device)

        tgt = self.ids_embed(tgt_ids)
        tgt = self.ids_pos_enc(tgt)
        tgt = self.ids_norm(tgt)

        hidden = self.ids_decoder(
            tgt=tgt.transpose(0, 1),
            memory=ids_memory.transpose(0, 1),
            height=1,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=None,
        ).transpose(0, 1)
        return self.proj_ids(hidden)

    def _ids_ar_loss(
        self,
        logits_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        if logits_ids.size(0) == 0:
            return logits_ids.new_zeros(())

        pred_seq = logits_ids[:, :-1, :].contiguous()
        tgt_seq = tgt_ids[:, 1:].contiguous()
        loss = self.ids_seq_ce(
            pred_seq.reshape(-1, pred_seq.size(-1)),
            tgt_seq.reshape(-1),
        )
        valid = tgt_seq.ne(self.ignore_index).reshape(-1)
        return loss.masked_select(valid).mean() if valid.any() else loss.mean()

    @torch.no_grad()
    def _ids_ar_greedy_decode(
        self,
        ids_memory: torch.Tensor,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        num_chars = ids_memory.size(0)
        device = ids_memory.device

        if num_chars == 0:
            return [], ids_memory.new_zeros((0,))

        tgt = torch.full(
            (num_chars, 1),
            self.bos_id,
            dtype=torch.long,
            device=device,
        )
        step_probs = []

        for _ in range(self.ids_ar_max_len + 1):
            logits = self._decode_ids_ar_train(ids_memory, tgt)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            step_probs.append(probs)

            next_token = probs.argmax(dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            if next_token.eq(self.eos_id).all():
                break

        pred_tokens = tgt[:, 1:]
        pred_conf = torch.stack(step_probs, dim=1).max(dim=-1).values

        decoded = []
        conf = []
        for i in range(num_chars):
            tokens = []
            token_conf = []
            for step_idx, token in enumerate(pred_tokens[i].tolist()):
                if token == self.eos_id:
                    break
                if token in (self.ignore_index, self.bos_id):
                    continue
                tokens.append(token)
                token_conf.append(float(pred_conf[i, step_idx].item()))
            decoded.append(tokens)
            conf.append(sum(token_conf) / len(token_conf) if token_conf else 0.0)

        return decoded, ids_memory.new_tensor(conf)

    def forward_train(self, x, data):
        if not self.enable_ids_branch:
            return super().forward_train(x, data)

        memory, h, mem_mask, feat2d = self._prep_memory(x)
        text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths = data[:4]

        max_chars = self.max_text_len
        char_feat, valid_logits = self._extract_char_features(
            memory,
            mem_mask,
            max_chars,
            lengths=None,
        )
        char_feat_2d = char_feat[:, :, 0, :]

        if self.use_valid_head:
            valid_targets = (
                torch.arange(max_chars, device=memory.device).unsqueeze(0)
                < text_lengths.unsqueeze(1)
            ).to(valid_logits.dtype)
            valid_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
        else:
            valid_loss = torch.tensor(0.0, device=memory.device)
        self.last_valid_loss = float(valid_loss.detach().item())

        style_line = None
        if self.use_line_style_film:
            style_line = self._extract_line_style_code(feat2d, memory, mem_mask)

        device = memory.device
        valid_char_mask = (
            torch.arange(max_chars, device=device).unsqueeze(0)
            < text_lengths.unsqueeze(1)
        )
        ids_label_2d = per_char_ids_labels[:, :max_chars, :]
        ids_len_2d = per_char_ids_lengths[:, :max_chars]

        char_feat_valid = char_feat_2d[valid_char_mask]
        ids_valid = ids_label_2d[valid_char_mask]
        ids_len_valid = ids_len_2d[valid_char_mask]

        if char_feat_valid.numel() > 0:
            style_code_valid = None
            if self.use_line_style_film:
                style_valid = self._expand_style_to_valid_chars(
                    style_line,
                    text_lengths,
                    max_chars,
                )
                style_code_valid = self._build_char_aware_style_code(
                    style_valid,
                    char_feat_valid,
                )

            ids_memory_valid = self._build_ids_memory(
                char_feat_valid,
                style_code=style_code_valid,
            )
            max_ids = int(ids_len_valid.max().item())
            tgt_ids = ids_valid[:, :2 + max_ids]
            logits_ids = self._decode_ids_ar_train(ids_memory_valid, tgt_ids)
            ids_loss = self._ids_ar_loss(logits_ids, tgt_ids)

            with torch.no_grad():
                ids_decoded_valid, _ = self._ids_ar_greedy_decode(ids_memory_valid)
        else:
            ids_loss = torch.tensor(0.0, device=device)
            ids_decoded_valid = []

        batch_size = memory.size(0)
        all_char_ids_train: List[List[List[int]]] = []
        idx = 0
        for b in range(batch_size):
            num_valid = min(int(text_lengths[b].item()), max_chars)
            all_char_ids_train.append(ids_decoded_valid[idx:idx + num_valid])
            idx += num_valid

        struct_mem = self._build_struct_memory(char_feat_2d, valid_logits)

        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, :2 + max_text]
        text_hidden = self._decode_text_with_struct(
            tgt=tgt_text,
            memory=memory,
            h=h,
            mem_mask=mem_mask,
            struct_mem=struct_mem,
        )
        logits_text = self.proj_text(text_hidden)

        zero = torch.tensor(0.0, device=device)
        return logits_text, (ids_loss, all_char_ids_train, valid_loss), char_feat, zero, max_text

    def forward_test(self, x):
        if not self.enable_ids_branch:
            return super().forward_test(x)

        memory, h, mem_mask, feat2d = self._prep_memory(x)
        batch_size = memory.size(0)
        device = memory.device

        max_chars = self.max_text_len
        char_feat_all, valid_logits = self._extract_char_features(
            memory,
            mem_mask,
            max_chars,
            lengths=None,
        )
        char_feat_2d = char_feat_all[:, :, 0, :]

        if self.use_valid_head:
            valid_len_pred, valid_char_mask = self._prefix_lengths_from_valid_logits(valid_logits)
        else:
            valid_len_pred = torch.full(
                (batch_size,),
                max_chars,
                dtype=torch.long,
                device=device,
            )
            valid_char_mask = torch.ones(
                (batch_size, max_chars),
                dtype=torch.bool,
                device=device,
            )
        valid_len_pred = torch.clamp(valid_len_pred, min=0, max=max_chars)

        char_feat_valid = char_feat_2d[valid_char_mask]
        if char_feat_valid.numel() > 0:
            style_code_valid = None
            if self.use_line_style_film:
                style_line = self._extract_line_style_code(feat2d, memory, mem_mask)
                style_valid = self._expand_style_to_valid_chars(
                    style_line,
                    valid_len_pred,
                    max_chars,
                )
                style_code_valid = self._build_char_aware_style_code(
                    style_valid,
                    char_feat_valid,
                )

            ids_memory_valid = self._build_ids_memory(
                char_feat_valid,
                style_code=style_code_valid,
            )
            all_ids_valid, char_conf_valid = self._ids_ar_greedy_decode(ids_memory_valid)
        else:
            all_ids_valid = []
            char_conf_valid = torch.zeros((0,), device=device)

        decoded_char_ids = []
        all_char_conf = []
        idx = 0
        for b in range(batch_size):
            num_valid = min(int(valid_len_pred[b].item()), max_chars)
            decoded_char_ids.append(all_ids_valid[idx:idx + num_valid])
            all_char_conf.append(char_conf_valid[idx:idx + num_valid].tolist())
            idx += num_valid

        ids_bonus = self._build_ids_bonus(decoded_char_ids, all_char_conf)
        struct_mem = self._build_struct_memory(char_feat_2d, valid_logits)

        tgt = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        probs_text_steps = []

        for step_idx in range(self.max_text_len + 1):
            hidden = self._decode_text_with_struct(
                tgt=tgt,
                memory=memory,
                h=h,
                mem_mask=mem_mask,
                struct_mem=struct_mem,
            )

            logits_i = self.proj_text(hidden[:, -1:, :])

            if ids_bonus is not None and step_idx < self.max_text_len:
                base_top1 = logits_i.squeeze(1).argmax(dim=-1)
                step_bonus = ids_bonus[:, step_idx, :].clone()
                step_bonus[base_top1.eq(self.eos_id)] = 0.0
                logits_i = logits_i + step_bonus.unsqueeze(1)

            probs_i = F.softmax(logits_i, dim=-1)
            probs_text_steps.append(probs_i)

            if step_idx < self.max_text_len:
                next_token = logits_i.squeeze(1).argmax(dim=-1)
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
                if tgt.eq(self.eos_id).any(dim=-1).all():
                    break

        probs_text = torch.cat(probs_text_steps, dim=1)

        pred_tokens = tgt[:, 1:]
        eos_mask = pred_tokens.eq(self.eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.zeros((batch_size,), dtype=torch.long, device=device)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)
        text_len_pred = torch.where(
            has_eos,
            first_eos,
            torch.full(
                (batch_size,),
                pred_tokens.size(1),
                dtype=torch.long,
                device=device,
            ),
        )
        text_len_pred = torch.clamp(text_len_pred, min=0)

        all_char_ids = []
        for b in range(batch_size):
            n_final = min(int(text_len_pred[b].item()), max_chars)
            n_ids = min(int(valid_len_pred[b].item()), max_chars)
            cur = decoded_char_ids[b][:min(n_final, n_ids)]
            if n_final > len(cur):
                cur = cur + ([[]] * (n_final - len(cur)))
            all_char_ids.append(cur)

        return probs_text, all_char_ids, text_len_pred
