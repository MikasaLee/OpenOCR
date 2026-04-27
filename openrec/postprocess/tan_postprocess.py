from pathlib import Path

import numpy as np
import torch


class TANLabelDecode(object):

    def __init__(self,
                 character_dict_path,
                 relation_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.character = self._load_dict(character_dict_path)
        self.relations = (self._load_dict(relation_dict_path)
                          if relation_dict_path is not None else [])

    @staticmethod
    def _load_dict(path):
        data = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            if ' ' in line:
                token = line.rsplit(' ', 1)[0]
            else:
                token = line
            data.append(token)
        return data

    def get_character_num(self):
        return len(self.character)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _decode_seq(self, token_ids):
        tokens = []
        for tid in token_ids:
            tid = int(tid)
            if tid < 0 or tid >= len(self.character):
                continue
            tokens.append(self.character[tid])
        return ' '.join(tokens)

    def __call__(self, preds, batch=None, *args, **kwargs):
        logits = preds['char_logits']
        if isinstance(logits, torch.Tensor):
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).detach().cpu().numpy()
            pred_prob = probs.max(dim=-1).values.detach().cpu().numpy()
        else:
            logits = np.asarray(logits)
            logits = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
            pred_idx = probs.argmax(axis=-1)
            pred_prob = probs.max(axis=-1)

        pred_idx = np.asarray(pred_idx).transpose(1, 0)
        pred_prob = np.asarray(pred_prob).transpose(1, 0)

        if batch is None:
            result = []
            for seq_ids, seq_prob in zip(pred_idx, pred_prob):
                result.append((self._decode_seq(seq_ids), float(np.mean(seq_prob))))
            return result

        gt_ids = self._to_numpy(batch[2])
        gt_mask = self._to_numpy(batch[3])
        if gt_ids.ndim == 2:
            gt_ids = gt_ids
            gt_mask = gt_mask
        else:
            raise ValueError('TANLabelDecode expects gt ids/mask with shape [B, L] or [L, B].')

        if gt_ids.shape[0] != pred_idx.shape[0] and gt_ids.shape[1] == pred_idx.shape[0]:
            gt_ids = gt_ids.transpose(1, 0)
            gt_mask = gt_mask.transpose(1, 0)

        text = []
        label = []
        for pred_seq, pred_seq_prob, gt_seq, gt_seq_mask in zip(
                pred_idx, pred_prob, gt_ids, gt_mask):
            valid_len = int(gt_seq_mask.sum())
            valid_len = max(valid_len, 0)
            pred_tokens = pred_seq[:valid_len]
            pred_conf = pred_seq_prob[:valid_len]
            gt_tokens = gt_seq[:valid_len]
            text.append((self._decode_seq(pred_tokens),
                         float(np.mean(pred_conf)) if len(pred_conf) else 0.0))
            label.append((self._decode_seq(gt_tokens), 1.0))
        return text, label
