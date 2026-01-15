import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class TAMERLabelDecode(BaseRecLabelDecode):
    """贪心解码：忽略 <pad>(0)/<sos>(1)，遇到 <eos>(2) 截断。"""

    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        super(TAMERLabelDecode, self).__init__(character_dict_path,
                                               use_space_char)

    def __call__(self, preds, batch=None, training=False, *args, **kwargs):
        # 支持训练阶段 (logits, sim) 元组：取 logits 解码
        preds = preds['res'] if isinstance(preds, dict) and 'res' in preds else preds
        if isinstance(preds, (tuple, list)) and len(preds) >= 1:
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        # preds: [B, T, V] 概率或 logits（若是 logits，argmax 等效）
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob)
        if batch is None:
            return text
        # label 也按相同规则去掉起始 <sos>
        label = batch[1]
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        label = label[:, 1:]  # 去掉 <sos>
        label_text = self.decode(label)
        return text, label_text

    def add_special_char(self, dict_character):
        # 与预处理一致：<pad>=0, <sos>=1, <eos>=2, <unk>=3
        return ["<pad>", "<sos>", "<eos>", "<unk>"] + dict_character

    def decode(self, text_index, text_prob=None):
        result_list = []
        batch_size = len(text_index)
        for b in range(batch_size):
            chars = []
            confs = []
            for idx, t in enumerate(text_index[b]):
                t = int(t)
                if t == 0:  # <pad>
                    continue
                if t == 1:  # <sos>
                    continue
                if t == 2:  # <eos>
                    break
                if t < len(self.character):
                    ch = self.character[t]
                    chars.append(ch)
                    confs.append(text_prob[b][idx] if text_prob is not None else 1.0)
            text = ''.join(chars)
            conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
            result_list.append((text, conf))
        return result_list
