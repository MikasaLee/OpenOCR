"""
RTDLabelDecode: Post-processor for NRTRRTDDecoder.

Decodes AR recognition output into text, and extracts per-character RTD
mismatch scores.

Returns:
  - During eval with labels:  (preds_list, labels_list)
    where each pred is (text, avg_conf, rtd_scores_list)
  - During inference (no labels): preds_list
    where each pred is (text, avg_conf, rtd_scores_list)
"""

import numpy as np
import torch

from .ctc_postprocess import BaseRecLabelDecode


class RTDLabelDecode(BaseRecLabelDecode):

    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=True,
                 **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, batch=None, *args, **kwargs):
        # preds is a dict {'rec_pred': ..., 'rtd_pred': ...}
        rec_preds = preds['rec_pred']
        rtd_preds = preds['rtd_pred']

        if isinstance(rec_preds, torch.Tensor):
            rec_preds = rec_preds.detach().cpu().numpy()
        if isinstance(rtd_preds, torch.Tensor):
            rtd_preds = rtd_preds.detach().cpu().numpy()

        preds_idx = rec_preds.argmax(axis=2)
        preds_prob = rec_preds.max(axis=2)

        text_results = self.decode(preds_idx, preds_prob, rtd_preds)

        if batch is None:
            return text_results

        # decode labels
        label = batch[1]
        label_results = self.decode_label(label)
        return text_results, label_results

    def decode(self, text_index, text_prob=None, rtd_scores=None):
        """Decode recognition indices into text + per-char RTD scores."""
        result_list = []
        batch_size = len(text_index)
        for b in range(batch_size):
            char_list = []
            conf_list = []
            rtd_list = []
            for idx in range(len(text_index[b])):
                try:
                    char_idx = self.character[int(text_index[b][idx])]
                except Exception:
                    continue
                if char_idx == self.EOS:
                    break
                if char_idx == self.BOS or char_idx == self.PAD:
                    continue
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(float(text_prob[b][idx]))
                else:
                    conf_list.append(1.0)
                # RTD score: position idx in rec output corresponds to
                # position idx+1 in decoder hidden (because rec predicts
                # next token). So RTD score for this char comes from
                # rtd_scores[b][idx+1] during training forward,
                # but during inference the decoder already aligns them.
                if rtd_scores is not None:
                    # During inference: rtd_scores[b] has same length as logits
                    # Position 0 = BOS step, position idx = step that produced
                    # the char at rec position idx-1. But our forward_test
                    # collects RTD at each step, and step i's RTD reflects
                    # the input token at step i. So for the char generated
                    # at step i (rec logits position i), the RTD score
                    # for the INPUT at step i is rtd_scores[b][i].
                    # The input at step i is the char from step i-1's output.
                    # So rtd_scores[b][i] judges the char at rec position i-1.
                    # For rec position 0 (first output char), RTD is at
                    # position 1 (judging that char as input at step 1).
                    rtd_idx = idx + 1 if idx + 1 < len(rtd_scores[b]) else idx
                    rtd_list.append(float(rtd_scores[b][rtd_idx]))
                else:
                    rtd_list.append(0.0)
            text = ''.join(char_list)
            avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
            result_list.append((text, avg_conf, rtd_list))
        return result_list

    def decode_label(self, label):
        """Decode GT label indices into text (for metric computation)."""
        result_list = []
        batch_size = len(label)
        for b in range(batch_size):
            char_list = []
            for idx in range(len(label[b])):
                try:
                    char_idx = self.character[int(label[b][idx])]
                except Exception:
                    continue
                if char_idx == self.EOS:
                    break
                if char_idx == self.BOS or char_idx == self.PAD:
                    continue
                char_list.append(char_idx)
            text = ''.join(char_list)
            result_list.append((text, 1.0))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = [self.EOS] + dict_character + [self.BOS, self.PAD]
        return dict_character


if __name__ == '__main__':
    import sys, os
    DICT_PATH = './tools/utils/dict/visual_c3_ids/char_dict.txt'
    if not os.path.exists(DICT_PATH):
        print(f'[SKIP] {DICT_PATH} not found'); sys.exit(0)

    print('=' * 60)
    print('TEST RTDLabelDecode')
    print('=' * 60)

    dec = RTDLabelDecode(character_dict_path=DICT_PATH, use_space_char=True)
    vocab_size = len(dec.character)
    BOS_IDX = vocab_size - 2
    EOS_IDX = 0
    PAD_IDX = vocab_size - 1

    # TEST 1: decode from logits
    print('\nTEST 1: decode from logits')
    B, T, num_cls = 2, 5, vocab_size - 2
    rec_pred = np.zeros((B, T, num_cls), dtype=np.float32)
    target_0 = [1, 2, 3, 0]
    for t, idx in enumerate(target_0):
        rec_pred[0, t, idx] = 10.0
    for t in range(len(target_0), T):
        rec_pred[0, t, 1] = 10.0
    target_1 = [5, 6, 0]
    for t, idx in enumerate(target_1):
        rec_pred[1, t, idx] = 10.0
    for t in range(len(target_1), T):
        rec_pred[1, t, 1] = 10.0

    rtd_pred = np.random.randn(B, T).astype(np.float32)
    preds = {'rec_pred': rec_pred, 'rtd_pred': rtd_pred}
    results = dec(preds, batch=None)
    assert len(results) == B
    for i, (text, conf, rtd_scores) in enumerate(results):
        print(f'  Sample {i}: text="{text}", conf={conf:.3f}, rtd_len={len(rtd_scores)}')
    assert len(results[0][0]) == 3
    assert len(results[1][0]) == 2
    print('  [PASS] Decode logits OK\n')

    # TEST 2: with label (eval mode)
    print('TEST 2: decode + label')
    max_text_len = 15
    label = np.full((B, max_text_len + 2), PAD_IDX, dtype=np.int64)
    label[0, :5] = [BOS_IDX, 1, 2, 3, EOS_IDX]
    label[1, :4] = [BOS_IDX, 5, 6, EOS_IDX]
    batch = [None, label, None, None]
    text_results, label_results = dec(preds, batch=batch)
    assert len(label_results) == B
    for i, (text, conf) in enumerate(label_results):
        print(f'  Label {i}: text="{text}"')
    print('  [PASS] Decode + label OK\n')

    # TEST 3: torch tensor input
    print('TEST 3: torch tensor input')
    import torch as th
    preds_t = {'rec_pred': th.from_numpy(rec_pred), 'rtd_pred': th.from_numpy(rtd_pred)}
    results_t = dec(preds_t, batch=None)
    assert results_t[0][0] == results[0][0]
    print('  [PASS] Torch input OK\n')

    print('=' * 60)
    print('ALL RTDLabelDecode TESTS PASSED')
    print('=' * 60)
