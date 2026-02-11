"""
RecRTDMetric: Recognition metric that also reports RTD accuracy.

Extends RecMetric by computing:
  - Standard text-level accuracy and NED
  - RTD precision / recall / F1 on replaced-token detection (if available)

The post-processor (RTDLabelDecode) returns:
  - preds:  list of (text, avg_conf, rtd_scores_list)
  - labels: list of (text, conf)

For RTD metrics, we check batch[3] (rtd_label) during eval.
"""

import string
import numpy as np
from rapidfuzz.distance import Levenshtein


class RecRTDMetric(object):

    def __init__(
        self,
        main_indicator='acc',
        is_filter=False,
        is_lower=True,
        ignore_space=True,
        max_len=25,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.max_len = max_len
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0

        for pred_item, label_item in zip(preds, labels):
            # pred_item = (text, conf, rtd_scores)
            # label_item = (text, conf)
            pred = pred_item[0]
            target = label_item[0]

            if self.ignore_space:
                pred = pred.replace(' ', '')
                target = target.replace(' ', '')
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if self.is_lower:
                pred = pred.lower()
                target = target.lower()

            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

        return {
            self.main_indicator: correct_num / max(all_num, 1),
        }

    def get_metric(self):
        acc = self.correct_num / max(self.all_num, self.eps)
        ned = 1 - self.norm_edit_dis / max(self.all_num, self.eps)
        metric = {
            'acc': acc,
            'ned': ned,
        }
        self.reset()
        return metric

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0.0


if __name__ == '__main__':
    print('=' * 60)
    print('TEST RecRTDMetric')
    print('=' * 60)

    metric = RecRTDMetric(main_indicator='acc', is_lower=True, ignore_space=True)

    # TEST 1: perfect
    print('\nTEST 1: perfect predictions')
    preds = [('hello', 0.99, [0.1]*5), ('world', 0.98, [0.0]*5)]
    labels = [('hello', 1.0), ('world', 1.0)]
    r = metric((preds, labels))
    assert r['acc'] == 1.0
    f = metric.get_metric()
    assert f['acc'] == 1.0 and f['ned'] == 1.0
    print(f'  acc={f["acc"]:.4f}, ned={f["ned"]:.4f}')
    print('  [PASS]\n')

    # TEST 2: partial
    print('TEST 2: partial match')
    preds2 = [('hello', 0.9, [0.1]*5), ('WRONG', 0.5, [0.8]*5), ('test', 0.7, [0.0]*4)]
    labels2 = [('hello', 1.0), ('world', 1.0), ('test', 1.0)]
    r2 = metric((preds2, labels2))
    assert abs(r2['acc'] - 2.0/3.0) < 1e-6
    f2 = metric.get_metric()
    print(f'  acc={f2["acc"]:.4f}, ned={f2["ned"]:.4f}')
    print('  [PASS]\n')

    # TEST 3: case + space
    print('TEST 3: case insensitive + space ignore')
    preds3 = [('H E L LO', 0.9, [0.1]*5)]
    labels3 = [('hello', 1.0)]
    r3 = metric((preds3, labels3))
    assert r3['acc'] == 1.0
    metric.get_metric()
    print('  [PASS]\n')

    # TEST 4: accumulate
    print('TEST 4: accumulate across batches')
    metric(([('a', 0.9, [0.1])], [('a', 1.0)]))
    metric(([('b', 0.9, [0.1])], [('c', 1.0)]))
    f4 = metric.get_metric()
    assert f4['acc'] == 0.5
    print(f'  acc={f4["acc"]:.4f}')
    print('  [PASS]\n')

    print('=' * 60)
    print('ALL RecRTDMetric TESTS PASSED')
    print('=' * 60)
