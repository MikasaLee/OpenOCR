import os
import sys
import copy
from pathlib import Path

import torch
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader  # noqa: E402
from tools.engine.config import Config  # noqa: E402
from tools.engine.trainer import Trainer  # noqa: E402
from tools.utility import ArgsParser  # noqa: E402


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--benchmark_root', type=str, required=True,
                        help='Root dir of benchmark, each subset has label.txt')
    parser.add_argument('--subsets', type=str, default=None,
                        help='Comma separated subset folder names; default all subfolders of benchmark_root')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Override Global.pretrained_model')
    parser.add_argument('--save_xlsx', type=str, default=None,
                        help='Path to save xlsx with preds+summary. Default: <output_dir>/crnn_benchmark.xlsx')
    parser.add_argument('--s_weight', type=str, default=None,
                        help='Comma weights for subsets, same order as --subsets')
    return parser.parse_args()


def prepare_cfg(cfg, pretrained_override=None):
    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if pretrained_override:
        cfg.cfg['Global']['pretrained_model'] = pretrained_override
    elif cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    keep_keys = cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys']['keep_keys']
    if 'real_ratio' not in keep_keys:
        keep_keys.append('real_ratio')
    return cfg


def build_eval_cfg(base_cfg, subset_dir, label_file):
    cfg_each = copy.deepcopy(base_cfg)
    cfg_each['Eval']['dataset']['name'] = 'SimpleDataSet'
    cfg_each['Eval']['dataset']['data_dir'] = subset_dir
    cfg_each['Eval']['dataset']['label_file_list'] = [label_file]
    cfg_each['Eval']['loader']['shuffle'] = False
    cfg_each['Eval']['loader']['drop_last'] = False
    cfg_each['Eval']['loader']['num_workers'] = min(2, cfg_each['Eval']['loader'].get('num_workers', 4))
    if 'sampler' in cfg_each['Eval']:
        cfg_each['Eval'].pop('sampler')
    return cfg_each


def run_subset(trainer, cfg_each, dataset_name, rows):
    valid_loader = build_dataloader(cfg_each, 'Eval', trainer.logger)
    trainer.valid_dataloader = valid_loader
    trainer.eval_data_name = dataset_name
    model = trainer.model
    device = trainer.device
    post_process = trainer.post_process_class
    model.eval()
    true_num = 0
    ned_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_loader, desc=f'eval {dataset_name}', leave=True)):
            batch_tensor = [t.to(device) for t in batch]
            batch_numpy = [t.numpy() for t in batch]
            preds = model(batch_tensor[0], data=batch_tensor[1:])
            post_result = post_process(preds, batch_numpy)
            if isinstance(post_result, tuple):
                texts, gts = post_result
            else:
                texts, gts = post_result, None
            for i, (txt, prob) in enumerate(texts):
                gt_text = gts[i][0] if gts is not None and i < len(gts) else ''
                ned = 1 - Levenshtein.normalized_distance(txt, gt_text) if gt_text else 0.0
                true_num += int(ned == 1)
                ned_list.append(ned)
                rows.append([dataset_name, f"{dataset_name}_{batch_idx}_{i}", gt_text, txt, float(prob), float(ned)])
    num = len(ned_list)
    acc = true_num / num if num else 0.0
    ned_mean = sum(ned_list) / num if num else 0.0
    model.train()
    return acc, ned_mean, num


def parse_weights(weights_str, n):
    if not weights_str:
        return [1.0] * n
    ws = [float(x) for x in weights_str.split(',') if x.strip()]
    if len(ws) != n:
        print(f'[WARN] s_weight length {len(ws)} != subsets {n}, fallback to all-ones')
        return [1.0] * n
    return ws


def main():
    args = parse_args()
    cfg = Config(args.config)
    cfg.merge_dict(vars(args))
    cfg = prepare_cfg(cfg, args.pretrained_model)

    benchmark_root = Path(args.benchmark_root)
    if args.subsets:
        subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]
    else:
        subsets = [p.name for p in benchmark_root.iterdir() if p.is_dir()]
    s_weight = parse_weights(args.s_weight, len(subsets))

    save_xlsx = args.save_xlsx or os.path.join(cfg.cfg['Global']['output_dir'], 'crnn_benchmark.xlsx')
    os.makedirs(os.path.dirname(save_xlsx), exist_ok=True)

    trainer = Trainer(cfg, mode='eval')

    rows = []
    per_subset = []
    total_true = 0.0
    total_num = 0
    total_ned_sum = 0.0

    for subset in subsets:
        subset_dir = benchmark_root / subset
        label_file = subset_dir / 'label.txt'
        if not label_file.exists():
            print(f'[WARN] skip {subset}: no label.txt')
            continue
        cfg_each = build_eval_cfg(cfg.cfg, str(subset_dir), str(label_file))
        acc, ned_mean, num = run_subset(trainer, cfg_each, subset, rows)
        per_subset.append((subset, acc, ned_mean, num))
        total_true += acc * num
        total_num += num
        total_ned_sum += ned_mean * num
        print(f"{subset}:\t acc: {acc*100:.3f}, norm_edit_dis:{ned_mean*100:.3f}")

    if total_num > 0:
        total_acc = total_true / total_num
        total_ned = total_ned_sum / total_num
    else:
        total_acc = 0.0
        total_ned = 0.0

    pn_list = [p[1] for p in per_subset]
    ned_list = [p[2] for p in per_subset]
    if per_subset:
        s_mean_acc = sum(pn_list) / len(pn_list)
        s_mean_ned = sum(ned_list) / len(ned_list)
        s_weight_acc = sum([a * w for a, w in zip(pn_list, s_weight)])
        s_weight_ned = sum([n * w for n, w in zip(ned_list, s_weight)])
    else:
        s_mean_acc = s_mean_ned = s_weight_acc = s_weight_ned = 0.0

    for name, acc, ned_mean, _ in per_subset:
        print(f"{name}:\t acc: {acc*100:.3f}, norm_edit_dis:{ned_mean*100:.3f}")
    print(f"total:\t acc: {total_acc*100:.3f}, norm_edit_dis:{total_ned*100:.3f}")
    print(f"S_mean:\t acc: {s_mean_acc*100:.3f}, norm_edit_dis:{s_mean_ned*100:.3f}")
    print(f"S_weight:\t acc: {s_weight_acc*100:.3f}, norm_edit_dis:{s_weight_ned*100:.3f}")

    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=['dataset', 'sample_id', 'label', 'pred', 'conf', 'ned'])
        summary_df = pd.DataFrame([
            *[(name, acc, ned_mean, num) for name, acc, ned_mean, num in per_subset],
            ('total', total_acc, total_ned, total_num),
            ('S_mean', s_mean_acc, s_mean_ned, len(per_subset)),
            ('S_weight', s_weight_acc, s_weight_ned, len(per_subset)),
        ], columns=['dataset', 'acc', 'ned_mean', 'num'])
        with pd.ExcelWriter(save_xlsx) as writer:
            df.to_excel(writer, index=False, sheet_name='preds')
            summary_df.to_excel(writer, index=False, sheet_name='summary')
        print(f'Saved xlsx to {save_xlsx}')
    except Exception as e:
        print(f'[WARN] Failed to save XLSX ({e}). Install pandas & openpyxl to enable XLSX export.')


if __name__ == '__main__':
    main()
