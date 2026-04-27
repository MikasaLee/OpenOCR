import copy
import json
import os
import sys
import time
from collections import OrderedDict

import lmdb
import numpy as np
import torch
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import eval_rec_textline_triple as triple_base
import eval_rec_textline_triple_eff as eff_base

from openrec.preprocess import create_operators, transform
from openrec.preprocess.tan_label_encode import TANLabelEncode
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--eval_data_dirs',
        nargs='*',
        default=None,
        help='Optional char-LMDB dirs to evaluate. If omitted, use Eval.dataset.data_dir/data_dir_list from config.',
    )
    parser.add_argument(
        '--char_to_ids_path',
        type=str,
        default='./tools/utils/dict/visual_c3_ids/char_to_ids.txt',
        help='Path to char<tab>ids mapping for runtime label construction and ids->char recovery.',
    )
    parser.add_argument(
        '--runtime_label_field',
        type=str,
        default='tgt',
        choices=['src', 'tgt'],
        help='Which char label is converted to IDS to drive the current TAN forward path.',
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=0,
        help='Maximum char samples to evaluate per dataset. 0 means all.',
    )
    parser.add_argument(
        '--save_eff_json',
        type=str,
        default=None,
        help='Optional path to save TAN line-efficiency summaries as JSON.',
    )
    parser.add_argument(
        '--unmapped_char',
        type=str,
        default='?',
        help='Fallback character when predicted IDS cannot be mapped back to a character.',
    )
    return parser.parse_args()


def prepare_cfg(cfg):
    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global'].get('pretrained_model') is None:
        cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    return cfg


def is_lmdb_dir(path):
    return os.path.isfile(os.path.join(path, 'data.mdb'))


def collect_leaf_lmdb_dirs(path):
    if is_lmdb_dir(path):
        return [path]
    out = []
    for dirpath, _dirnames, filenames in os.walk(path):
        if 'data.mdb' in filenames:
            out.append(dirpath)
    return sorted(out)


def resolve_eval_data_dirs(cfg, eval_data_dirs=None):
    if eval_data_dirs:
        roots = eval_data_dirs
    else:
        dataset_cfg = cfg.cfg['Eval']['dataset']
        if dataset_cfg.get('data_dir_list'):
            roots = dataset_cfg['data_dir_list']
        elif dataset_cfg.get('data_dir'):
            roots = [dataset_cfg['data_dir']]
        else:
            raise ValueError('No eval data dirs found in config and --eval_data_dirs not provided.')

    out = []
    for root in roots:
        out.extend(collect_leaf_lmdb_dirs(root))
    if not out:
        raise ValueError(f'No LMDB dirs found from: {roots}')
    return out


def load_lmdb(path):
    env = lmdb.open(
        path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False)
    num_samples = int(txn.get(b'num-samples'))
    return env, txn, num_samples


def parse_compound_label(raw_label, index, dataset_name):
    parts = str(raw_label).split('\t')
    if len(parts) >= 3:
        src_char = parts[0]
        tgt_char = parts[1]
        line_seg_id = '\t'.join(parts[2:])
    elif len(parts) == 2:
        src_char = parts[0]
        tgt_char = parts[0]
        line_seg_id = parts[1]
    else:
        raise ValueError(
            f'{dataset_name} sample#{index} label must contain at least 2 fields '
            f'(src\\ttgt\\tline_seg_id or char\\tline_seg_id), got: {raw_label!r}')
    return src_char, tgt_char, line_seg_id


def tokenize_ids(text):
    text = str(text).strip()
    if not text:
        return []
    if ' ' in text:
        return [tok for tok in text.split(' ') if tok]
    return list(text)


def normalize_ids_str(text):
    return ' '.join(tokenize_ids(text))


def load_char_to_ids(path):
    char_to_ids = {}
    ids_to_char = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or '\t' not in line:
                continue
            ch, ids_str = line.split('\t', 1)
            ids_norm = normalize_ids_str(ids_str.strip())
            if not ids_norm:
                continue
            char_to_ids[ch] = ids_norm
            if ids_norm not in ids_to_char:
                ids_to_char[ids_norm] = ch
    return char_to_ids, ids_to_char


def build_image_ops(cfg):
    transforms = copy.deepcopy(cfg.cfg['Eval']['dataset']['transforms'])
    filtered = []
    for op_info in transforms:
        op_name = list(op_info.keys())[0]
        if op_name == 'KeepKeys':
            continue
        if 'LabelEncode' in op_name:
            continue
        filtered.append(op_info)
    if not filtered:
        raise ValueError('No image preprocessing ops left after filtering label transforms.')
    return create_operators(filtered, cfg.cfg['Global'])


def build_runtime_label_encoder(cfg):
    return TANLabelEncode(
        max_text_length=cfg.cfg['Global']['max_text_length'],
        character_dict_path=cfg.cfg['Global']['character_dict_path'],
        relation_dict_path=cfg.cfg['Global'].get('relation_dict_path'),
        use_space_char=cfg.cfg['Global'].get('use_space_char', False),
    )


def preprocess_image(image_bin, image_ops):
    data = transform({'image': image_bin}, image_ops)
    if data is None or 'image' not in data:
        raise ValueError('Image preprocess returned None.')
    return data['image']


def build_runtime_batch(ids_label, label_encoder, device):
    encoded = label_encoder({'label': ids_label})
    if encoded is None:
        raise ValueError(f'Failed to TAN-encode ids label: {ids_label!r}')

    label = torch.from_numpy(encoded['label']).unsqueeze(0).to(device)
    length = torch.tensor([int(encoded['length'])], dtype=torch.long, device=device)
    ly = torch.from_numpy(encoded['ly']).unsqueeze(0).to(device)
    ly_mask = torch.from_numpy(encoded['ly_mask']).unsqueeze(0).to(device)
    ry = torch.from_numpy(encoded['ry']).unsqueeze(0).to(device)
    ry_mask = torch.from_numpy(encoded['ry_mask']).unsqueeze(0).to(device)
    lp = torch.from_numpy(encoded['lp']).unsqueeze(0).to(device)
    rp = torch.from_numpy(encoded['rp']).unsqueeze(0).to(device)
    re = torch.from_numpy(encoded['re']).unsqueeze(0).to(device)
    rre = torch.from_numpy(encoded['rre']).unsqueeze(0).to(device)
    rre_mask = torch.from_numpy(encoded['rre_mask']).unsqueeze(0).to(device)
    return [label, length, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask], int(encoded['length'])


def decode_char_logits(preds, token_list, valid_len):
    logits = preds['char_logits']
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1)[:, 0].detach().cpu().tolist()
    pred_conf = probs.max(dim=-1).values[:, 0].detach().cpu().tolist()
    pred_idx = pred_idx[:valid_len]
    pred_conf = pred_conf[:valid_len]
    tokens = []
    for tid in pred_idx:
        if 0 <= int(tid) < len(token_list):
            tokens.append(token_list[int(tid)])
    ids_str = ' '.join(tokens)
    conf = float(np.mean(pred_conf)) if pred_conf else 0.0
    return ids_str, conf


def init_tan_efficiency_stats(dataset_name):
    return {
        'dataset_name': dataset_name,
        'num_chars': 0,
        'num_lines': 0,
        'total_data': 0.0,
        'total_label_prep': 0.0,
        'total_model': 0.0,
        'total_decode': 0.0,
        'total_line_finalize': 0.0,
        'char_model_ms': [],
        'char_e2e_ms': [],
        'line_model_ms': [],
        'line_e2e_ms': [],
        'char_counts_per_line': [],
        'num_unmapped_pred_chars': 0,
        'num_skipped_chars': 0,
    }


def finalize_tan_efficiency_stats(stats):
    summary = dict(stats)
    total_eval = summary['total_data'] + summary['total_model'] + summary['total_decode'] + summary['total_line_finalize']
    summary['total_eval'] = total_eval
    summary['script_wall_total'] = total_eval + summary['total_label_prep']
    summary['char_throughput'] = triple_base._safe_div(summary['num_chars'], total_eval)
    summary['line_throughput'] = triple_base._safe_div(summary['num_lines'], total_eval)
    summary['line_throughput_model_only'] = triple_base._safe_div(summary['num_lines'], summary['total_model'])
    summary['avg_char_e2e_ms'] = None if summary['num_chars'] == 0 else total_eval / summary['num_chars'] * 1000.0
    summary['avg_char_model_ms'] = None if summary['num_chars'] == 0 else summary['total_model'] / summary['num_chars'] * 1000.0
    summary['avg_char_data_ms'] = None if summary['num_chars'] == 0 else summary['total_data'] / summary['num_chars'] * 1000.0
    summary['avg_char_decode_ms'] = None if summary['num_chars'] == 0 else summary['total_decode'] / summary['num_chars'] * 1000.0
    summary['avg_char_label_prep_ms'] = None if summary['num_chars'] == 0 else summary['total_label_prep'] / summary['num_chars'] * 1000.0
    summary['avg_line_e2e_ms'] = None if summary['num_lines'] == 0 else total_eval / summary['num_lines'] * 1000.0
    summary['avg_line_model_ms'] = None if summary['num_lines'] == 0 else summary['total_model'] / summary['num_lines'] * 1000.0
    summary['avg_chars_per_line'] = None if summary['num_lines'] == 0 else summary['num_chars'] / summary['num_lines']

    for key, q in [('p50_char_model_ms', 50), ('p90_char_model_ms', 90)]:
        summary[key] = None if not summary['char_model_ms'] else float(np.percentile(summary['char_model_ms'], q))
    for key, q in [('p50_line_model_ms', 50), ('p90_line_model_ms', 90)]:
        summary[key] = None if not summary['line_model_ms'] else float(np.percentile(summary['line_model_ms'], q))
    summary['max_chars_per_line'] = 0 if not summary['char_counts_per_line'] else int(max(summary['char_counts_per_line']))
    return summary


def merge_tan_efficiency_stats(dst, src):
    if dst is None:
        merged = init_tan_efficiency_stats('total')
    else:
        merged = dst
    merged['num_chars'] += src['num_chars']
    merged['num_lines'] += src['num_lines']
    merged['total_data'] += src['total_data']
    merged['total_label_prep'] += src['total_label_prep']
    merged['total_model'] += src['total_model']
    merged['total_decode'] += src['total_decode']
    merged['total_line_finalize'] += src['total_line_finalize']
    merged['char_model_ms'].extend(src['char_model_ms'])
    merged['char_e2e_ms'].extend(src['char_e2e_ms'])
    merged['line_model_ms'].extend(src['line_model_ms'])
    merged['line_e2e_ms'].extend(src['line_e2e_ms'])
    merged['char_counts_per_line'].extend(src['char_counts_per_line'])
    merged['num_unmapped_pred_chars'] += src['num_unmapped_pred_chars']
    merged['num_skipped_chars'] += src['num_skipped_chars']
    return merged


def print_tan_efficiency_summary(summary):
    name = summary['dataset_name']
    char_tp = 'N/A' if summary['char_throughput'] is None else f"{summary['char_throughput']:.2f}"
    line_tp = 'N/A' if summary['line_throughput'] is None else f"{summary['line_throughput']:.2f}"
    line_tp_model = 'N/A' if summary['line_throughput_model_only'] is None else f"{summary['line_throughput_model_only']:.2f}"
    avg_chars = 'N/A' if summary['avg_chars_per_line'] is None else f"{summary['avg_chars_per_line']:.2f}"
    avg_char_data = 'N/A' if summary['avg_char_data_ms'] is None else f"{summary['avg_char_data_ms']:.2f}"
    avg_char_model = 'N/A' if summary['avg_char_model_ms'] is None else f"{summary['avg_char_model_ms']:.2f}"
    avg_char_decode = 'N/A' if summary['avg_char_decode_ms'] is None else f"{summary['avg_char_decode_ms']:.2f}"
    avg_line_e2e = 'N/A' if summary['avg_line_e2e_ms'] is None else f"{summary['avg_line_e2e_ms']:.2f}"
    avg_line_model = 'N/A' if summary['avg_line_model_ms'] is None else f"{summary['avg_line_model_ms']:.2f}"
    p50_line_model = 'N/A' if summary['p50_line_model_ms'] is None else f"{summary['p50_line_model_ms']:.2f}"
    p90_line_model = 'N/A' if summary['p90_line_model_ms'] is None else f"{summary['p90_line_model_ms']:.2f}"
    avg_label_prep = 'N/A' if summary['avg_char_label_prep_ms'] is None else f"{summary['avg_char_label_prep_ms']:.2f}"

    print(f"[EFF][{name}] chars={summary['num_chars']}, lines={summary['num_lines']}, avg_chars_per_line={avg_chars}, max_chars_per_line={summary['max_chars_per_line']}")
    print(f"[EFF][{name}] char_end2end_total={summary['total_eval']:.4f}s, char_throughput={char_tp} chars/s")
    print(
        f"[EFF][{name}] avg_char_data={avg_char_data} ms, "
        f"avg_char_model={avg_char_model} ms, "
        f"avg_char_decode={avg_char_decode} ms"
    )
    print(
        f"[EFF][{name}] line_end2end_total={summary['total_eval']:.4f}s, "
        f"throughput_line_e2e={line_tp} lines/s, "
        f"throughput_line_model_only={line_tp_model} lines/s"
    )
    print(
        f"[EFF][{name}] avg_line_e2e={avg_line_e2e} ms, "
        f"avg_line_model={avg_line_model} ms, "
        f"p50_line_model={p50_line_model} ms, "
        f"p90_line_model={p90_line_model} ms"
    )
    print(
        f"[EFF][{name}] aux_label_prep_total={summary['total_label_prep']:.4f}s "
        f"(avg_char={avg_label_prep} ms, excluded from main end2end summary)"
    )
    print(
        f"[EFF][{name}] unmapped_pred_chars={summary['num_unmapped_pred_chars']}, "
        f"skipped_chars={summary['num_skipped_chars']}, "
        f"script_wall_total={summary['script_wall_total']:.4f}s"
    )


def print_tan_table_summary(tag, summary):
    line_latency = 'N/A' if summary.get('avg_line_model_ms') is None else f"{summary['avg_line_model_ms']:.2f}"
    line_tp = 'N/A' if summary.get('line_throughput') is None else f"{summary['line_throughput']:.2f}"
    char_latency = 'N/A' if summary.get('avg_char_model_ms') is None else f"{summary['avg_char_model_ms']:.2f}"
    print(
        f"[EFF-SUMMARY][{tag}] "
        f"latency_line_model={line_latency} ms/line | "
        f"throughput_line_e2e={line_tp} lines/s | "
        f"latency_char_model={char_latency} ms/char"
    )


def evaluate_dataset(
    trainer,
    datadir,
    dataset_name,
    image_ops,
    label_encoder,
    char_to_ids,
    ids_to_char,
    runtime_label_field='tgt',
    max_samples=0,
    unmapped_char='?',
):
    env, txn, num_samples_total = load_lmdb(datadir)
    if max_samples > 0:
        num_samples_total = min(num_samples_total, max_samples)

    model = trainer.model
    device = trainer.device
    token_list = trainer.post_process_class.character

    stats = init_tan_efficiency_stats(dataset_name)
    line_store = OrderedDict()
    det_inputs = {'gts': [], 'preds': []}

    src_true = 0
    tgt_true = 0
    src_ned_sum = 0.0
    tgt_ned_sum = 0.0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(1, num_samples_total + 1), desc=f'eval {dataset_name}', position=0, leave=True)
        for idx in pbar:
            label_key = f'label-{idx:09d}'.encode()
            image_key = f'image-{idx:09d}'.encode()
            raw_label = txn.get(label_key)
            image_bin = txn.get(image_key)
            if raw_label is None or image_bin is None:
                stats['num_skipped_chars'] += 1
                continue

            raw_label = raw_label.decode('utf-8')
            src_char, tgt_char, line_seg_id = parse_compound_label(raw_label, idx, dataset_name)
            runtime_char = tgt_char if runtime_label_field == 'tgt' else src_char
            ids_label = char_to_ids.get(runtime_char)
            if ids_label is None:
                stats['num_skipped_chars'] += 1
                continue

            t_data_start = time.perf_counter()
            image_tensor = preprocess_image(image_bin, image_ops)
            if not torch.is_tensor(image_tensor):
                image_tensor = torch.as_tensor(image_tensor)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            t_data_end = time.perf_counter()

            t_label_start = time.perf_counter()
            batch_data, valid_len = build_runtime_batch(ids_label, label_encoder, device)
            t_label_end = time.perf_counter()

            eff_base.sync_device(device)
            t_model_start = time.perf_counter()
            preds = model(image_tensor, data=batch_data)
            eff_base.sync_device(device)
            t_model_end = time.perf_counter()

            t_decode_start = time.perf_counter()
            pred_ids_str, pred_conf = decode_char_logits(preds, token_list, valid_len)
            pred_char = ids_to_char.get(normalize_ids_str(pred_ids_str), unmapped_char)
            if pred_char == unmapped_char:
                stats['num_unmapped_pred_chars'] += 1
            t_decode_end = time.perf_counter()

            data_time = t_data_end - t_data_start
            label_prep_time = t_label_end - t_label_start
            model_time = t_model_end - t_model_start
            decode_time = t_decode_end - t_decode_start
            e2e_time = data_time + model_time + decode_time

            stats['num_chars'] += 1
            stats['total_data'] += data_time
            stats['total_label_prep'] += label_prep_time
            stats['total_model'] += model_time
            stats['total_decode'] += decode_time
            stats['char_model_ms'].append(model_time * 1000.0)
            stats['char_e2e_ms'].append(e2e_time * 1000.0)

            line_item = line_store.setdefault(line_seg_id, {
                'src_chars': [],
                'tgt_chars': [],
                'pred_chars': [],
                'pred_ids': [],
                'char_model_ms': [],
                'char_e2e_ms': [],
                'char_confs': [],
            })
            line_item['src_chars'].append(src_char)
            line_item['tgt_chars'].append(tgt_char)
            line_item['pred_chars'].append(pred_char)
            line_item['pred_ids'].append(pred_ids_str)
            line_item['char_model_ms'].append(model_time * 1000.0)
            line_item['char_e2e_ms'].append(e2e_time * 1000.0)
            line_item['char_confs'].append(pred_conf)
        pbar.close()

    t_finalize_start = time.perf_counter()
    for line_seg_id, item in line_store.items():
        pred_text = ''.join(item['pred_chars'])
        src_text = ''.join(item['src_chars'])
        tgt_text = ''.join(item['tgt_chars'])

        pred_norm = triple_base.replace_punctuation(pred_text)
        src_norm = triple_base.replace_punctuation(src_text)
        tgt_norm = triple_base.replace_punctuation(tgt_text)

        s_ned = 1 - triple_base.Levenshtein.normalized_distance(pred_norm, src_norm) if src_norm is not None else 0.0
        t_ned = 1 - triple_base.Levenshtein.normalized_distance(pred_norm, tgt_norm) if tgt_norm is not None else 0.0
        src_ned_sum += s_ned
        tgt_ned_sum += t_ned
        if int(s_ned) == 1:
            src_true += 1
        if int(t_ned) == 1:
            tgt_true += 1

        det_inputs['gts'].append(src_norm)
        det_inputs['preds'].append(pred_norm)

        line_model_ms = float(sum(item['char_model_ms']))
        line_e2e_ms = float(sum(item['char_e2e_ms']))
        stats['line_model_ms'].append(line_model_ms)
        stats['line_e2e_ms'].append(line_e2e_ms)
        stats['char_counts_per_line'].append(len(item['src_chars']))

    t_finalize_end = time.perf_counter()
    stats['total_line_finalize'] += (t_finalize_end - t_finalize_start)
    stats['num_lines'] = len(line_store)
    env.close()

    det_metrics = triple_base.calculate_cuo_metric_compact(det_inputs['gts'], det_inputs['preds'], X='X')
    summary = finalize_tan_efficiency_stats(stats)
    return {
        'dataset_name': dataset_name,
        'num_lines': len(line_store),
        'src_true': src_true,
        'tgt_true': tgt_true,
        'src_ned_sum': src_ned_sum,
        'tgt_ned_sum': tgt_ned_sum,
        'src_acc': src_true / len(line_store) if line_store else 0.0,
        'tgt_acc': tgt_true / len(line_store) if line_store else 0.0,
        'src_ned': src_ned_sum / len(line_store) if line_store else 0.0,
        'tgt_ned': tgt_ned_sum / len(line_store) if line_store else 0.0,
        'det_metrics': det_metrics,
        'efficiency': summary,
    }


def fmt_metric(x):
    return "N/A" if x is None else f"{x:.3f}"


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)

    flags = vars(FLAGS)
    opt = flags.pop('opt')
    eval_data_dirs = flags.pop('eval_data_dirs')
    char_to_ids_path = flags.pop('char_to_ids_path')
    runtime_label_field = flags.pop('runtime_label_field')
    max_samples = flags.pop('max_samples')
    save_eff_json = flags.pop('save_eff_json')
    unmapped_char = flags.pop('unmapped_char')

    cfg.merge_dict(flags)
    cfg.merge_dict(opt)
    cfg = prepare_cfg(cfg)

    if save_eff_json:
        save_eff_dir = os.path.dirname(save_eff_json)
        if save_eff_dir:
            os.makedirs(save_eff_dir, exist_ok=True)

    trainer = Trainer(cfg, mode='test')
    trainer.logger.info(
        'TAN line-efficiency evaluation uses sequential char-LMDB reading and '
        'accumulates per-char time into per-line cost.'
    )
    trainer.logger.info(
        'Current OpenOCR TAN adapter has no free-running inference API; '
        f'this script drives the forward path with {runtime_label_field}_char -> IDS '
        'structure labels for runtime measurement.'
    )

    eval_dirs = resolve_eval_data_dirs(cfg, eval_data_dirs)
    char_to_ids, ids_to_char = load_char_to_ids(char_to_ids_path)
    image_ops = build_image_ops(cfg)
    label_encoder = build_runtime_label_encoder(cfg)

    total_eff_stats = None
    total_lines = 0
    total_src_true = 0
    total_tgt_true = 0
    total_src_ned_sum = 0.0
    total_tgt_ned_sum = 0.0
    dataset_payload = {}

    for datadir in eval_dirs:
        dataset_name = datadir[:-1].split('/')[-1] if datadir.endswith('/') else datadir.split('/')[-1]
        result = evaluate_dataset(
            trainer=trainer,
            datadir=datadir,
            dataset_name=dataset_name,
            image_ops=image_ops,
            label_encoder=label_encoder,
            char_to_ids=char_to_ids,
            ids_to_char=ids_to_char,
            runtime_label_field=runtime_label_field,
            max_samples=max_samples,
            unmapped_char=unmapped_char,
        )

        print(
            f"{dataset_name}:\t line_src_acc: {100*result['src_acc']:6g}, "
            f"line_src_NED:{100*result['src_ned']:6g}, "
            f"line_tgt_acc: {100*result['tgt_acc']:6g}, "
            f"line_tgt_NED:{100*result['tgt_ned']:6g}"
        )
        print_tan_efficiency_summary(result['efficiency'])
        print_tan_table_summary(dataset_name, result['efficiency'])

        det = result['det_metrics']
        print("\nCuo detection metrics (line text from TAN chars):")
        if det:
            print(f"N_sent={det['N_sent']} | clean={det['N_clean_sent']} | error={det['N_error_sent']}")
            print(f"Char_P={fmt_metric(det['Char_P'])}%  Char_R={fmt_metric(det['Char_R'])}%  Char_F1={fmt_metric(det['Char_F1'])}%")
            print(f"Char_BalAcc={fmt_metric(det['Char_BalAcc'])}%  Char_MCC={fmt_metric(det['Char_MCC'])}")
            print(f"Sent_FA={fmt_metric(det['Sent_FA'])}%  Sent_EM={fmt_metric(det['Sent_EM'])}%")
        else:
            print("No sentences for this dataset.")

        dataset_payload[dataset_name] = result
        total_eff_stats = merge_tan_efficiency_stats(total_eff_stats, result['efficiency'])
        total_lines += result['num_lines']
        total_src_true += result['src_true']
        total_tgt_true += result['tgt_true']
        total_src_ned_sum += result['src_ned_sum']
        total_tgt_ned_sum += result['tgt_ned_sum']

    total_src_acc = total_src_true / total_lines if total_lines else 0.0
    total_tgt_acc = total_tgt_true / total_lines if total_lines else 0.0
    total_src_ned = total_src_ned_sum / total_lines if total_lines else 0.0
    total_tgt_ned = total_tgt_ned_sum / total_lines if total_lines else 0.0

    print(f"\ntotal (TAN line text):\t src_acc: {100*total_src_acc:6g}, src_NED:{100*total_src_ned:6g}, tgt_acc: {100*total_tgt_acc:6g}, tgt_NED:{100*total_tgt_ned:6g}")

    total_eff_summary = finalize_tan_efficiency_stats(total_eff_stats) if total_eff_stats is not None else None
    if total_eff_summary is not None:
        print_tan_efficiency_summary(total_eff_summary)
        print_tan_table_summary('total', total_eff_summary)

    if save_eff_json:
        with open(save_eff_json, 'w', encoding='utf-8') as f:
            json.dump({
                'datasets': dataset_payload,
                'total': {
                    'src_acc': total_src_acc,
                    'tgt_acc': total_tgt_acc,
                    'src_ned': total_src_ned,
                    'tgt_ned': total_tgt_ned,
                    'efficiency': total_eff_summary,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Efficiency summary saved to {save_eff_json}")


if __name__ == '__main__':
    main()
