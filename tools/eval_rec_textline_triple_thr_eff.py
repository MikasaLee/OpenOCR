import copy
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import eval_rec_textline_triple_eff as eff_base
import eval_rec_textline_triple_thr as thr_base

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def aggregate_latency_results(latency_results):
    if hasattr(eff_base, 'aggregate_latency_results'):
        return eff_base.aggregate_latency_results(latency_results)

    valid = []
    for item in latency_results.values():
        latency = item.get('latency_bs1', item) if isinstance(item, dict) else item
        if latency is None:
            continue
        if latency.get('avg_ms') is None or latency.get('num_measured', 0) <= 0:
            continue
        valid.append(latency)
    if not valid:
        return None

    total_measured = sum(item['num_measured'] for item in valid)
    weighted_avg = sum(item['avg_ms'] * item['num_measured'] for item in valid) / total_measured
    return {
        'dataset_name': 'total',
        'avg_ms': float(weighted_avg),
        'p50_ms': None,
        'p90_ms': None,
        'num_measured': int(total_measured),
        'warmup': valid[0].get('warmup'),
        'fixed_first_n_samples': sum(item.get('fixed_first_n_samples', 0) for item in valid),
    }


def print_efficiency_table_summary(tag, summary, latency=None):
    if hasattr(eff_base, 'print_efficiency_table_summary'):
        return eff_base.print_efficiency_table_summary(tag, summary, latency)

    cfg_bs = summary.get('cfg_batch_size')
    throughput_key = f'throughput_bs{cfg_bs}' if cfg_bs is not None else 'throughput'
    latency_avg = 'N/A' if latency is None or latency.get('avg_ms') is None else f"{latency['avg_ms']:.2f}"
    throughput = 'N/A' if summary.get('throughput') is None else f"{summary['throughput']:.2f}"
    model_sample_time = 'N/A' if summary.get('avg_model_sample_ms') is None else f"{summary['avg_model_sample_ms']:.2f}"
    print(
        f"[EFF-SUMMARY][{tag}] "
        f"latency_bs1_model={latency_avg} ms/line | "
        f"{throughput_key}={throughput} lines/s | "
        f"model_sample_time={model_sample_time} ms/line"
    )


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--save_pred_xlsx',
        type=str,
        default=None,
        help='Path to save prediction XLSX. Default: <output_dir>/preds_dump_text_confdet_eff.xlsx',
    )
    parser.add_argument(
        '--skip_xlsx',
        type=eff_base.str2bool,
        default=False,
        help='Skip XLSX/image dump so efficiency runs are not polluted by dump artifacts.',
    )
    parser.add_argument(
        '--measure_latency_bs1',
        type=eff_base.str2bool,
        default=False,
        help='Additionally measure bs=1 model-only latency with warmup.',
    )
    parser.add_argument(
        '--latency_warmup',
        type=int,
        default=20,
        help='Warmup iterations to exclude for bs=1 model-only latency.',
    )
    parser.add_argument(
        '--latency_max_samples',
        type=int,
        default=100,
        help='Maximum measured samples for bs=1 latency. <=0 means use all remaining samples after warmup.',
    )
    parser.add_argument(
        '--save_eff_json',
        type=str,
        default=None,
        help='Optional path to save efficiency summaries as JSON.',
    )
    parser.add_argument(
        '--use_cfg_eval_dirs',
        type=eff_base.str2bool,
        default=False,
        help='Use Eval.dataset.data_dir/data_dir_list from config. Default False to match eval_rec_textline_triple_thr.py fixed two-set behavior.',
    )
    return parser.parse_args()


def build_eval_dataloader_from_dir(trainer, datadir, batch_size_override=None):
    config_each = copy.deepcopy(trainer.cfg)
    if 'RatioDataSet' in config_each['Eval']['dataset']['name']:
        config_each['Eval']['dataset']['data_dir_list'] = [datadir]
    else:
        config_each['Eval']['dataset']['data_dir'] = datadir

    if batch_size_override is not None:
        config_each['Eval']['loader']['batch_size_per_card'] = int(batch_size_override)
        config_each['Eval']['loader']['drop_last'] = False
        config_each['Eval']['loader']['shuffle'] = False

    valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
    trainer.logger.info(f'{datadir} valid dataloader has {len(valid_dataloader)} iters')
    return config_each, valid_dataloader


def dump_predictions(
    trainer,
    datadir,
    output_log,
    dataset_name,
    image_bytes,
    det_inputs_text,
    det_inputs_confdet,
    conf_thr=thr_base.CONF_THR,
    collect_artifacts=True,
):
    config_each, valid_dataloader = build_eval_dataloader_from_dir(trainer, datadir)

    model = trainer.model
    device = trainer.device
    post_process = trainer.post_process_class
    model.eval()

    num = 0
    src_true_text = 0
    tgt_true_text = 0
    src_ned_text = []
    tgt_ned_text = []

    src_true_confdet = 0
    tgt_true_confdet = 0
    src_ned_confdet = []
    tgt_ned_confdet = []

    eff_stats = eff_base.init_efficiency_stats(
        dataset_name=dataset_name,
        num_batches=len(valid_dataloader),
        cfg_batch_size=config_each['Eval']['loader'].get('batch_size_per_card', None),
    )

    with torch.no_grad():
        pbar = tqdm(total=len(valid_dataloader), desc=f'eval {dataset_name}', position=0, leave=True)
        sample_offset = 0
        prev_loop_end = time.perf_counter()
        data_iter = iter(valid_dataloader)

        for _batch_idx in range(len(valid_dataloader)):
            t_fetch_start = prev_loop_end
            batch = next(data_iter)
            t_fetch_end = time.perf_counter()

            t_prep_start = time.perf_counter()
            batch_tensor, batch_numpy, batch_extra = thr_base.split_batch(batch, device)
            raw_images = thr_base.find_raw_images(batch_extra)
            raw_labels = thr_base.find_raw_labels(batch_extra)
            model_data = batch_tensor[1:3] if len(batch_tensor) >= 3 else batch_tensor[1:]
            t_model_start = time.perf_counter()

            eff_base.sync_device(device)
            preds = model(batch_tensor[0], data=model_data)
            eff_base.sync_device(device)
            t_model_end = time.perf_counter()

            post_result = post_process(preds, batch_numpy)
            texts = post_result[0] if isinstance(post_result, tuple) else post_result

            try:
                pred_texts, pred_char_confs = thr_base.decode_text_with_char_conf(preds, post_process)
            except Exception as exc:
                trainer.logger.warning(f'Char-confidence decode failed on {dataset_name}: {exc}')
                pred_texts = []
                pred_char_confs = []

            text_list = eff_base.norm_pairs(texts)
            sample_records = []

            for i in range(len(text_list)):
                pred_text_post, _ = text_list[i]
                pred_text = pred_texts[i] if i < len(pred_texts) and pred_texts[i] != '' else pred_text_post
                char_conf = pred_char_confs[i] if i < len(pred_char_confs) else []
                pred_confdet = thr_base.build_confdet_text(pred_text, char_conf, conf_thr)

                gt_text = raw_labels[i] if raw_labels is not None and i < len(raw_labels) else ''
                if isinstance(gt_text, bytes):
                    gt_text = gt_text.decode('utf-8', errors='ignore')
                gt_src_txt, gt_tgt_txt = thr_base.split_src_tgt(gt_text)

                pred_text_norm = thr_base.replace_punctuation(pred_text)
                pred_confdet_norm = thr_base.replace_punctuation(pred_confdet)
                gt_src_norm = thr_base.replace_punctuation(gt_src_txt)
                gt_tgt_norm = thr_base.replace_punctuation(gt_tgt_txt)

                s_ned_text = 1 - thr_base.Levenshtein.normalized_distance(pred_text_norm, gt_src_norm) if gt_src_norm is not None else 0.0
                t_ned_text = 1 - thr_base.Levenshtein.normalized_distance(pred_text_norm, gt_tgt_norm) if gt_tgt_norm is not None else 0.0
                src_ned_text.append(s_ned_text)
                tgt_ned_text.append(t_ned_text)
                if int(s_ned_text) == 1:
                    src_true_text += 1
                if int(t_ned_text) == 1:
                    tgt_true_text += 1

                s_ned_confdet = 1 - thr_base.Levenshtein.normalized_distance(pred_confdet_norm, gt_src_norm) if gt_src_norm is not None else 0.0
                t_ned_confdet = 1 - thr_base.Levenshtein.normalized_distance(pred_confdet_norm, gt_tgt_norm) if gt_tgt_norm is not None else 0.0
                src_ned_confdet.append(s_ned_confdet)
                tgt_ned_confdet.append(t_ned_confdet)
                if int(s_ned_confdet) == 1:
                    src_true_confdet += 1
                if int(t_ned_confdet) == 1:
                    tgt_true_confdet += 1

                num += 1

                det_inputs_text['gts'].append(gt_src_norm)
                det_inputs_text['preds'].append(pred_text_norm)
                det_inputs_confdet['gts'].append(gt_src_norm)
                det_inputs_confdet['preds'].append(pred_confdet_norm)

                sample_records.append({
                    'img_name': f"{dataset_name}_{sample_offset + i}",
                    'type': dataset_name,
                    'label_src': gt_src_norm,
                    'label_tgt': gt_tgt_norm,
                    'pred_text': pred_text_norm,
                    'pred_confdet': pred_confdet_norm,
                    'char_conf': ' '.join(f'{c:.4f}' for c in char_conf),
                    'NED_text_src': float(s_ned_text),
                    'NED_text_tgt': float(t_ned_text),
                    'NED_confdet_src': float(s_ned_confdet),
                    'NED_confdet_tgt': float(t_ned_confdet),
                    'sample_img': raw_images[i] if raw_images is not None and i < len(raw_images) else None,
                })

            t_post_end = time.perf_counter()

            if collect_artifacts:
                for record in sample_records:
                    output_log['img_name'].append(record['img_name'])
                    output_log['type'].append(record['type'])
                    output_log['label_src'].append(record['label_src'])
                    output_log['label_tgt'].append(record['label_tgt'])
                    output_log['pred_text'].append(record['pred_text'])
                    output_log['pred_confdet'].append(record['pred_confdet'])
                    output_log['char_conf'].append(record['char_conf'])
                    output_log['NED_text_src'].append(record['NED_text_src'])
                    output_log['NED_text_tgt'].append(record['NED_text_tgt'])
                    output_log['NED_confdet_src'].append(record['NED_confdet_src'])
                    output_log['NED_confdet_tgt'].append(record['NED_confdet_tgt'])
                    try:
                        image_bytes.append(thr_base.to_png_bytes(record['sample_img']))
                    except Exception:
                        image_bytes.append(None)

            t_dump_end = time.perf_counter()

            batch_data_time = (t_fetch_end - t_fetch_start) + (t_model_start - t_prep_start)
            batch_model_time = t_model_end - t_model_start
            batch_post_time = t_post_end - t_model_end
            batch_dump_time = t_dump_end - t_post_end

            eff_stats['num_samples'] += len(sample_records)
            eff_stats['total_data'] += batch_data_time
            eff_stats['total_model'] += batch_model_time
            eff_stats['total_post'] += batch_post_time
            eff_stats['total_dump'] += batch_dump_time

            sample_offset += len(text_list)
            pbar.update(1)
            prev_loop_end = t_dump_end
        pbar.close()

    model.train()

    return (
        src_true_text / num if num else 0.0,
        tgt_true_text / num if num else 0.0,
        float(np.mean(src_ned_text)) if src_ned_text else 0.0,
        float(np.mean(tgt_ned_text)) if tgt_ned_text else 0.0,
        src_true_confdet / num if num else 0.0,
        tgt_true_confdet / num if num else 0.0,
        float(np.mean(src_ned_confdet)) if src_ned_confdet else 0.0,
        float(np.mean(tgt_ned_confdet)) if tgt_ned_confdet else 0.0,
        num,
        eff_base.finalize_efficiency_stats(eff_stats),
    )


def measure_latency_bs1(trainer, datadir, dataset_name, warmup=20, max_samples=100):
    _config_each, valid_dataloader = build_eval_dataloader_from_dir(
        trainer,
        datadir,
        batch_size_override=1,
    )

    model = trainer.model
    device = trainer.device
    model.eval()

    latencies_ms = []
    with torch.no_grad():
        pbar = tqdm(total=len(valid_dataloader), desc=f'latency {dataset_name}', position=0, leave=True)
        for batch_idx, batch in enumerate(valid_dataloader):
            batch_tensor, _batch_numpy, _batch_extra = thr_base.split_batch(batch, device)
            model_data = batch_tensor[1:3] if len(batch_tensor) >= 3 else batch_tensor[1:]

            eff_base.sync_device(device)
            t0 = time.perf_counter()
            _ = model(batch_tensor[0], data=model_data)
            eff_base.sync_device(device)
            t1 = time.perf_counter()

            if batch_idx >= warmup:
                latencies_ms.append((t1 - t0) * 1000.0)
                if max_samples > 0 and len(latencies_ms) >= max_samples:
                    pbar.update(1)
                    break
            pbar.update(1)
        pbar.close()

    model.train()

    if not latencies_ms:
        return {
            'dataset_name': dataset_name,
            'avg_ms': None,
            'p50_ms': None,
            'p90_ms': None,
            'num_measured': 0,
            'warmup': warmup,
            'fixed_first_n_samples': max(warmup, 0),
        }

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    if max_samples > 0:
        fixed_first_n_samples = warmup + min(max_samples, int(lat_arr.size))
    else:
        fixed_first_n_samples = warmup + int(lat_arr.size)
    return {
        'dataset_name': dataset_name,
        'avg_ms': float(lat_arr.mean()),
        'p50_ms': float(np.percentile(lat_arr, 50)),
        'p90_ms': float(np.percentile(lat_arr, 90)),
        'num_measured': int(lat_arr.size),
        'warmup': warmup,
        'fixed_first_n_samples': int(fixed_first_n_samples),
    }


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)

    flags = vars(FLAGS)
    opt = flags.pop('opt')
    save_pred_xlsx = flags.pop('save_pred_xlsx')
    skip_xlsx = flags.pop('skip_xlsx')
    measure_latency_flag = flags.pop('measure_latency_bs1')
    latency_warmup = flags.pop('latency_warmup')
    latency_max_samples = flags.pop('latency_max_samples')
    save_eff_json = flags.pop('save_eff_json')
    use_cfg_eval_dirs = flags.pop('use_cfg_eval_dirs')

    cfg.merge_dict(flags)
    cfg.merge_dict(opt)
    cfg = thr_base.prepare_cfg(cfg)

    if save_pred_xlsx is None:
        save_pred_xlsx = os.path.join(
            cfg.cfg['Global']['output_dir'],
            'preds_dump_text_confdet_eff.xlsx',
        )
    if (not skip_xlsx) and save_pred_xlsx:
        save_pred_dir = os.path.dirname(save_pred_xlsx)
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
    if save_eff_json:
        save_eff_dir = os.path.dirname(save_eff_json)
        if save_eff_dir:
            os.makedirs(save_eff_dir, exist_ok=True)

    trainer = Trainer(cfg, mode='eval')
    trainer.logger.info(f'Confidence threshold fixed to {thr_base.CONF_THR} (Visual-C3 OCR heuristic)')
    if trainer.cfg.get('PostProcess', {}).get('name') == 'GTCLabelDecode':
        raise NotImplementedError(
            'eval_rec_textline_triple_thr_eff.py only supports single-branch postprocess, '
            'e.g. CTCLabelDecode/ARLabelDecode.'
        )

    if use_cfg_eval_dirs:
        data_dirs_list = []
        if cfg.cfg['Eval']['dataset'].get('data_dir_list', None):
            data_dirs_list = [cfg.cfg['Eval']['dataset']['data_dir_list']]
        else:
            data_dir_single = cfg.cfg['Eval']['dataset'].get('data_dir', None)
            if data_dir_single:
                data_dirs_list = [[data_dir_single]]
        if not data_dirs_list:
            raise ValueError('No eval data dirs found in config while --use_cfg_eval_dirs=True.')
    else:
        data_dirs_list = [[
            r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_correct',
            r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_fakedv2',
        ]]

    output_log = OrderedDict([
        ('img_name', []),
        ('type', []),
        ('label_src', []),
        ('label_tgt', []),
        ('pred_text', []),
        ('pred_confdet', []),
        ('char_conf', []),
        ('NED_text_src', []),
        ('NED_text_tgt', []),
        ('NED_confdet_src', []),
        ('NED_confdet_tgt', []),
    ])
    det_inputs_text = {'gts': [], 'preds': []}
    det_inputs_confdet = {'gts': [], 'preds': []}
    image_bytes = []

    total_num = 0
    total_src_text_true = 0
    total_tgt_text_true = 0
    total_src_text_neds = []
    total_tgt_text_neds = []
    total_src_confdet_true = 0
    total_tgt_confdet_true = 0
    total_src_confdet_neds = []
    total_tgt_confdet_neds = []

    total_eff_stats = None
    dataset_eff_payload = {}

    for data_dirs in data_dirs_list:
        for datadir in data_dirs:
            dataset_name = datadir[:-1].split('/')[-1] if datadir.endswith('/') else datadir.split('/')[-1]
            (
                src_pnacc_text, tgt_pnacc_text, src_ned_text_mean, tgt_ned_text_mean,
                src_pnacc_confdet, tgt_pnacc_confdet, src_ned_confdet_mean, tgt_ned_confdet_mean,
                num, eff_summary,
            ) = dump_predictions(
                trainer,
                datadir,
                output_log,
                dataset_name,
                image_bytes,
                det_inputs_text,
                det_inputs_confdet,
                conf_thr=thr_base.CONF_THR,
                collect_artifacts=(not skip_xlsx),
            )

            print(f"{dataset_name}:\t text_src_acc: {100*src_pnacc_text:6g}, text_src_NED:{100*src_ned_text_mean:6g}, text_tgt_acc: {100*tgt_pnacc_text:6g}, text_tgt_NED:{100*tgt_ned_text_mean:6g}")
            print(f"{dataset_name}:\t confdet_src_acc: {100*src_pnacc_confdet:6g}, confdet_src_NED:{100*src_ned_confdet_mean:6g}, confdet_tgt_acc: {100*tgt_pnacc_confdet:6g}, confdet_tgt_NED:{100*tgt_ned_confdet_mean:6g}")

            latency_result = None
            if measure_latency_flag:
                latency_result = measure_latency_bs1(
                    trainer,
                    datadir,
                    dataset_name,
                    warmup=latency_warmup,
                    max_samples=latency_max_samples,
                )

            eff_base.print_efficiency_summary(eff_summary, latency_result)
            print_efficiency_table_summary(dataset_name, eff_summary, latency_result)
            dataset_eff_payload[dataset_name] = {
                'efficiency': eff_summary,
                'latency_bs1': latency_result,
            }
            total_eff_stats = eff_base.merge_efficiency_stats(total_eff_stats, eff_summary)

            total_num += num
            total_src_text_true += int(src_pnacc_text * num)
            total_tgt_text_true += int(tgt_pnacc_text * num)
            total_src_text_neds.extend([src_ned_text_mean] * num)
            total_tgt_text_neds.extend([tgt_ned_text_mean] * num)
            total_src_confdet_true += int(src_pnacc_confdet * num)
            total_tgt_confdet_true += int(tgt_pnacc_confdet * num)
            total_src_confdet_neds.extend([src_ned_confdet_mean] * num)
            total_tgt_confdet_neds.extend([tgt_ned_confdet_mean] * num)

    total_src_text_acc = (total_src_text_true / total_num) if total_num else 0.0
    total_tgt_text_acc = (total_tgt_text_true / total_num) if total_num else 0.0
    total_src_text_ned = float(np.mean(total_src_text_neds)) if total_src_text_neds else 0.0
    total_tgt_text_ned = float(np.mean(total_tgt_text_neds)) if total_tgt_text_neds else 0.0

    total_src_confdet_acc = (total_src_confdet_true / total_num) if total_num else 0.0
    total_tgt_confdet_acc = (total_tgt_confdet_true / total_num) if total_num else 0.0
    total_src_confdet_ned = float(np.mean(total_src_confdet_neds)) if total_src_confdet_neds else 0.0
    total_tgt_confdet_ned = float(np.mean(total_tgt_confdet_neds)) if total_tgt_confdet_neds else 0.0

    print(f"total (text):\t src_acc: {100*total_src_text_acc:6g}, src_NED:{100*total_src_text_ned:6g}, tgt_acc: {100*total_tgt_text_acc:6g}, tgt_NED:{100*total_tgt_text_ned:6g}")
    print(f"total (confdet):\t src_acc: {100*total_src_confdet_acc:6g}, src_NED:{100*total_src_confdet_ned:6g}, tgt_acc: {100*total_tgt_confdet_acc:6g}, tgt_NED:{100*total_tgt_confdet_ned:6g}")

    det_text = thr_base.calculate_cuo_metric_compact(det_inputs_text['gts'], det_inputs_text['preds'], X='X')
    det_confdet = thr_base.calculate_cuo_metric_compact(det_inputs_confdet['gts'], det_inputs_confdet['preds'], X='X')

    print("\nCuo detection metrics (text):")
    if det_text:
        print(f"N_sent={det_text['N_sent']} | clean={det_text['N_clean_sent']} | error={det_text['N_error_sent']}")
        print(f"Char_P={thr_base.fmt(det_text['Char_P'])}%  Char_R={thr_base.fmt(det_text['Char_R'])}%  Char_F1={thr_base.fmt(det_text['Char_F1'])}%")
        print(f"Char_BalAcc={thr_base.fmt(det_text['Char_BalAcc'])}%  Char_MCC={thr_base.fmt(det_text['Char_MCC'])}")
        print(f"Sent_FA={thr_base.fmt(det_text['Sent_FA'])}%  Sent_EM={thr_base.fmt(det_text['Sent_EM'])}%")
    else:
        print("No sentences for text branch.")

    print("\nCuo detection metrics (confdet):")
    if det_confdet:
        print(f"N_sent={det_confdet['N_sent']} | clean={det_confdet['N_clean_sent']} | error={det_confdet['N_error_sent']}")
        print(f"Char_P={thr_base.fmt(det_confdet['Char_P'])}%  Char_R={thr_base.fmt(det_confdet['Char_R'])}%  Char_F1={thr_base.fmt(det_confdet['Char_F1'])}%")
        print(f"Char_BalAcc={thr_base.fmt(det_confdet['Char_BalAcc'])}%  Char_MCC={thr_base.fmt(det_confdet['Char_MCC'])}")
        print(f"Sent_FA={thr_base.fmt(det_confdet['Sent_FA'])}%  Sent_EM={thr_base.fmt(det_confdet['Sent_EM'])}%")
    else:
        print("No sentences for confdet branch.")

    total_eff_summary = eff_base.finalize_efficiency_stats(total_eff_stats) if total_eff_stats is not None else None
    total_latency_summary = aggregate_latency_results(dataset_eff_payload)
    if total_eff_summary is not None:
        eff_base.print_efficiency_summary(total_eff_summary, total_latency_summary)
        print_efficiency_table_summary('total', total_eff_summary, total_latency_summary)

    if save_eff_json:
        with open(save_eff_json, 'w', encoding='utf-8') as f:
            json.dump({
                'datasets': dataset_eff_payload,
                'total': total_eff_summary,
                'total_latency_bs1': total_latency_summary,
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Efficiency summary saved to {save_eff_json}")

    if skip_xlsx:
        print("[INFO] XLSX dump skipped (--skip_xlsx=True).")
    else:
        try:
            eff_base.maybe_save_xlsx(save_pred_xlsx, output_log, image_bytes)
        except Exception as e:
            print(f"[WARN] Failed to save XLSX ({e}). Install pandas & openpyxl to enable XLSX export.")


if __name__ == '__main__':
    main()
