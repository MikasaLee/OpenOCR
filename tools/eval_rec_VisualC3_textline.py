import io
import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def replace_punctuation(text: str) -> str:
    """将常见中文标点替换为英文标点，保持与参考脚本一致。"""
    if text is None:
        return ''
    mapping = {
        r'，': r',',
        r'。': r'.',
        r'！': r'!',
        r'？': r'?',
        r'；': r';',
        r'：': r':',
        r'“': r'"',
        r'”': r'"',
        r'‘': r"'",
        r'’': r"'",
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def to_png_bytes(img_array):
    """将 numpy 图像数组转成 PNG bytes，兼容单通道/三通道及 0~1/0~255 输入。"""
    if img_array is None:
        return None
    arr = np.array(img_array)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.squeeze(arr)
    if arr.max() <= 1.0 + 1e-3:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    mode = 'L' if arr.ndim == 2 else 'RGB'
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()
    buf.close()
    return data


def split_src_tgt(gt_raw: str):
    """解析 src???tgt 形式标签，缺失时做容错。"""
    if gt_raw is None:
        return '', ''
    parts = str(gt_raw).split('???')
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(gt_raw), ''


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--save_pred_xlsx',
        type=str,
        default=None,
        help='Path to save prediction XLSX. Default: <output_dir>/preds_dump_textline.xlsx',
    )
    args = parser.parse_args()
    return args


def prepare_cfg(cfg):
    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 30
    cfg.cfg['Global']['max_text_length'] = 30
    cfg.cfg['Metric']['max_ratio'] = 12
    keep_keys = cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys']['keep_keys']
    if 'real_ratio' not in keep_keys:
        keep_keys.append('real_ratio')
    return cfg


def dump_predictions(trainer, datadir, output_log, dataset_name, image_bytes):
    config_each = trainer.cfg.copy()
    if 'RatioDataSet' in config_each['Eval']['dataset']['name']:
        config_each['Eval']['dataset']['data_dir_list'] = [datadir]
    else:
        config_each['Eval']['dataset']['data_dir'] = datadir
    valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
    trainer.logger.info(f'{datadir} valid dataloader has {len(valid_dataloader)} iters')

    model = trainer.model
    device = trainer.device
    post_process = trainer.post_process_class
    model.eval()
    num = 0
    src_true_num = 0
    tgt_true_num = 0
    src_ned_list = []
    tgt_ned_list = []
    with torch.no_grad():
        pbar = tqdm(total=len(valid_dataloader), desc=f'eval {dataset_name}', position=0, leave=True)
        sample_offset = 0
        for batch_idx, batch in enumerate(valid_dataloader):
            batch_tensor = [t.to(device) for t in batch[:3]]
            batch_numpy = [t.numpy() for t in batch[:3]]
            raw_images = batch[3] if len(batch) > 3 else None
            preds = model(batch_tensor[0], data=batch_tensor[1:])
            post_result = post_process(preds, batch_numpy)
            texts, gts = post_result if isinstance(post_result, tuple) else (post_result, None)

            for i, (txt, prob) in enumerate(texts):
                gt_text = ''
                if gts is not None and i < len(gts):
                    gt_text = gts[i][0]
                gt_src_txt, gt_tgt_txt = split_src_tgt(gt_text)

                txt_norm = replace_punctuation(txt)
                gt_src_norm = replace_punctuation(gt_src_txt)
                gt_tgt_norm = replace_punctuation(gt_tgt_txt)
                src_ned = 1 - Levenshtein.normalized_distance(txt_norm, gt_src_norm) if gt_src_norm is not None else 0.0
                tgt_ned = 1 - Levenshtein.normalized_distance(txt_norm, gt_tgt_norm) if gt_tgt_norm is not None else 0.0
                src_ned_list.append(src_ned)
                tgt_ned_list.append(tgt_ned)
                num += 1
                if int(src_ned) == 1:
                    src_true_num += 1
                if int(tgt_ned) == 1:
                    tgt_true_num += 1

                img_name = f"{dataset_name}_{sample_offset + i}"
                output_log['img_name'].append(img_name)
                output_log['type'].append(dataset_name)
                output_log['label_src'].append(gt_src_norm)
                output_log['label_tgt'].append(gt_tgt_norm)
                output_log['pred'].append(txt_norm)
                output_log['NED_src'].append(float(src_ned))
                output_log['NED_tgt'].append(float(tgt_ned))
                try:
                    sample_img = raw_images[i] if raw_images is not None else None
                    image_bytes.append(to_png_bytes(sample_img))
                except Exception:
                    image_bytes.append(None)
            sample_offset += len(texts)
            pbar.update(1)
        pbar.close()
    model.train()
    src_pnacc = src_true_num / num if num else 0.0
    tgt_pnacc = tgt_true_num / num if num else 0.0
    src_ned_mean = float(np.mean(src_ned_list)) if src_ned_list else 0.0
    tgt_ned_mean = float(np.mean(tgt_ned_list)) if tgt_ned_list else 0.0
    return src_pnacc, tgt_pnacc, src_ned_mean, tgt_ned_mean, num


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    cfg = prepare_cfg(cfg)

    save_pred_xlsx = FLAGS.get('save_pred_xlsx')
    if save_pred_xlsx is None:
        save_pred_xlsx = os.path.join(cfg.cfg['Global']['output_dir'], 'preds_dump_textline.xlsx')
    os.makedirs(os.path.dirname(save_pred_xlsx), exist_ok=True)

    trainer = Trainer(cfg, mode='eval')

    data_dirs_list = []
    if cfg.cfg['Eval']['dataset'].get('data_dir_list', None):
        data_dirs_list = [cfg.cfg['Eval']['dataset']['data_dir_list']]
    else:
        data_dir_single = cfg.cfg['Eval']['dataset'].get('data_dir', None)
        if data_dir_single:
            data_dirs_list = [[data_dir_single]]

    # 默认同时评测 textline test_correct 与 test_faked
    data_dirs_list = [[
        r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_correct',
        r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_faked',
    ]]

    output_log = OrderedDict([
        ('img_name', []),
        ('type', []),
        ('label_src', []),
        ('label_tgt', []),
        ('pred', []),
        ('NED_src', []),
        ('NED_tgt', []),
    ])
    image_bytes = []
    every_src_PNacc_list = []
    every_tgt_PNacc_list = []
    every_src_ned_list = []
    every_tgt_ned_list = []
    total_num = 0
    total_src_True_num = 0
    total_tgt_True_num = 0
    total_src_ned_list = []
    total_tgt_ned_list = []
    for data_dirs in data_dirs_list:
        for datadir in data_dirs:
            dataset_name = datadir[:-1].split('/')[-1] if datadir.endswith('/') else datadir.split('/')[-1]
            src_pnacc, tgt_pnacc, src_ned_mean, tgt_ned_mean, num = dump_predictions(trainer, datadir, output_log, dataset_name, image_bytes)
            print(f"{dataset_name}:\t\t src_acc: {100 * src_pnacc:6g}, src_norm_edit_dis:{100 * src_ned_mean:6g}, tgt_acc: {100 * tgt_pnacc:6g}, tgt_norm_edit_dis:{100 * tgt_ned_mean:6g}")
            every_src_PNacc_list.append(src_pnacc)
            every_tgt_PNacc_list.append(tgt_pnacc)
            every_src_ned_list.append(src_ned_mean)
            every_tgt_ned_list.append(tgt_ned_mean)
            total_num += num
            total_src_True_num += int(src_pnacc * num)
            total_src_ned_list.extend([src_ned_mean] * num)
            total_tgt_True_num += int(tgt_pnacc * num)
            total_tgt_ned_list.extend([tgt_ned_mean] * num)

    try:
        import pandas as pd
        df = pd.DataFrame(output_log)
        df.to_excel(save_pred_xlsx, index=False)
        try:
            from openpyxl import load_workbook
            from openpyxl.drawing.image import Image as OpenpyxlImage
            from openpyxl.utils import get_column_letter

            wb = load_workbook(save_pred_xlsx)
            ws = wb.active
            img_col = ws.max_column + 1
            ws.cell(row=1, column=img_col, value='image')
            img_col_letter = get_column_letter(img_col)
            target_px = 160
            ws.column_dimensions[img_col_letter].width = max(ws.column_dimensions[img_col_letter].width or 0, target_px / 7.0)
            embedded = 0
            for r_idx, data in enumerate(image_bytes, start=2):
                if not data:
                    continue
                img_obj = OpenpyxlImage(io.BytesIO(data))
                try:
                    with Image.open(io.BytesIO(data)) as pil_img:
                        w, h = pil_img.size
                except Exception:
                    w, h = None, None
                img_obj.width = target_px
                if w and h and w > 0:
                    img_obj.height = h * (target_px / float(w))
                ws.row_dimensions[r_idx].height = max(ws.row_dimensions[r_idx].height or 0, img_obj.height * 0.75)
                img_obj.anchor = f"{img_col_letter}{r_idx}"
                ws.add_image(img_obj)
                embedded += 1
            print(f"[INFO] Embedded {embedded} images into Excel")
            wb.save(save_pred_xlsx)
        except Exception as embed_err:
            print(f"[WARN] Failed to embed images into XLSX ({embed_err}). Ensure openpyxl is installed.")

        total_src_acc = (total_src_True_num / total_num) if total_num else 0.0
        total_src_ned = float(np.mean(total_src_ned_list)) if total_src_ned_list else 0.0
        s_mean_src_acc = float(np.mean(every_src_PNacc_list)) if every_src_PNacc_list else 0.0
        s_mean_src_ned = float(np.mean(every_src_ned_list)) if every_src_ned_list else 0.0
        total_tgt_acc = (total_tgt_True_num / total_num) if total_num else 0.0
        total_tgt_ned = float(np.mean(total_tgt_ned_list)) if total_tgt_ned_list else 0.0
        s_mean_tgt_acc = float(np.mean(every_tgt_PNacc_list)) if every_tgt_PNacc_list else 0.0
        s_mean_tgt_ned = float(np.mean(every_tgt_ned_list)) if every_tgt_ned_list else 0.0
        s_weight_src_acc = float(np.sum(np.array(every_src_PNacc_list))) if every_src_PNacc_list else 0.0
        s_weight_src_ned = float(np.sum(np.array(every_src_ned_list))) if every_src_ned_list else 0.0
        s_weight_tgt_acc = float(np.sum(np.array(every_tgt_PNacc_list))) if every_tgt_PNacc_list else 0.0
        s_weight_tgt_ned = float(np.sum(np.array(every_tgt_ned_list))) if every_tgt_ned_list else 0.0

        print(f"total:\t\t src_acc: {100 * total_src_acc:6g}, src_norm_edit_dis:{100 * total_src_ned:6g},tgt_acc: {100 * total_tgt_acc:6g}, tgt_norm_edit_dis:{100 * total_tgt_ned:6g}")
        print(f"S_mean:\t\t src_acc: {100 * s_mean_src_acc:6g}, src_norm_edit_dis:{100 * s_mean_src_ned:6g}, tgt_acc: {100 * s_mean_tgt_acc:6g}, tgt_norm_edit_dis:{100 * s_mean_tgt_ned:6g}")
        print(f"S_weight:\t\t src_acc: {100 * s_weight_src_acc:6g}, src_norm_edit_dis:{100 * s_weight_src_ned:6g}, tgt_acc: {100 * s_weight_tgt_acc:6g}, tgt_norm_edit_dis:{100 * s_weight_tgt_ned:6g}")
        print(f'Predictions (with NED) saved to {save_pred_xlsx}')
    except Exception as e:
        print(f'[WARN] Failed to save XLSX ({e}). Install pandas & openpyxl to enable XLSX export.')


if __name__ == '__main__':
    main()
