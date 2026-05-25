import copy
import csv
import inspect
import json
import math
import os
import re
import sys
import types
from contextlib import ExitStack, contextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


BENCHMARK_DIRS = {
    'bnu_zh': [
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/common_0',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/LongText_1',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/UltraLongText_2',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/illegibleText_3',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/wrongText_4',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/heavilyStrickenthrough_5',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/lightlyStrickenthrough_6',
        r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/textInsertion_7',
        r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/textInversion_8',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/textReplacement_9',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/tailSupply_10',
        # r'/a800data1/lirunrui/origin_datasets/bnu_zh_benchmark_lmdb/complexText_11',
    ],
    'bnu_en': [
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/common_0',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/LongText_1',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/UltraLongText_2',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/illegibleText_3',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/unspacedText_4',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/heavilyStrickenthrough_5',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/lightlyStrickenthrough_6',
        r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/textInsertion_7',
        r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/textInversion_8',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/textReplacement_9',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/tailSupply_10',
        # r'/a800data1/lirunrui/origin_datasets/bnu_en_benchmark_lmdb/complexText_11',
    ],
    'bctr': [
        r'/a800data1/lirunrui/origin_datasets/bchw_dataset/scene/scene_test',
        r'/a800data1/lirunrui/origin_datasets/bchw_dataset/web/web_test',
        r'/a800data1/lirunrui/origin_datasets/bchw_dataset/document/document_test',
        r'/a800data1/lirunrui/origin_datasets/bchw_dataset/hw/hw_test',
    ],
}


BENCHMARK_WEIGHTS = {
    'bnu_zh': np.array([
        0.496, 0.176, 0.021, 0.068, 0.008, 0.133,
        0.042, 0.011, 0.002, 0.026, 0.004, 0.013,
    ], dtype=np.float32),
    'bnu_en': np.array([
        0.259, 0.159, 0.227, 0.029, 0.021, 0.133,
        0.085, 0.017, 0.0, 0.03, 0.007, 0.033,
    ], dtype=np.float32),
    'bctr': np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}


ATTENTION_DECODER_NAMES = {
    'SSMDecoder',
    'PPHDecoder',
    'SARDecoder',
    'NRTRDecoder',
    'NRTRRTDDecoder',
    'PARSeqDecoder',
    'DptrParseq',
    'SMTRDecoder',
    'SMTRDecoderNumAttn',
}


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for per-character attention visualizations.')
    parser.add_argument('--benchmark', type=str, default='config',
                        choices=['config', 'bnu_zh', 'bnu_en', 'bctr'],
                        help='Use a built-in benchmark data_dir list or the config Eval dir.')
    parser.add_argument('--eval_data_dirs', nargs='*', default=None,
                        help='Optional explicit Eval data dirs. Overrides --benchmark.')
    parser.add_argument('--max_samples_per_dataset', type=int, default=32,
                        help='Maximum samples visualized for each dataset. 0 means all samples.')
    parser.add_argument('--max_batches_per_dataset', type=int, default=0,
                        help='Maximum batches evaluated for each dataset. 0 means all batches.')
    parser.add_argument('--max_chars_per_sample', type=int, default=64,
                        help='Maximum predicted characters visualized per branch/sample. 0 means all characters.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Eval batch size. Keep 1 for decoders that cache only sample-0 attention.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Eval dataloader workers.')
    parser.add_argument('--max_data_tensors', type=int, default=5,
                        help='Number of leading tensor fields passed to model.')
    parser.add_argument('--branches', nargs='*', default=None,
                        help='Optional branch name filter, e.g. eduocr_ssm svtrv2_sar.')
    parser.add_argument('--alpha', type=float, default=0.45,
                        help='Heatmap overlay alpha.')
    parser.add_argument('--cmap', type=str, default='jet',
                        choices=['jet', 'turbo', 'hot', 'viridis'],
                        help='OpenCV colormap for attention overlays.')
    parser.add_argument('--save_grid', action='store_true',
                        help='Also save a summary grid for each sample/branch.')
    parser.add_argument('--grid_cols', type=int, default=6,
                        help='Maximum columns in optional per-sample grids.')
    parser.add_argument('--image_channel_order', type=str, default='rgb',
                        choices=['rgb', 'bgr'],
                        help='Channel order of the normalized input tensor before visualization.')
    parser.add_argument('--save_npy', action='store_true',
                        help='Also save raw per-character attention maps as .npy.')
    return parser.parse_args()


def prepare_cfg(cfg: Config, flags: dict) -> Config:
    out_dir = cfg.cfg['Global'].get('output_dir', 'output')
    if out_dir.endswith('/'):
        out_dir = out_dir[:-1]
        cfg.cfg['Global']['output_dir'] = out_dir
    if cfg.cfg['Global'].get('pretrained_model') is None:
        cfg.cfg['Global']['pretrained_model'] = os.path.join(out_dir, 'best.pth')
    cfg.cfg['Global']['use_amp'] = False

    eval_cfg = cfg.cfg.get('Eval', {})
    loader_cfg = eval_cfg.get('loader', {})
    sampler_cfg = eval_cfg.get('sampler', {})
    if flags.get('batch_size') is not None:
        batch_size = int(flags['batch_size'])
        loader_cfg['batch_size_per_card'] = batch_size
        if sampler_cfg:
            sampler_cfg['first_bs'] = batch_size
            sampler_cfg['max_bs'] = batch_size
            sampler_cfg['fix_bs'] = True
    if flags.get('num_workers') is not None:
        loader_cfg['num_workers'] = int(flags['num_workers'])
    loader_cfg['shuffle'] = False
    loader_cfg['drop_last'] = False
    if sampler_cfg:
        sampler_cfg['is_training'] = False

    decoder_cfg = cfg.cfg.get('Architecture', {}).get('Decoder', {})
    post_name = cfg.cfg.get('PostProcess', {}).get('name')
    if decoder_cfg.get('name') in ('GTCDecoder', 'GTCDecoderTwo'):
        if post_name == 'GTCLabelDecode':
            # The GTC branch must run at eval time so its character attention exists.
            decoder_cfg['infer_gtc'] = True
        else:
            decoder_cfg['infer_gtc'] = False
    if post_name == 'GTCLabelDecode':
        cfg.cfg['PostProcess']['only_gtc'] = False
    return cfg


def dataset_name(path: str) -> str:
    clean = str(path).replace('\\', '/').rstrip('/')
    return clean.split('/')[-1] if clean else 'dataset'


def safe_name(text: str, limit: int = 96) -> str:
    text = '' if text is None else str(text)
    text = re.sub(r'[\\/:*?"<>|\s]+', '_', text)
    text = text.strip('_')
    return (text[:limit] or 'sample')


def char_name(ch: str) -> str:
    return safe_name(ch, limit=24)


def replace_punctuation(text: str) -> str:
    if text is None:
        return ''
    mapping = {
        '\uff0c': ',',
        '\u3002': '.',
        '\uff01': '!',
        '\uff1f': '?',
        '\uff1b': ';',
        '\uff1a': ':',
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
    }
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    return text


def new_metric_stats() -> dict:
    return {
        'num': 0,
        'true_num': 0,
        'ned_sum': 0.0,
    }


def normalized_edit_similarity(pred: str, gt: str) -> float:
    if gt is None:
        return 0.0
    return 1.0 - Levenshtein.normalized_distance(pred, gt)


def update_metric_stats(stats: dict, pred: str, gt: str):
    ned = normalized_edit_similarity(pred, gt)
    stats['num'] += 1
    stats['ned_sum'] += float(ned)
    if int(ned) == 1:
        stats['true_num'] += 1
    return ned


def finalize_metric_stats(stats: dict) -> dict:
    num = int(stats.get('num', 0))
    true_num = int(stats.get('true_num', 0))
    ned_sum = float(stats.get('ned_sum', 0.0))
    return {
        'num': num,
        'true_num': true_num,
        'acc': true_num / num if num else 0.0,
        'norm_edit_dis': ned_sum / num if num else 0.0,
    }


def print_metric_line(name: str, metrics: dict, branch: Optional[str] = None):
    prefix = f'{name} [{branch}]' if branch else name
    print(f"{prefix}:\t\t acc: {100 * metrics['acc']:6g}, "
          f"norm_edit_dis:{100 * metrics['norm_edit_dis']:6g}")


def resolve_eval_data_dirs(cfg: dict, flags: dict) -> List[str]:
    if flags.get('eval_data_dirs'):
        return list(flags['eval_data_dirs'])
    benchmark = flags.get('benchmark', 'config')
    if benchmark != 'config':
        return list(BENCHMARK_DIRS[benchmark])
    dataset_cfg = cfg['Eval']['dataset']
    if dataset_cfg.get('data_dir_list'):
        return list(dataset_cfg['data_dir_list'])
    if dataset_cfg.get('data_dir'):
        return [dataset_cfg['data_dir']]
    raise ValueError('No Eval data_dir/data_dir_list found. Use --eval_data_dirs.')


def set_eval_dir(config: dict, datadir: str) -> dict:
    config_each = copy.deepcopy(config)
    dataset_cfg = config_each['Eval']['dataset']
    if 'RatioDataSet' in dataset_cfg.get('name', ''):
        dataset_cfg['data_dir_list'] = [datadir]
    else:
        dataset_cfg['data_dir'] = datadir
        dataset_cfg.pop('data_dir_list', None)
    return config_each


def split_batch(batch, device, max_data_tensors: int):
    tensors = []
    arrays = []
    for item in batch:
        if not torch.is_tensor(item):
            break
        tensors.append(item.to(device))
        arrays.append(item.detach().cpu().numpy())
        if len(tensors) >= max_data_tensors:
            break
    return tensors, arrays


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def get_core_decoders(model: torch.nn.Module):
    core = unwrap_model(model)
    decoder = getattr(core, 'decoder', None)
    gtc_decoder = getattr(decoder, 'gtc_decoder', None)
    ctc_decoder = getattr(decoder, 'ctc_decoder', None)
    if gtc_decoder is None and decoder is not None:
        if decoder.__class__.__name__ in ATTENTION_DECODER_NAMES:
            gtc_decoder = decoder
    return decoder, gtc_decoder, ctc_decoder


def find_modules(root, class_names):
    out = []
    if root is None:
        return out
    for module in root.modules():
        if module.__class__.__name__ in class_names:
            out.append(module)
    return out


@contextmanager
def capture_decoder_hw(*modules):
    handles = []

    def pre_hook(module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x) and x.dim() == 4:
            module._charvis_last_hw = (int(x.shape[2]), int(x.shape[3]))

    try:
        for module in modules:
            if module is not None:
                handles.append(module.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_ssm_cross_attn(ssm_decoder):
    if ssm_decoder is None or not hasattr(ssm_decoder, 'transformer_dec'):
        yield
        return

    backups = []
    try:
        for layer in getattr(ssm_decoder.transformer_dec, 'layers', []):
            mha = getattr(layer, 'multihead_attn', None)
            if mha is None:
                continue
            orig_forward = mha.forward
            sig = inspect.signature(orig_forward)
            supports_avg = 'average_attn_weights' in sig.parameters

            def wrapped_forward(*args, _orig=orig_forward, _mha=mha,
                                _supports_avg=supports_avg, **kwargs):
                kwargs['need_weights'] = True
                if _supports_avg:
                    kwargs['average_attn_weights'] = False
                out = _orig(*args, **kwargs)
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    _mha._charvis_last_attn = out[1].detach()
                return out

            mha.forward = wrapped_forward
            backups.append((mha, orig_forward))
        yield
    finally:
        for mha, orig_forward in backups:
            mha.forward = orig_forward


@contextmanager
def capture_pph_attention(pph_decoder):
    if pph_decoder is None or pph_decoder.__class__.__name__ != 'PPHDecoder':
        yield
        return

    orig_forward = pph_decoder.forward

    def wrapped_forward(self, x, data=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        kv = self.fc_kv(x_flat).reshape(B, H * W, 2, self.num_heads,
                                        self.head_dim)
        kv = kv.permute(2, 0, 3, 4, 1)
        x_k, x_v = kv[0], kv[1]

        base_token = self.char_token + self.permutation_bias
        base_token = base_token.expand(B, -1, -1)

        query_list = [base_token]
        for _ in range(self.perturb_samples - 1):
            noise = torch.randn_like(base_token) * self.perturb_std
            query_list.append(base_token + noise)

        attn_scores_all = []
        for q in query_list:
            q = q.unsqueeze(2)
            scores = torch.matmul(q, x_k)
            attn_scores_all.append(scores)
        attn_scores = torch.stack(attn_scores_all, dim=0).mean(0)

        attn_2d = attn_scores.view(B * self.num_heads, 1, H, W)
        attn_2d = F.softmax(attn_2d, dim=2)
        attn_2d = attn_2d.permute(0, 3, 1, 2)
        self._charvis_last_hw = (int(H), int(W))
        self._charvis_last_attn = attn_2d.detach().view(B, self.num_heads, W, H)

        x_v_4d = x_v.reshape(B * self.num_heads, self.head_dim, H, W)
        x_v_for_attn = x_v_4d.permute(0, 3, 2, 1)
        head_feats = (attn_2d @ x_v_for_attn).squeeze(2)

        head_feats = head_feats.view(B, self.num_heads, W, self.head_dim)
        feats = head_feats.permute(0, 2, 1, 3).reshape(B, W, C)

        logits = self.fc(feats)

        if self.return_feats and self.training:
            return feats, logits
        if not self.training:
            return F.softmax(logits, dim=2)
        return logits

    try:
        pph_decoder.forward = types.MethodType(wrapped_forward, pph_decoder)
        yield
    finally:
        pph_decoder.forward = orig_forward


@contextmanager
def capture_sar_attention(sar_decoder):
    if sar_decoder is None or sar_decoder.__class__.__name__ != 'SARDecoder':
        yield
        return

    orig_attn = sar_decoder._2d_attation
    orig_forward = sar_decoder.forward

    def wrapped_attn(self, feat, tokens, data, training):
        hidden_state = self.rnndecoder(tokens)[0]
        attn_query = self.conv1x1_1(hidden_state)
        bsz, seq_len, _ = attn_query.size()
        attn_query = attn_query.unsqueeze(-1).unsqueeze(-1)
        attn_key = self.conv3x3_1(feat).unsqueeze(1)

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        attn_weight = self.conv1x1_2(attn_weight)

        _, T, h, w, c = attn_weight.size()

        if self.mask:
            valid_ratios = data[-1]
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()
        self._charvis_last_hw = (int(h), int(w))
        self._charvis_last_attn = attn_weight.detach()

        attn_feat = torch.sum(torch.mul(feat.unsqueeze(1), attn_weight),
                              (3, 4),
                              keepdim=False)
        return [hidden_state, attn_feat]

    def wrapped_forward(self, feat, data=None):
        if self.use_lstm:
            holistic_feat = self.encoder(feat)
        else:
            holistic_feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

        if self.training:
            return self.forward_train(feat, holistic_feat, data=data)
        return self.forward_test(feat, holistic_feat, data=data)

    try:
        sar_decoder._2d_attation = types.MethodType(wrapped_attn, sar_decoder)
        sar_decoder.forward = types.MethodType(wrapped_forward, sar_decoder)
        yield
    finally:
        sar_decoder._2d_attation = orig_attn
        sar_decoder.forward = orig_forward


def get_last_ssm_attn(ssm_decoder) -> Optional[torch.Tensor]:
    if ssm_decoder is None or not hasattr(ssm_decoder, 'transformer_dec'):
        return None
    layers = getattr(ssm_decoder.transformer_dec, 'layers', [])
    if not layers:
        return None
    mha = getattr(layers[-1], 'multihead_attn', None)
    return getattr(mha, '_charvis_last_attn', None) if mha is not None else None


def reduce_cross_attn(attn: Optional[torch.Tensor], bsz: int) -> Optional[np.ndarray]:
    if attn is None:
        return None
    if attn.dim() == 4:
        attn = attn.mean(dim=1)  # [B, T, S]
    elif attn.dim() != 3:
        return None
    if attn.shape[0] != bsz and attn.shape[1] == bsz:
        attn = attn.transpose(0, 1)
    return attn.detach().float().cpu().numpy()


def norm_pairs(items):
    out = []
    for item in items or []:
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            text = item[0]
            conf = item[1] if len(item) > 1 else 1.0
            out.append((str(text), float(conf)))
        else:
            out.append((str(item), 1.0))
    return out


def parse_post_result(post_result):
    branches = {}
    gts = None
    if isinstance(post_result, list) and len(post_result) == 2:
        if isinstance(post_result[0], tuple):
            branches['gtc'] = norm_pairs(post_result[0][0])
            gts = post_result[0][1]
        else:
            branches['gtc'] = norm_pairs(post_result[0])
        if isinstance(post_result[1], tuple):
            branches['ctc'] = norm_pairs(post_result[1][0])
            if gts is None:
                gts = post_result[1][1]
        else:
            branches['ctc'] = norm_pairs(post_result[1])
    elif isinstance(post_result, tuple):
        branches['pred'] = norm_pairs(post_result[0])
        gts = post_result[1]
    else:
        branches['pred'] = norm_pairs(post_result)
    return branches, gts


def decode_predictions(post_process, preds, batch_numpy):
    if isinstance(preds, dict):
        if hasattr(post_process, 'gtc_label_decode'):
            return parse_post_result(post_process(preds, batch_numpy))
        if 'ctc_pred' in preds:
            branches, gts = parse_post_result(post_process(preds['ctc_pred'], batch_numpy))
            if 'pred' in branches:
                branches = {'ctc': branches['pred']}
            return branches, gts
        first_value = next(iter(preds.values()))
        return parse_post_result(post_process(first_value, batch_numpy))
    return parse_post_result(post_process(preds, batch_numpy))


def gt_text_at(gts, batch_numpy, idx: int) -> str:
    if gts is not None and idx < len(gts):
        item = gts[idx]
        if isinstance(item, (tuple, list)) and item:
            return str(item[0])
        return str(item)
    if len(batch_numpy) > 1 and idx < len(batch_numpy[1]):
        item = batch_numpy[1][idx]
        if isinstance(item, bytes):
            return item.decode('utf-8', errors='ignore')
        return str(item)
    return ''


def get_ctc_characters(post_process) -> List[str]:
    if hasattr(post_process, 'ctc_label_decode'):
        return list(post_process.ctc_label_decode.character)
    if hasattr(post_process, 'character'):
        return list(post_process.character)
    return []


def get_hw(*modules) -> Tuple[Optional[int], Optional[int]]:
    for module in modules:
        hw = getattr(module, '_charvis_last_hw', None) if module is not None else None
        if hw is not None:
            return int(hw[0]), int(hw[1])
    return None, None


def normalize_prob(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.maximum(vec, 0.0)
    total = float(vec.sum())
    if total <= 0:
        return np.full_like(vec, 1.0 / max(1, vec.size), dtype=np.float64)
    return vec / total


def attention_vector_to_map(vec: np.ndarray,
                            h: Optional[int],
                            w: Optional[int]) -> np.ndarray:
    vec = normalize_prob(vec)
    size = int(vec.size)
    if h is not None and w is not None and h * w == size:
        return vec.reshape(h, w).astype(np.float32)
    if w is not None and w == size:
        return vec.reshape(1, w).astype(np.float32)
    if h is not None and h > 0 and size % h == 0:
        return vec.reshape(h, size // h).astype(np.float32)
    side = int(round(math.sqrt(size)))
    if side * side == size:
        return vec.reshape(side, side).astype(np.float32)
    return vec.reshape(1, size).astype(np.float32)


def select_step_attention(attn, bidx: int = 0) -> np.ndarray:
    arr = attn.detach().float().cpu().numpy() if torch.is_tensor(attn) else np.asarray(attn)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 4:
        # [B, heads, T, S]
        b = min(bidx, arr.shape[0] - 1)
        return arr[b].mean(axis=0)[-1]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0, -1]
        if arr.shape[1] == 1:
            return arr.mean(axis=0)[0]
        return arr.mean(axis=0)[-1]
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)
    return arr.reshape(-1)


def ctc_emit_positions(probs: np.ndarray, characters: List[str], blank_id: int = 0):
    idx = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    out = []
    prev = None
    for t, cls_id in enumerate(idx.tolist()):
        if cls_id == prev:
            prev = cls_id
            continue
        prev = cls_id
        if int(cls_id) == blank_id:
            continue
        if cls_id < 0 or cls_id >= len(characters):
            continue
        out.append({
            'frame_index': int(t),
            'class_id': int(cls_id),
            'pred_char': str(characters[int(cls_id)]),
            'confidence': float(conf[t]),
        })
    return out


def pph_attention_maps(pph_decoder,
                       ctc_probs_b: Optional[np.ndarray],
                       pred: str,
                       characters: List[str],
                       bidx: int,
                       blank_id: int = 0):
    if ctc_probs_b is None:
        return []
    emits = ctc_emit_positions(ctc_probs_b, characters, blank_id=blank_id)
    if pred and len(emits) > len(pred):
        emits = emits[:len(pred)]
    pph_attn = getattr(pph_decoder, '_charvis_last_attn', None)
    if pph_attn is not None:
        pph_attn = pph_attn.detach().float().cpu().numpy()
    h, w = get_hw(pph_decoder)
    maps = []
    for pos, emit in enumerate(emits[:len(pred)]):
        if pph_attn is None or bidx >= pph_attn.shape[0]:
            frame_w = w or max(1, int(ctc_probs_b.shape[0]))
            frame_h = h or 1
            frame = min(int(emit['frame_index']), frame_w - 1)
            amap = np.zeros((frame_h, frame_w), dtype=np.float32)
            amap[:, frame] = 1.0 / max(1, frame_h)
        else:
            # Stored as [B, heads, W, H], softmax over H for each column.
            heads, attn_w, attn_h = pph_attn[bidx].shape
            frame = min(int(emit['frame_index']), attn_w - 1)
            y_prob = normalize_prob(pph_attn[bidx, :, frame, :].mean(axis=0))
            amap = np.zeros((attn_h, attn_w), dtype=np.float32)
            amap[:, frame] = y_prob.astype(np.float32)
        maps.append({
            'pos': pos,
            'char': pred[pos] if pos < len(pred) else emit['pred_char'],
            'map': amap,
            'source': 'pph_ctc',
        })
    return maps


def ssm_attention_maps(ssm_decoder, pred: str, bidx: int, top_decoder=None):
    attn = reduce_cross_attn(get_last_ssm_attn(ssm_decoder), bidx + 1)
    if attn is None or bidx >= attn.shape[0]:
        return []
    h, w = get_hw(ssm_decoder, top_decoder)
    maps = []
    max_len = min(len(pred), attn.shape[1])
    for pos in range(max_len):
        maps.append({
            'pos': pos,
            'char': pred[pos],
            'map': attention_vector_to_map(attn[bidx, pos], h, w),
            'source': 'ssm_cross_attn',
        })
    return maps


def sar_attention_maps(sar_decoder, pred: str, bidx: int):
    attn = getattr(sar_decoder, '_charvis_last_attn', None)
    if attn is None:
        return []
    arr = attn.detach().float().cpu().numpy()
    if bidx >= arr.shape[0]:
        return []
    # [B, T, 1, H, W]. T includes holistic token at index 0.
    sample = arr[bidx, :, 0]
    offset = 1 if sample.shape[0] > len(pred) else 0
    maps = []
    for pos, ch in enumerate(pred[:max(0, sample.shape[0] - offset)]):
        maps.append({
            'pos': pos,
            'char': ch,
            'map': sample[pos + offset].astype(np.float32),
            'source': 'sar_2d_attn',
        })
    return maps


def cached_step_attention_maps(module, pred: str, bidx: int, top_decoder=None):
    attn_maps = getattr(module, 'attn_maps', None)
    if not attn_maps:
        return []
    h, w = get_hw(module, top_decoder)
    out = []
    for pos, attn in enumerate(attn_maps[:len(pred)]):
        vec = select_step_attention(attn, bidx)
        out.append({
            'pos': pos,
            'char': pred[pos],
            'map': attention_vector_to_map(vec, h, w),
            'source': f'{module.__class__.__name__}.attn_maps',
        })
    return out


def attention_maps_for_branch(branch_key: str,
                              module,
                              pred: str,
                              bidx: int,
                              top_decoder,
                              preds,
                              ctc_characters: List[str]):
    if module is None:
        return []
    name = module.__class__.__name__
    if name == 'SSMDecoder':
        return ssm_attention_maps(module, pred, bidx, top_decoder=top_decoder)
    if name == 'PPHDecoder':
        ctc_probs = None
        if isinstance(preds, dict) and 'ctc_pred' in preds:
            ctc_probs = preds['ctc_pred'].detach().float().cpu().numpy()
        elif torch.is_tensor(preds):
            ctc_probs = preds.detach().float().cpu().numpy()
        if ctc_probs is None or bidx >= ctc_probs.shape[0]:
            return []
        return pph_attention_maps(module, ctc_probs[bidx], pred, ctc_characters, bidx)
    if name == 'SARDecoder':
        return sar_attention_maps(module, pred, bidx)
    return cached_step_attention_maps(module, pred, bidx, top_decoder=top_decoder)


def branch_display_name(branch_key: str, module, cfg: dict) -> str:
    decoder_name = module.__class__.__name__ if module is not None else branch_key
    if decoder_name == 'SSMDecoder':
        return 'eduocr_ssm'
    if decoder_name == 'PPHDecoder':
        return 'eduocr_pph'
    base = decoder_name.replace('Decoder', '').replace('Dptr', 'dptr').lower()
    encoder_name = cfg.get('Architecture', {}).get('Encoder', {}).get('name', '')
    prefix = 'svtrv2' if 'SVTRv2' in str(encoder_name) else 'attention'
    return f'{prefix}_{base}'


def tensor_to_image_rgb(tensor: torch.Tensor, channel_order: str = 'rgb') -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f'Expected CHW tensor, got shape {arr.shape}')
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = arr[:3]
    if float(arr.min()) < -0.05:
        arr = (arr + 1.0) * 0.5
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    if channel_order == 'bgr':
        arr = arr[:, :, ::-1]
    return (arr * 255.0).round().astype(np.uint8)


def cv_colormap(name: str):
    return {
        'jet': cv2.COLORMAP_JET,
        'turbo': getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
    }[name]


def normalize_map_for_display(attn_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(attn_map, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - float(arr.min())
    maxv = float(arr.max())
    if maxv > 0:
        arr = arr / maxv
    return arr


def overlay_heatmap(image_rgb: np.ndarray,
                    attn_map: np.ndarray,
                    alpha: float,
                    cmap: str) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    attn = normalize_map_for_display(attn_map)
    attn = cv2.resize(attn, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((attn * 255).astype(np.uint8), cv_colormap(cmap))
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_png(path: str, image_rgb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def ascii_label(text: str) -> str:
    return text.encode('ascii', errors='replace').decode('ascii')


def make_grid(images: List[np.ndarray], labels: List[str], cols: int) -> np.ndarray:
    if not images:
        raise ValueError('No images for grid.')
    cols = max(1, min(cols, len(images)))
    h, w = images[0].shape[:2]
    label_h = 24
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.full((rows * (h + label_h), cols * w, 3), 255, dtype=np.uint8)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        y = r * (h + label_h)
        x = c * w
        canvas[y:y + h, x:x + w] = img
        cv2.putText(canvas, ascii_label(label)[:32], (x + 4, y + h + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def save_attention_visuals(sample_dir: str,
                           sample_name: str,
                           branch_name: str,
                           image_rgb: np.ndarray,
                           maps: List[dict],
                           flags: dict):
    branch_dir = os.path.join(sample_dir, safe_name(branch_name))
    char_dir = os.path.join(branch_dir, 'char_attention')
    os.makedirs(branch_dir, exist_ok=True)
    os.makedirs(char_dir, exist_ok=True)
    save_png(os.path.join(branch_dir, 'input.png'), image_rgb)

    max_chars = int(flags['max_chars_per_sample'])
    limited_maps = maps if max_chars <= 0 else maps[:max_chars]
    grid_images = [image_rgb]
    grid_labels = ['input']
    char_rows = []
    for item in limited_maps:
        overlay = overlay_heatmap(image_rgb, item['map'], float(flags['alpha']), flags['cmap'])
        pos = int(item['pos'])
        ch = item.get('char', '')
        stem = f"char{pos:02d}_{char_name(ch)}"
        overlay_path = os.path.join(char_dir, stem + '_overlay.png')
        save_png(overlay_path, overlay)

        npy_path = ''
        if flags.get('save_npy'):
            npy_path = os.path.join(char_dir, stem + '_attn.npy')
            np.save(npy_path, item['map'].astype(np.float32))

        char_rows.append({
            'char_index': pos,
            'pred_char': ch,
            'image_path': overlay_path,
            'npy_path': npy_path,
            'attention_source': item.get('source', ''),
        })

        if flags.get('save_grid'):
            label = f"#{pos:02d} {ch}"
            grid_images.append(overlay)
            grid_labels.append(label)

    grid_path = ''
    if flags.get('save_grid') and char_rows:
        grid = make_grid(grid_images, grid_labels, int(flags['grid_cols']))
        grid_path = os.path.join(branch_dir, f'{safe_name(sample_name)}_{safe_name(branch_name)}_grid.png')
        save_png(grid_path, grid)

    return char_rows, grid_path


def write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def main():
    args = parse_args()
    flags = vars(args)

    cfg = Config(args.config)
    cfg.merge_dict(flags.get('opt') or {})
    cfg = prepare_cfg(cfg, flags)

    save_dir = flags.get('save_dir')
    if not save_dir:
        save_dir = os.path.join(cfg.cfg['Global']['output_dir'], 'char_attention_vis')
    os.makedirs(save_dir, exist_ok=True)

    data_dirs = resolve_eval_data_dirs(cfg.cfg, flags)

    trainer = Trainer(cfg, mode='test')
    model = trainer.model
    device = trainer.device
    post_process = trainer.post_process_class
    decoder, gtc_decoder, ctc_decoder = get_core_decoders(model)
    if decoder is None:
        raise RuntimeError('Model has no decoder.')

    core = unwrap_model(model)
    attention_modules = find_modules(core, ATTENTION_DECODER_NAMES)
    ssm_modules = [m for m in attention_modules if m.__class__.__name__ == 'SSMDecoder']
    pph_modules = [m for m in attention_modules if m.__class__.__name__ == 'PPHDecoder']
    sar_modules = [m for m in attention_modules if m.__class__.__name__ == 'SARDecoder']

    if not attention_modules:
        print('[WARN] No supported attention decoder found. Pure CTC models have no per-character attention map.')

    ctc_characters = get_ctc_characters(post_process)
    branch_filter = set(flags['branches'] or [])
    manifest_rows = []
    skipped_rows = []
    metric_rows = []
    branch_dataset_metrics: Dict[str, List[dict]] = {}
    branch_total_stats: Dict[str, dict] = {}
    total_samples = 0
    total_visualized = 0
    total_char_maps = 0

    model.eval()
    for datadir in data_dirs:
        dname = dataset_name(datadir)
        config_each = set_eval_dir(trainer.cfg, datadir)
        valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
        trainer.logger.info(f'{datadir} valid dataloader has {len(valid_dataloader)} iters')

        dataset_samples = 0
        dataset_metric_stats: Dict[str, dict] = {}
        pbar = tqdm(total=len(valid_dataloader), desc=f'char-attn {dname}', position=0, leave=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                if flags['max_batches_per_dataset'] > 0 and batch_idx >= flags['max_batches_per_dataset']:
                    break
                if flags['max_samples_per_dataset'] > 0 and dataset_samples >= flags['max_samples_per_dataset']:
                    break

                batch_tensor, batch_numpy = split_batch(batch, device, int(flags['max_data_tensors']))
                if not batch_tensor:
                    pbar.update(1)
                    continue

                with ExitStack() as stack:
                    stack.enter_context(capture_decoder_hw(decoder, gtc_decoder, ctc_decoder, *attention_modules))
                    for module in ssm_modules:
                        stack.enter_context(capture_ssm_cross_attn(module))
                    for module in pph_modules:
                        stack.enter_context(capture_pph_attention(module))
                    for module in sar_modules:
                        stack.enter_context(capture_sar_attention(module))
                    preds = model(batch_tensor[0], data=batch_tensor[1:])
                    branches, gts = decode_predictions(post_process, preds, batch_numpy)

                bsz = int(batch_tensor[0].shape[0])
                branch_modules = {}
                if gtc_decoder is not None and 'gtc' in branches:
                    branch_modules['gtc'] = gtc_decoder
                if ctc_decoder is not None and 'ctc' in branches:
                    branch_modules['ctc'] = ctc_decoder
                if 'pred' in branches:
                    if decoder.__class__.__name__ in ('GTCDecoder', 'GTCDecoderTwo') and ctc_decoder is not None:
                        branch_modules['pred'] = ctc_decoder
                    else:
                        branch_modules['pred'] = decoder

                for i in range(bsz):
                    if flags['max_samples_per_dataset'] > 0 and dataset_samples >= flags['max_samples_per_dataset']:
                        break
                    gt = replace_punctuation(gt_text_at(gts, batch_numpy, i))
                    sample_name = f'{dname}_{dataset_samples:06d}'
                    image_rgb = tensor_to_image_rgb(batch_tensor[0][i], flags['image_channel_order'])
                    sample_dir = os.path.join(save_dir, dname, safe_name(sample_name))

                    for branch_key, pairs in branches.items():
                        if i >= len(pairs):
                            continue
                        module = branch_modules.get(branch_key)
                        branch_name = branch_display_name(branch_key, module, cfg.cfg)
                        if branch_filter and branch_name not in branch_filter and branch_key not in branch_filter:
                            continue

                        pred_text, pred_conf = pairs[i]
                        pred_text = replace_punctuation(pred_text)
                        metric_stats = dataset_metric_stats.setdefault(branch_name, new_metric_stats())
                        update_metric_stats(metric_stats, pred_text, gt)
                        total_stats = branch_total_stats.setdefault(branch_name, new_metric_stats())
                        update_metric_stats(total_stats, pred_text, gt)

                        maps = attention_maps_for_branch(
                            branch_key,
                            module,
                            pred_text,
                            i,
                            decoder,
                            preds,
                            ctc_characters,
                        )

                        if not maps:
                            skipped_rows.append({
                                'sample_name': sample_name,
                                'dataset': dname,
                                'branch': branch_name,
                                'label': gt,
                                'pred': pred_text,
                                'reason': 'no_supported_attention_map',
                            })
                            continue

                        char_rows, grid_path = save_attention_visuals(
                            sample_dir,
                            sample_name,
                            branch_name,
                            image_rgb,
                            maps,
                            flags,
                        )
                        total_visualized += 1
                        total_char_maps += len(char_rows)
                        for char_row in char_rows:
                            row = {
                                'sample_name': sample_name,
                                'dataset': dname,
                                'branch': branch_name,
                                'label': gt,
                                'pred': pred_text,
                                'confidence': pred_conf,
                                'num_chars': len(pred_text),
                                'num_attention_maps': len(maps),
                                'grid_path': grid_path,
                            }
                            row.update(char_row)
                            manifest_rows.append(row)

                    dataset_samples += 1
                    total_samples += 1
                pbar.update(1)
        pbar.close()

        for branch_name, stats in dataset_metric_stats.items():
            metrics = finalize_metric_stats(stats)
            metric_row = {
                'scope': 'dataset',
                'dataset': dname,
                'branch': branch_name,
            }
            metric_row.update(metrics)
            metric_rows.append(metric_row)
            branch_dataset_metrics.setdefault(branch_name, []).append(metric_row)
            print_metric_line(dname, metrics, branch=branch_name)

    metric_summary = {}
    benchmark = flags.get('benchmark', 'config')
    weight = None if flags.get('eval_data_dirs') else BENCHMARK_WEIGHTS.get(benchmark)
    for branch_name, stats in branch_total_stats.items():
        total_metrics = finalize_metric_stats(stats)
        dataset_metrics = branch_dataset_metrics.get(branch_name, [])
        if dataset_metrics:
            s_mean_metrics = {
                'num': sum(item['num'] for item in dataset_metrics),
                'true_num': sum(item['true_num'] for item in dataset_metrics),
                'acc': float(np.mean([item['acc'] for item in dataset_metrics])),
                'norm_edit_dis': float(np.mean([item['norm_edit_dis'] for item in dataset_metrics])),
            }
        else:
            s_mean_metrics = finalize_metric_stats(new_metric_stats())

        s_weight_metrics = None
        if weight is not None:
            if len(dataset_metrics) == len(weight):
                norm_weight = weight / np.sum(weight)
                acc_arr = np.array([item['acc'] for item in dataset_metrics], dtype=np.float32)
                ned_arr = np.array([item['norm_edit_dis'] for item in dataset_metrics], dtype=np.float32)
                s_weight_metrics = {
                    'num': sum(item['num'] for item in dataset_metrics),
                    'true_num': sum(item['true_num'] for item in dataset_metrics),
                    'acc': float(np.sum(acc_arr * norm_weight)),
                    'norm_edit_dis': float(np.sum(ned_arr * norm_weight)),
                }
            else:
                print(f"[WARN] S_WEIGHT length mismatch for {branch_name}: "
                      f"metrics={len(dataset_metrics)}, weights={len(weight)}. "
                      "Fallback to S_mean.")
                s_weight_metrics = dict(s_mean_metrics)

        metric_summary[branch_name] = {
            'datasets': dataset_metrics,
            'total': total_metrics,
            'S_mean': s_mean_metrics,
        }
        if s_weight_metrics is not None:
            metric_summary[branch_name]['S_weight'] = s_weight_metrics

        print_metric_line('total', total_metrics, branch=branch_name)
        print_metric_line('S_mean', s_mean_metrics, branch=branch_name)
        if s_weight_metrics is not None:
            print_metric_line('S_weight', s_weight_metrics, branch=branch_name)

    manifest_csv = os.path.join(save_dir, 'manifest.csv')
    skipped_csv = os.path.join(save_dir, 'skipped.csv')
    if manifest_rows:
        write_csv(manifest_csv, manifest_rows)
    if skipped_rows:
        write_csv(skipped_csv, skipped_rows)

    summary = {
        'config': args.config,
        'data_dirs': data_dirs,
        'total_samples': total_samples,
        'visualized_branch_samples': total_visualized,
        'exported_char_maps': total_char_maps,
        'manifest_csv': manifest_csv if manifest_rows else '',
        'skipped_csv': skipped_csv if skipped_rows else '',
        'metrics': metric_summary,
        'save_dir': save_dir,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)

    print(f'[DONE] samples={total_samples}')
    print(f'[DONE] visualized_branch_samples={total_visualized}')
    print(f'[DONE] exported_char_maps={total_char_maps}')
    print(f'[DONE] save_dir={save_dir}')
    if skipped_rows:
        print(f'[WARN] skipped={len(skipped_rows)}; see {skipped_csv}')


if __name__ == '__main__':
    main()
