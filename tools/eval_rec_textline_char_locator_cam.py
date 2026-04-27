import gc
import io
import os
import re
import sys
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser
from tools.utils.ids_to_tree import ids_to_tree_supervision
from tools.utils.ids_syntax import validate_ids_prefix, DEFAULT_IDC_ARITY


def parse_args():
    parser = ArgsParser()
    parser.add_argument('--save_dir', type=str, default=None,
                        help='输出目录，默认 <Global.output_dir>/char_locator_cam')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='最多导出多少个样本（跨数据集累计），<=0 表示全部')
    parser.add_argument('--max_chars_per_sample', type=int, default=12,
                        help='每个样本最多导出多少个字符位热力图，<=0 表示该样本全部字符')
    parser.add_argument('--max_parts_per_char', type=int, default=0,
                        help='每个字符最多导出多少个 IDS 节点热力图；=0 表示该字符的全部节点')
    parser.add_argument('--save_only_cross_match', action='store_true',
                        help='只保存满足 pred_text==label_tgt 且 pred_idsText==label_src 的样本')
    parser.add_argument('--occlusion_chunk_size', type=int, default=32,
                        help='部件 occlusion 的分块大小，越大越快但更占显存')
    parser.add_argument('--target_score', type=str, default='mean_top_class',
                        choices=['mean_top_class', 'max_logit'],
                        help='Grad-CAM 反传打分方式')
    parser.add_argument('--save_npy', action='store_true',
                        help='是否额外保存 CAM 的 npy')
    parser.add_argument('--flatten_order', type=str, default='wh',
                        choices=['hw', 'wh'],
                        help='注意力向量反展平顺序: hw=idx->(y,x), wh=idx->(x,y)')
    return parser.parse_args()


def replace_punctuation(text: str) -> str:
    if text is None:
        return ''
    mapping = {
        '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
        '“': '"', '”': '"', '‘': "'", '’': "'",
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def split_src_tgt(gt_raw: str):
    if gt_raw is None:
        return '', ''
    parts = str(gt_raw).split('<unk>', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(gt_raw), ''


def _char_at(text: str, idx: int) -> str:
    if text is None:
        return ''
    if idx < 0 or idx >= len(text):
        return ''
    return text[idx]


def norm_pairs(lst):
    out = []
    for item in lst:
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            s = item[0]
            p = item[1] if len(item) > 1 else 1.0
            out.append((str(s), float(p)))
        else:
            out.append((str(item), 1.0))
    return out


def parse_post_result(post_result, post_process):
    gts = None
    outputs = None
    if isinstance(post_result, dict):
        outputs = post_result
        if 'label_text' in post_result:
            gts = post_result['label_text']
    elif isinstance(post_result, list) and len(post_result) == 2 and isinstance(post_result[0], tuple):
        (text_res, text_gt), (ids_res, ids_gt) = post_result
        ids_text_res = []
        for item in ids_res:
            ids_str = item[0] if isinstance(item, (tuple, list)) else item
            conf = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else 1.0
            if hasattr(post_process, 'map_ids_to_text'):
                ids_text_res.append((post_process.map_ids_to_text(ids_str), conf))
            else:
                ids_text_res.append(('', conf))
        outputs = {'text': text_res, 'ids': ids_res, 'text_from_ids': ids_text_res}
        gts = text_gt
    elif isinstance(post_result, tuple):
        outputs, gts = post_result
    else:
        outputs = {'text': post_result}

    text_list = norm_pairs(outputs.get('text', []))
    ids_list = norm_pairs(outputs.get('ids', []))
    ids_text_list = norm_pairs(outputs.get('text_from_ids', []))
    return text_list, ids_list, ids_text_list, gts


def prepare_cfg(cfg):
    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'

    cfg.cfg['Metric']['max_len'] = 30
    cfg.cfg['Metric']['max_len_ids'] = 200
    cfg.cfg['Metric']['max_single_char_ids_len'] = 50
    cfg.cfg['Global']['max_text_length'] = 30
    cfg.cfg['Global']['max_ids_length'] = 200
    cfg.cfg['Global']['max_single_char_ids_len'] = 50

    cfg.cfg['Global']['use_amp'] = False

    eval_transforms = cfg.cfg['Eval']['dataset']['transforms']

    if not any('SaveRawImageBytes' in t for t in eval_transforms):
        insert_idx = 0
        for i, t in enumerate(eval_transforms):
            if 'DecodeImagePIL' in t:
                insert_idx = i + 1
                break
        eval_transforms.insert(insert_idx, {'SaveRawImageBytes': {'dst_key': 'image_raw', 'src_key': 'image'}})

    if 'loader' not in cfg.cfg['Eval']:
        cfg.cfg['Eval']['loader'] = {}
    cfg.cfg['Eval']['loader']['collate_fn'] = 'RecWithRawCollator'

    keep_keys_list = None
    for t in eval_transforms:
        if 'KeepKeys' in t:
            keep_keys_list = t['KeepKeys']['keep_keys']
            break

    if keep_keys_list is not None:
        if 'real_ratio' not in keep_keys_list:
            keep_keys_list.append('real_ratio')
        if 'image_raw' not in keep_keys_list:
            keep_keys_list.append('image_raw')

    return cfg


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def find_raw_images_from_batch(batch) -> Optional[List[np.ndarray]]:
    if len(batch) <= 6:
        return None
    for item in batch[6:]:
        if isinstance(item, list) and len(item) > 0:
            first_sample = item[0]
            if isinstance(first_sample, np.ndarray) and first_sample.ndim >= 2:
                return item
            if isinstance(first_sample, bytes):
                out = []
                for b in item:
                    try:
                        with Image.open(io.BytesIO(b)) as im:
                            out.append(np.array(im.convert('RGB')))
                    except Exception:
                        out.append(None)
                return out
    return None


def find_real_ratios_from_batch(batch, batch_size: int) -> Optional[List[float]]:
    if len(batch) <= 6:
        return None
    for item in batch[6:]:
        vals = None
        if isinstance(item, torch.Tensor) and item.ndim == 1 and item.numel() == batch_size:
            vals = item.detach().cpu().numpy().tolist()
        elif isinstance(item, list) and len(item) == batch_size:
            first = item[0]
            if isinstance(first, (float, int, np.floating, np.integer)):
                vals = [float(v) for v in item]
        if vals is None:
            continue
        if all(np.isfinite(v) for v in vals) and all(v > 0 for v in vals):
            if all(v <= 2.0 for v in vals):
                return [float(v) for v in vals]
    return None


def tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.squeeze(arr)
    arr = (arr * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def normalize_map(cam: torch.Tensor) -> np.ndarray:
    c = cam.detach().float().cpu().numpy()
    c = np.maximum(c, 0)
    mx = float(c.max())
    mn = float(c.min())
    if mx > mn:
        c = (c - mn) / (mx - mn)
    else:
        c = np.zeros_like(c, dtype=np.float32)
    return c.astype(np.float32)


def colorize_cam(cam01: np.ndarray) -> np.ndarray:
    heat = np.zeros((cam01.shape[0], cam01.shape[1], 3), dtype=np.float32)
    heat[..., 0] = cam01
    heat[..., 1] = np.clip(2.0 * cam01 - 0.2, 0.0, 1.0) * 0.7
    heat[..., 2] = np.clip(1.0 - cam01, 0.0, 1.0) * 0.3
    return (heat * 255.0).astype(np.uint8)


def resize_cam_to_image(cam01: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    cam_img = Image.fromarray((cam01 * 255.0).astype(np.uint8), mode='L')
    cam_img = cam_img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(cam_img).astype(np.float32) / 255.0


def overlay_cam_on_image(image_u8: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_cam(cam01).astype(np.float32)
    base = image_u8.astype(np.float32)
    out = np.clip((1.0 - alpha) * base + alpha * color, 0, 255).astype(np.uint8)
    return out


def safe_name(text: str, limit: int = 48) -> str:
    text = text if text is not None else ''
    text = re.sub(r'[\\/:*?"<>|\s]+', '_', str(text)).strip('_')
    if not text:
        text = 'empty'
    return text[:limit]


def get_score_from_ctc_logits(logits_1char: torch.Tensor, mode: str) -> Tuple[torch.Tensor, int]:
    if mode == 'max_logit':
        v, idx = logits_1char.max(dim=-1)
        score = v.max()
        cls_id = int(idx[v.argmax()].item())
        return score, cls_id

    mean_logits = logits_1char.mean(dim=0)
    cls_id = int(mean_logits.argmax().item())
    score = logits_1char[:, cls_id].mean()
    return score, cls_id


def extract_ctc_token_spans(logits_1char: torch.Tensor, blank_id: int = 0) -> List[Tuple[int, List[int]]]:
    pred_ids = logits_1char.argmax(dim=-1).tolist()
    spans = []
    prev = None
    current_token = None
    current_frames: List[int] = []

    for frame_idx, token_id in enumerate(pred_ids):
        if token_id == blank_id:
            if current_token is not None and current_frames:
                spans.append((current_token, current_frames))
                current_token = None
                current_frames = []
            prev = token_id
            continue

        if token_id == prev and current_token is not None:
            current_frames.append(frame_idx)
            prev = token_id
            continue

        if current_token is not None and current_frames:
            spans.append((current_token, current_frames))

        current_token = token_id
        current_frames = [frame_idx]
        prev = token_id

    if current_token is not None and current_frames:
        spans.append((current_token, current_frames))

    return spans


def extract_ctc_component_spans(logits_1char: torch.Tensor, decoder, leaf_only: bool = True):
    spans = extract_ctc_token_spans(
        logits_1char,
        blank_id=int(getattr(decoder, 'ctc_blank_id', 0)),
    )

    special_ids = {
        int(getattr(decoder, 'ctc_blank_id', 0)),
        int(getattr(decoder, 'bos_id', 1)),
        int(getattr(decoder, 'eos_id', 2)),
        int(getattr(decoder, 'unk_id', 3)),
    }
    leaf_ids = getattr(decoder, '_ids_leaf_ids', set())
    ids_tokens = getattr(decoder, 'ids_tokens', None)

    out = []
    for token_id, frame_indices in spans:
        if token_id in special_ids:
            continue
        if leaf_only and len(leaf_ids) > 0 and token_id not in leaf_ids:
            continue

        token_name = str(token_id)
        if ids_tokens is not None and 0 <= token_id < len(ids_tokens):
            token_name = ids_tokens[token_id]

        out.append((int(token_id), token_name, frame_indices))
    return out


def get_valid_map_width(real_ratios: Optional[List[float]], sample_idx: int, map_w: int) -> int:
    if real_ratios is None or sample_idx >= len(real_ratios):
        return map_w
    ratio = float(real_ratios[sample_idx])
    valid_w = int(round(map_w * ratio))
    return max(1, min(map_w, valid_w))


def _parse_ids_tree(ids_str: str):
    if not ids_str:
        return None
    try:
        sup = ids_to_tree_supervision(ids_str, add_sos=False, add_eos=False)
    except Exception:
        return None

    if not sup.tokens:
        return None

    children = [[] for _ in sup.tokens]
    root = None
    for idx, parent in enumerate(sup.parents):
        if parent < 0:
            root = idx
        else:
            children[parent].append(idx)
    if root is None:
        return None
    return {
        'tokens': sup.tokens,
        'children': children,
        'root': root,
    }


def build_ids_node_targets(ids_str: str, token_spans_all):
    parsed = _parse_ids_tree(ids_str)
    if parsed is None or len(parsed['tokens']) != len(token_spans_all):
        nodes = []
        for span_idx, (token_id, token_name, frame_indices) in enumerate(token_spans_all):
            if not frame_indices:
                continue
            nodes.append({
                'node_index': span_idx,
                'node_kind': 'leaf',
                'node_token': token_name,
                'node_label': token_name,
                'score_refs': [{'token_id': int(token_id), 'frames': list(frame_indices)}],
            })
        return sorted(nodes, key=lambda x: x['node_index'])

    tokens = parsed['tokens']
    children = parsed['children']
    root = parsed['root']
    nodes = []

    def _dfs(node_idx: int):
        child_ids = children[node_idx]
        node_entry = {
            'node_index': int(node_idx),
            'node_kind': 'composite' if len(child_ids) > 0 else 'leaf',
            'node_token': str(tokens[node_idx]),
        }
        nodes.append(node_entry)

        subtree_tokens = [str(tokens[node_idx])]
        leaf_span_indices = []
        if len(child_ids) == 0:
            leaf_span_indices.append(node_idx)
        else:
            for child_idx in child_ids:
                child_leaf_spans, child_tokens = _dfs(child_idx)
                leaf_span_indices.extend(child_leaf_spans)
                subtree_tokens.extend(child_tokens)

        node_entry['node_label'] = ''.join(subtree_tokens)
        node_entry['score_refs'] = []
        for span_idx in leaf_span_indices:
            token_id, token_name, frame_indices = token_spans_all[span_idx]
            if not frame_indices:
                continue
            node_entry['score_refs'].append({
                'token_id': int(token_id),
                'token_name': str(token_name),
                'frames': list(frame_indices),
            })
        return leaf_span_indices, subtree_tokens

    _dfs(root)
    nodes = [node for node in nodes if len(node.get('score_refs', [])) > 0]
    return sorted(nodes, key=lambda x: x['node_index'])


def compute_ids_node_scores_batch(logits_batch: torch.Tensor, node_targets) -> torch.Tensor:
    batch_size = logits_batch.size(0)
    scores = []
    for node in node_targets:
        refs = node.get('score_refs', [])
        ref_scores = []
        for ref in refs:
            frame_indices = ref.get('frames', [])
            if not frame_indices:
                continue
            token_id = int(ref['token_id'])
            frame_idx = torch.as_tensor(frame_indices, dtype=torch.long, device=logits_batch.device)
            ref_scores.append(logits_batch.index_select(1, frame_idx)[:, :, token_id].mean(dim=1))
        if len(ref_scores) == 0:
            scores.append(logits_batch.new_zeros((batch_size,)))
        else:
            scores.append(torch.stack(ref_scores, dim=1).mean(dim=1))
    if len(scores) == 0:
        return logits_batch.new_zeros((batch_size, 0))
    return torch.stack(scores, dim=1)


def is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cuda error: out of memory' in msg


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def build_occlusion_node_maps_for_char(
    decoder,
    memory_1sample: torch.Tensor,
    mem_mask_1sample: Optional[torch.Tensor],
    q_idx: int,
    max_chars: int,
    logits_1char: torch.Tensor,
    ids_str: str,
    token_spans_all,
    map_h: int,
    map_w: int,
    flatten_order: str = 'hw',
    occlusion_chunk_size: int = 32,
):
    node_targets = build_ids_node_targets(ids_str, token_spans_all)
    if len(node_targets) == 0:
        return []

    baseline_scores = compute_ids_node_scores_batch(logits_1char.unsqueeze(0), node_targets)[0]

    valid_mask = None
    if mem_mask_1sample is not None:
        valid_mask = ~mem_mask_1sample[0]
    if valid_mask is None:
        valid_mask = torch.ones((memory_1sample.size(1),), dtype=torch.bool, device=memory_1sample.device)
    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return []

    if bool(valid_mask.any()):
        fill_vec = memory_1sample[0][valid_mask].mean(dim=0)
    else:
        fill_vec = memory_1sample.new_zeros((memory_1sample.size(-1),))

    drop_maps = logits_1char.new_zeros((len(node_targets), memory_1sample.size(1)))
    chunk_size = max(1, int(occlusion_chunk_size))
    start = 0
    while start < valid_indices.numel():
        curr_chunk_size = min(chunk_size, int(valid_indices.numel() - start))
        flat_indices = valid_indices[start:start + curr_chunk_size]
        chunk = int(flat_indices.numel())
        row_indices = torch.arange(chunk, device=memory_1sample.device)
        memory_occ = None
        mem_mask_occ = None
        char_feat_all_occ = None
        char_feat_occ = None
        style_line_occ = None
        style_code_occ = None
        ctc_logits_occ = None
        occ_scores = None
        score_drops = None

        try:
            memory_occ = memory_1sample.expand(chunk, -1, -1).clone()
            memory_occ[row_indices, flat_indices, :] = fill_vec

            mem_mask_occ = None
            if mem_mask_1sample is not None:
                mem_mask_occ = mem_mask_1sample.expand(chunk, -1)

            char_feat_all_occ, _ = decoder._extract_char_features(
                memory_occ,
                mem_mask_occ,
                max_chars,
                lengths=None,
            )
            char_feat_occ = char_feat_all_occ[:, q_idx, 0, :]

            style_code_occ = None
            if getattr(decoder, 'use_line_style_film', False):
                style_line_occ = decoder._extract_line_style_code(None, memory_occ, mem_mask_occ)
                style_code_occ = decoder._build_char_aware_style_code(style_line_occ, char_feat_occ)

            ctc_logits_occ, _ = decoder._ctc_forward(
                char_feat_occ,
                style_code=style_code_occ,
                return_struct=False,
            )
            occ_scores = compute_ids_node_scores_batch(ctc_logits_occ, node_targets)
            score_drops = (baseline_scores.unsqueeze(0) - occ_scores).clamp_min_(0.0)
            drop_maps.index_copy_(1, flat_indices, score_drops.transpose(0, 1))
            start += curr_chunk_size
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            del row_indices, flat_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if curr_chunk_size <= 1:
                raise RuntimeError(
                    'CUDA OOM even with occlusion_chunk_size=1; reduce input size or exported samples.'
                ) from exc
            chunk_size = max(1, curr_chunk_size // 2)
            continue
        finally:
            memory_occ = None
            mem_mask_occ = None
            char_feat_all_occ = None
            char_feat_occ = None
            style_line_occ = None
            style_code_occ = None
            ctc_logits_occ = None
            occ_scores = None
            score_drops = None

    outputs = []
    for node_idx, node in enumerate(node_targets):
        cam_small = unflatten_attn(
            drop_maps[node_idx],
            h=map_h,
            w=map_w,
            flatten_order=flatten_order,
        )
        cam_small = normalize_map(cam_small)
        outputs.append({
            'node_index': int(node['node_index']),
            'node_kind': str(node['node_kind']),
            'node_token': str(node['node_token']),
            'node_label': str(node['node_label']),
            'cam_small': cam_small,
        })
    return outputs


def ensure_2d_feature(memory: torch.Tensor, h: int, feat2d: Optional[torch.Tensor]) -> torch.Tensor:
    if feat2d is not None:
        return feat2d
    b, t, c = memory.shape
    if h <= 0 or t % h != 0:
        h = 1
    w = t // h
    return memory.view(b, h, w, c)


def unflatten_attn(attn_vec: torch.Tensor, h: int, w: int, flatten_order: str = 'hw') -> torch.Tensor:
    """
    将长度 T=h*w 的注意力向量按指定顺序还原为 [h,w]。
    - hw: idx = y*w + x
    - wh: idx = x*h + y
    """
    t = h * w
    if attn_vec.numel() != t:
        raise ValueError(f'attn length mismatch: {attn_vec.numel()} vs {t}')

    out = attn_vec.new_zeros((h, w))
    idx = torch.arange(t, device=attn_vec.device)
    if flatten_order == 'wh':
        x = idx // h
        y = idx % h
    else:
        y = idx // w
        x = idx % w
    out[y, x] = attn_vec
    return out


@contextmanager
def capture_char_locator_cross_attn(decoder):
    backups = []
    try:
        for layer in decoder.char_locator.layers:
            mha = layer.multihead_attn
            orig_forward = mha.forward

            def wrapped_forward(*args, _orig=orig_forward, _mha=mha, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                out = _orig(*args, **kwargs)
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    _mha._last_attn_weights = out[1].detach()
                return out

            mha.forward = wrapped_forward
            backups.append((mha, orig_forward))
        yield
    finally:
        for mha, orig_forward in backups:
            mha.forward = orig_forward


def dump_char_locator_cam(trainer, valid_dataloader, save_dir: str, dataset_name: str, max_samples: int,
                          max_chars_per_sample: int, max_parts_per_char: int, target_score: str,
                          save_npy: bool, flatten_order: str = 'hw',
                          occlusion_chunk_size: int = 32,
                          save_only_cross_match: bool = False):
    model = trainer.model
    device = trainer.device
    post_process = trainer.post_process_class
    model.eval()

    core = unwrap_model(model)
    if not hasattr(core, 'decoder'):
        raise RuntimeError('模型不包含 decoder，无法提取 char_locator CAM')

    decoder = core.decoder
    if not hasattr(decoder, 'char_locator') or decoder.char_locator is None:
        raise RuntimeError('当前 decoder 没有启用 IDS/char_locator 分支')

    os.makedirs(save_dir, exist_ok=True)
    xlsx_path = os.path.join(save_dir, 'cam_manifest.xlsx')

    exported_samples = 0
    exported_char_maps = 0
    exported_part_maps = 0
    manifest_rows = []

    pbar = tqdm(total=len(valid_dataloader), desc='export char_locator cam', leave=True)
    for batch_idx, batch in enumerate(valid_dataloader):
        if max_samples > 0 and exported_samples >= max_samples:
            break

        batch_tensor = [t.to(device) for t in batch[:6]]
        batch_numpy = [t.detach().cpu().numpy() for t in batch[:6]]
        images = batch_tensor[0]
        real_ratios = find_real_ratios_from_batch(batch, images.size(0))

        with torch.no_grad():
            preds = model(batch_tensor[0], data=batch_tensor[1:])
            post_result = post_process(preds, batch_numpy)
        text_list, ids_list, ids_text_list, gts = parse_post_result(post_result, post_process)

        sample_meta = []
        for i in range(images.size(0)):
            pred_text = text_list[i][0] if i < len(text_list) else ''
            pred_ids_text = ids_text_list[i][0] if i < len(ids_text_list) else ''
            pred_ids = ids_list[i][0] if i < len(ids_list) else ''

            gt_text = ''
            if gts is not None and i < len(gts):
                gt_item = gts[i]
                gt_text = gt_item[0] if isinstance(gt_item, (tuple, list)) and len(gt_item) > 0 else str(gt_item)
            elif i < len(batch_numpy[1]):
                try:
                    gt_text = batch_numpy[1][i].decode('utf-8') if isinstance(batch_numpy[1][i], bytes) else str(batch_numpy[1][i])
                except Exception:
                    gt_text = str(batch_numpy[1][i])

            gt_src, gt_tgt = split_src_tgt(gt_text)
            pred_text_norm = replace_punctuation(pred_text)
            pred_ids_text_norm = replace_punctuation(pred_ids_text)
            gt_src_norm = replace_punctuation(gt_src)
            gt_tgt_norm = replace_punctuation(gt_tgt)

            try:
                ned_text_src = 1 - Levenshtein.normalized_distance(pred_text_norm, gt_src_norm)
            except Exception:
                ned_text_src = float('nan')
            try:
                ned_text_tgt = 1 - Levenshtein.normalized_distance(pred_text_norm, gt_tgt_norm)
            except Exception:
                ned_text_tgt = float('nan')
            try:
                ned_ids_src = 1 - Levenshtein.normalized_distance(pred_ids_text_norm, gt_src_norm)
            except Exception:
                ned_ids_src = float('nan')
            try:
                ned_ids_tgt = 1 - Levenshtein.normalized_distance(pred_ids_text_norm, gt_tgt_norm)
            except Exception:
                ned_ids_tgt = float('nan')

            tokens = [tok for tok in str(pred_ids).strip().split() if tok]
            if tokens:
                token_validity = []
                for tok in tokens:
                    ok, _, _ = validate_ids_prefix([ch for ch in tok if not ch.isspace()], idc_arity=DEFAULT_IDC_ARITY, require_closed=True)
                    token_validity.append(ok)
                ids_is_valid = bool(all(token_validity))
            else:
                ids_is_valid = False

            sample_meta.append({
                'label_src': gt_src_norm,
                'label_tgt': gt_tgt_norm,
                'pred_text': pred_text_norm,
                'pred_idsText': pred_ids_text_norm,
                'pred_ids_line': str(pred_ids),
                'ids_valid_line': ids_is_valid,
                'ned_text_src': ned_text_src,
                'ned_text_tgt': ned_text_tgt,
                'ned_ids_src': ned_ids_src,
                'ned_ids_tgt': ned_ids_tgt,
                'exact_text_tgt': bool(pred_text_norm == gt_tgt_norm),
                'exact_ids_src': bool(pred_ids_text_norm == gt_src_norm),
                'cross_match': bool(pred_text_norm == gt_tgt_norm and pred_ids_text_norm == gt_src_norm),
            })

        x = images
        if getattr(core, 'use_transform', False):
            x = core.transform(x)
        if getattr(core, 'use_encoder', False):
            x = core.encoder(x)

        with torch.no_grad(), capture_char_locator_cross_attn(decoder):
            memory, h, mem_mask, feat2d = decoder._prep_memory(x)
            feat2d = ensure_2d_feature(memory, h, feat2d)

            max_chars = decoder.max_text_len
            char_feat_all, valid_logits = decoder._extract_char_features(memory, mem_mask, max_chars, lengths=None)

            attn_weights = None
            if len(decoder.char_locator.layers) > 0:
                attn_weights = getattr(decoder.char_locator.layers[-1].multihead_attn, '_last_attn_weights', None)
            char_feat_2d = char_feat_all[:, :, 0, :]

            bsz = memory.size(0)
            if decoder.use_valid_head:
                valid_len_pred, valid_mask = decoder._prefix_lengths_from_valid_logits(valid_logits)
            else:
                valid_len_pred = torch.full((bsz,), max_chars, dtype=torch.long, device=memory.device)
                valid_mask = torch.ones((bsz, max_chars), dtype=torch.bool, device=memory.device)

            valid_len_pred = torch.clamp(valid_len_pred, min=0, max=max_chars)

            char_feat_valid = char_feat_2d[valid_mask]
            style_line = None
            style_valid = None
            if char_feat_valid.numel() == 0:
                preds = None
                post_result = None
                batch_tensor = None
                images = None
                x = None
                memory = None
                feat2d = None
                char_feat_all = None
                valid_logits = None
                attn_weights = None
                char_feat_2d = None
                valid_len_pred = None
                valid_mask = None
                char_feat_valid = None
                style_code_valid = None
                style_line = None
                style_valid = None
                ctc_logits_valid = None
                ids_decoded_valid = None
                valid_indices = None
                release_cuda_memory()
                pbar.update(1)
                continue

            style_code_valid = None
            if getattr(decoder, 'use_line_style_film', False):
                style_line = decoder._extract_line_style_code(feat2d, memory, mem_mask)
                style_valid = decoder._expand_style_to_valid_chars(style_line, valid_len_pred, max_chars)
                style_code_valid = decoder._build_char_aware_style_code(style_valid, char_feat_valid)

            ctc_logits_valid, _ = decoder._ctc_forward(char_feat_valid, style_code=style_code_valid, return_struct=False)

            if getattr(decoder, 'constrained_ctc_decode', False):
                ids_decoded_valid = decoder._ctc_constrained_decode(ctc_logits_valid)
            else:
                ids_decoded_valid = decoder._ctc_greedy_decode(ctc_logits_valid)

            valid_indices = valid_mask.nonzero(as_tuple=False)

            for b in range(bsz):
                if max_samples > 0 and exported_samples >= max_samples:
                    break

                char_cap = max_chars if max_chars_per_sample <= 0 else max_chars_per_sample
                n_chars = int(min(valid_len_pred[b].item(), max_chars, char_cap))
                if n_chars <= 0:
                    continue

                line_meta = sample_meta[b] if b < len(sample_meta) else {
                    'label_src': '',
                    'label_tgt': '',
                    'pred_text': '',
                    'pred_idsText': '',
                    'pred_ids_line': '',
                    'ids_valid_line': False,
                    'ned_text_src': float('nan'),
                    'ned_text_tgt': float('nan'),
                    'ned_ids_src': float('nan'),
                    'ned_ids_tgt': float('nan'),
                    'exact_text_tgt': False,
                    'exact_ids_src': False,
                    'cross_match': False,
                }

                if save_only_cross_match and not bool(line_meta.get('cross_match', False)):
                    continue

                sample_dir = os.path.join(save_dir, f'batch{batch_idx:04d}_sample{b:02d}')
                char_dir = os.path.join(sample_dir, 'char_attention')
                part_root_dir = os.path.join(sample_dir, 'part_occlusion')
                os.makedirs(char_dir, exist_ok=True)
                os.makedirs(part_root_dir, exist_ok=True)

                base_img = tensor_to_rgb_uint8(images[b])
                map_h, map_w = feat2d.shape[1], feat2d.shape[2]
                if base_img.dtype != np.uint8:
                    base_img = np.clip(base_img, 0, 255).astype(np.uint8)

                valid_pos_in_batch = (valid_indices[:, 0] == b).nonzero(as_tuple=False).flatten()
                sample_valid_rows = valid_indices[valid_indices[:, 0] == b]
                valid_w = get_valid_map_width(real_ratios, b, map_w)
                valid_img_w = get_valid_map_width(real_ratios, b, base_img.shape[1])

                for j in range(min(n_chars, sample_valid_rows.shape[0])):
                    global_valid_idx = int(valid_pos_in_batch[j].item())
                    q_idx = int(sample_valid_rows[j, 1].item())
                    logits_1char = ctc_logits_valid[global_valid_idx]
                    _, cls_id = get_score_from_ctc_logits(logits_1char, target_score)
                    ids_tokens = ids_decoded_valid[global_valid_idx] if global_valid_idx < len(ids_decoded_valid) else []
                    if hasattr(decoder, '_ids_list_to_string'):
                        try:
                            ids_str = decoder._ids_list_to_string(ids_tokens)
                        except Exception:
                            ids_str = ' '.join([str(t) for t in ids_tokens])
                    else:
                        ids_str = ' '.join([str(t) for t in ids_tokens])

                    stem = f'char{j:02d}_q{q_idx:02d}_cls{cls_id}_{safe_name(ids_str)}'
                    part_dir = os.path.join(part_root_dir, stem)
                    os.makedirs(part_dir, exist_ok=True)
                    pred_char = _char_at(line_meta['pred_idsText'], q_idx)
                    char_overlay_path = ''

                    if attn_weights is not None:
                        if attn_weights.dim() == 4:
                            attn_vec = attn_weights[b, :, q_idx, :].mean(dim=0)
                        elif attn_weights.dim() == 3:
                            attn_vec = attn_weights[b, q_idx, :]
                        else:
                            attn_vec = None

                        if attn_vec is not None:
                            if mem_mask is not None:
                                attn_vec = attn_vec.masked_fill(mem_mask[b], 0.0)

                            if attn_vec.numel() == map_h * map_w:
                                attn_map = unflatten_attn(attn_vec, h=map_h, w=map_w, flatten_order=flatten_order)
                                if valid_w < map_w:
                                    attn_map = attn_map.clone()
                                    attn_map[:, valid_w:] = 0.0

                                cam01_small = normalize_map(attn_map)
                                cam01 = resize_cam_to_image(cam01_small, (base_img.shape[0], base_img.shape[1]))
                                overlay = overlay_cam_on_image(base_img, cam01, alpha=0.45)

                                char_overlay_path = os.path.join(char_dir, stem + '_overlay.png')
                                Image.fromarray(overlay).save(char_overlay_path)
                                if save_npy:
                                    np.save(os.path.join(char_dir, stem + '_cam.npy'), cam01.astype(np.float32))

                                exported_char_maps += 1
                                img_name = f"{dataset_name}_batch{batch_idx:04d}_sample{b:02d}_char{j:02d}"
                                manifest_rows.append({
                                    'img_name': img_name,
                                    'cam_kind': 'char',
                                    'type': dataset_name,
                                    'image_path': char_overlay_path,
                                    'label_src': line_meta['label_src'],
                                    'label_tgt': line_meta['label_tgt'],
                                    'pred_text': line_meta['pred_text'],
                                    'pred_idsText': line_meta['pred_idsText'],
                                    'pred_ids': line_meta['pred_ids_line'],
                                    'char_ids_pred': ids_str,
                                    'pred_char': pred_char,
                                    'ids_valid': line_meta['ids_valid_line'],
                                    'NED_text_src': float(line_meta['ned_text_src']),
                                    'NED_text_tgt': float(line_meta['ned_text_tgt']),
                                    'NED_idsText_src': float(line_meta['ned_ids_src']),
                                    'NED_idsText_tgt': float(line_meta['ned_ids_tgt']),
                                    'char_index': int(j),
                                    'query_index': int(q_idx),
                                    'ctc_class_id': int(cls_id),
                                    'part_index': '',
                                    'part_token': '',
                                    'part_token_id': '',
                                    'part_frames': '',
                                    'part_node_kind': '',
                                    'part_node_token': '',
                                })

                    token_spans_all = extract_ctc_component_spans(logits_1char, decoder, leaf_only=False)
                    occlusion_parts = build_occlusion_node_maps_for_char(
                        decoder=decoder,
                        memory_1sample=memory[b:b + 1],
                        mem_mask_1sample=(mem_mask[b:b + 1] if mem_mask is not None else None),
                        q_idx=q_idx,
                        max_chars=max_chars,
                        logits_1char=logits_1char,
                        ids_str=ids_str,
                        token_spans_all=token_spans_all,
                        map_h=map_h,
                        map_w=map_w,
                        flatten_order=flatten_order,
                        occlusion_chunk_size=occlusion_chunk_size,
                    )

                    part_cap = len(occlusion_parts) if max_parts_per_char <= 0 else min(len(occlusion_parts), max_parts_per_char)
                    for part_idx, node_item in enumerate(occlusion_parts[:part_cap]):
                        part_cam = node_item['cam_small']
                        token_name = node_item['node_token']
                        node_label = node_item['node_label']
                        node_kind = node_item['node_kind']

                        if valid_w < map_w:
                            part_cam = part_cam.copy()
                            part_cam[:, valid_w:] = 0.0

                        part_cam = resize_cam_to_image(part_cam, (base_img.shape[0], base_img.shape[1]))
                        if valid_img_w < base_img.shape[1]:
                            part_cam = part_cam.copy()
                            part_cam[:, valid_img_w:] = 0.0
                        part_overlay = overlay_cam_on_image(base_img, part_cam, alpha=0.45)

                        part_stem = f'{node_kind}{part_idx:02d}_{safe_name(node_label)}'
                        part_overlay_path = os.path.join(part_dir, part_stem + '_overlay.png')
                        Image.fromarray(part_overlay).save(part_overlay_path)
                        if save_npy:
                            np.save(os.path.join(part_dir, part_stem + '_cam.npy'), part_cam.astype(np.float32))
                        exported_part_maps += 1
                        img_name = f"{dataset_name}_batch{batch_idx:04d}_sample{b:02d}_char{j:02d}_part{part_idx:02d}"
                        manifest_rows.append({
                            'img_name': img_name,
                            'cam_kind': 'part_occlusion',
                            'type': dataset_name,
                            'image_path': part_overlay_path,
                            'label_src': line_meta['label_src'],
                            'label_tgt': line_meta['label_tgt'],
                            'pred_text': line_meta['pred_text'],
                            'pred_idsText': line_meta['pred_idsText'],
                            'pred_ids': line_meta['pred_ids_line'],
                            'char_ids_pred': ids_str,
                            'pred_char': pred_char,
                            'ids_valid': line_meta['ids_valid_line'],
                            'NED_text_src': float(line_meta['ned_text_src']),
                            'NED_text_tgt': float(line_meta['ned_text_tgt']),
                            'NED_idsText_src': float(line_meta['ned_ids_src']),
                            'NED_idsText_tgt': float(line_meta['ned_ids_tgt']),
                            'char_index': int(j),
                            'query_index': int(q_idx),
                            'ctc_class_id': int(cls_id),
                            'part_index': int(part_idx),
                            'part_token': node_label,
                            'part_token_id': '',
                            'part_frames': '',
                            'part_node_kind': node_kind,
                            'part_node_token': token_name,
                        })

                exported_samples += 1

        preds = None
        post_result = None
        batch_tensor = None
        images = None
        x = None
        memory = None
        feat2d = None
        char_feat_all = None
        valid_logits = None
        attn_weights = None
        char_feat_2d = None
        valid_len_pred = None
        valid_mask = None
        char_feat_valid = None
        style_code_valid = None
        style_line = None
        style_valid = None
        ctc_logits_valid = None
        ids_decoded_valid = None
        valid_indices = None
        release_cuda_memory()
        pbar.update(1)

    pbar.close()

    import pandas as pd
    df = pd.DataFrame(manifest_rows, columns=[
        'img_name',
        'cam_kind',
        'type',
        'image_path',
        'label_src',
        'label_tgt',
        'pred_text',
        'pred_idsText',
        'pred_ids',
        'char_ids_pred',
        'pred_char',
        'ids_valid',
        'NED_text_src',
        'NED_text_tgt',
        'NED_idsText_src',
        'NED_idsText_tgt',
        'char_index',
        'query_index',
        'ctc_class_id',
        'part_index',
        'part_token',
        'part_token_id',
        'part_frames',
        'part_node_kind',
        'part_node_token',
    ])
    df.to_excel(xlsx_path, index=False)

    return exported_samples, exported_char_maps, exported_part_maps, xlsx_path


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    cfg = prepare_cfg(cfg)

    trainer = Trainer(cfg, mode='eval')

    data_dirs_list = [[
        # r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/train_lmdb',
        # r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_correct',
        r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_fakedv2',
    ]]

    save_dir = FLAGS.get('save_dir')
    if not save_dir:
        save_dir = os.path.join(cfg.cfg['Global']['output_dir'], 'char_locator_cam')

    max_samples = int(FLAGS.get('max_samples', 200))
    total_exported_samples = 0
    total_exported_char_maps = 0
    total_exported_part_maps = 0
    all_xlsx_paths = []

    for data_dirs in data_dirs_list:
        for datadir in data_dirs:
            if max_samples > 0 and total_exported_samples >= max_samples:
                break

            dataset_name = datadir[:-1].split('/')[-1] if datadir.endswith('/') else datadir.split('/')[-1]
            config_each = trainer.cfg.copy()
            if 'RatioDataSet' in config_each['Eval']['dataset']['name']:
                config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            else:
                config_each['Eval']['dataset']['data_dir'] = datadir

            valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
            trainer.logger.info(f'{datadir} valid dataloader has {len(valid_dataloader)} iters')

            remaining = (max_samples - total_exported_samples) if max_samples > 0 else -1
            save_dir_each = os.path.join(save_dir, dataset_name)
            exported_samples, exported_char_maps, exported_part_maps, xlsx_path = dump_char_locator_cam(
                trainer=trainer,
                valid_dataloader=valid_dataloader,
                save_dir=save_dir_each,
                dataset_name=dataset_name,
                max_samples=remaining,
                max_chars_per_sample=int(FLAGS.get('max_chars_per_sample', 12)),
                max_parts_per_char=int(FLAGS.get('max_parts_per_char', 0)),
                target_score=str(FLAGS.get('target_score', 'mean_top_class')),
                save_npy=bool(FLAGS.get('save_npy', False)),
                flatten_order=str(FLAGS.get('flatten_order', 'wh')),
                occlusion_chunk_size=int(FLAGS.get('occlusion_chunk_size', 32)),
                save_only_cross_match=bool(FLAGS.get('save_only_cross_match', False)),
            )
            total_exported_samples += exported_samples
            total_exported_char_maps += exported_char_maps
            total_exported_part_maps += exported_part_maps
            all_xlsx_paths.append(xlsx_path)

    print(f'[DONE] exported_samples={total_exported_samples}, exported_char_maps={total_exported_char_maps}, exported_part_maps={total_exported_part_maps}')
    for p in all_xlsx_paths:
        print(f'[DONE] manifest={p}')
    print(f'[DONE] save_dir={save_dir}')


if __name__ == '__main__':
    main()
