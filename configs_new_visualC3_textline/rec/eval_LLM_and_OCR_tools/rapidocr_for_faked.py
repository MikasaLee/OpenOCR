import argparse
import csv
import io
import importlib
import json
import math
import os
import sys

import lmdb
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


DEFAULT_MODEL_NAME = 'rapidocr_server'
DEFAULT_FAILED_PRED = '#'
DEFAULT_CACHE_FORMAT_VERSION = 4
DEFAULT_CONF_THR = 0.2
DEFAULT_OCR_VERSION = 'PP-OCRv4'
DEFAULT_MODEL_TYPE = 'server'
DEFAULT_CLS_OCR_VERSION = 'PP-OCRv4'
DEFAULT_CLS_MODEL_TYPE = 'mobile'
DEFAULT_LANG_TYPE = 'ch'
DEFAULT_BACKEND = 'onnxruntime'
DEFAULT_CORRECT_LMDB = r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_correct'
DEFAULT_FAKED_LMDB = r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_fakedv2'
DEFAULT_OUTPUT_ROOT = r'/lirunrui/OpenOCR/outputs/output_new_visualC3_textline/eval_LLM_and_OCRtools'


def normalize_text(text):
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    else:
        text = str(text)
    text = text.replace('\ufeff', '').replace('\r', '').strip()
    text = text.replace('<UNRECOG>', '')
    text = text.replace('<IMAGE>', '')
    return text


def replace_punctuation(text):
    text = normalize_text(text)
    if not text:
        return ''
    mapping = {
        '\uFF0C': ',',
        '\u3002': '.',
        '\uFF01': '!',
        '\uFF1F': '?',
        '\uFF1B': ';',
        '\uFF1A': ':',
        '\u201C': '"',
        '\u201D': '"',
        '\u2018': "'",
        '\u2019': "'",
    }
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    return text


def split_src_tgt(gt_raw):
    text = normalize_text(gt_raw)
    if not text:
        return '', ''
    for sep in ('\t', '<unk>'):
        if sep in text:
            parts = text.split(sep, 1)
            return normalize_text(parts[0]), normalize_text(parts[1])
    return text, ''


def safe_div(num, den):
    return None if den == 0 else (num / den)


def safe_pct(num, den):
    value = safe_div(num, den)
    return None if value is None else value * 100.0


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def fmt(value):
    return 'N/A' if value is None else f'{value:.3f}'


def calc_ned(pred, gt):
    pred = normalize_text(pred)
    gt = normalize_text(gt)
    return 1.0 - Levenshtein.normalized_distance(pred, gt)


def align_by_opcodes(gt, pred, gap_char=None):
    aligned = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gt, pred):
        if tag in ('equal', 'replace'):
            len_a = i2 - i1
            len_b = j2 - j1
            match_len = min(len_a, len_b)
            for offset in range(match_len):
                gt_idx = i1 + offset
                pred_idx = j1 + offset
                aligned.append((gt[gt_idx], pred[pred_idx], gt_idx))
            for gt_idx in range(i1 + match_len, i2):
                aligned.append((gt[gt_idx], gap_char, gt_idx))
            for pred_idx in range(j1 + match_len, j2):
                aligned.append((gap_char, pred[pred_idx], None))
        elif tag == 'delete':
            for gt_idx in range(i1, i2):
                aligned.append((gt[gt_idx], gap_char, gt_idx))
        elif tag == 'insert':
            for pred_idx in range(j1, j2):
                aligned.append((gap_char, pred[pred_idx], None))
        else:
            raise ValueError(f'Unknown opcode tag: {tag}')
    return aligned


def calculate_cuo_metric_compact(gt_sentences, pred_sentences, x_token='X'):
    num_sent = min(len(gt_sentences), len(pred_sentences))
    if num_sent == 0:
        return {}

    num_clean_sent = 0
    num_error_sent = 0
    sent_fa = 0
    sent_em = 0
    tp = 0
    fp_align = 0
    fp_ins = 0
    fn = 0
    tn = 0

    for gt, pred in zip(gt_sentences[:num_sent], pred_sentences[:num_sent]):
        aligned = align_by_opcodes(gt, pred, gap_char=None)
        gt_x_pos = set()
        pred_x_pos = set()
        inserted_x = 0

        for gt_char, pred_char, gt_idx in aligned:
            gt_is_x = gt_char == x_token
            pred_is_x = pred_char == x_token
            if gt_idx is not None and gt_is_x:
                gt_x_pos.add(gt_idx)
            if pred_is_x:
                if gt_idx is None:
                    inserted_x += 1
                else:
                    pred_x_pos.add(gt_idx)
            if gt_idx is None:
                if pred_is_x:
                    fp_ins += 1
            else:
                if gt_is_x:
                    if pred_is_x:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred_is_x:
                        fp_align += 1
                    else:
                        tn += 1

        gt_has_x = len(gt_x_pos) > 0
        pred_has_x = (len(pred_x_pos) > 0) or (inserted_x > 0)
        if gt_has_x:
            num_error_sent += 1
            if pred_has_x and inserted_x == 0 and pred_x_pos == gt_x_pos:
                sent_em += 1
        else:
            num_clean_sent += 1
            if pred_has_x:
                sent_fa += 1

    fp_total = fp_align + fp_ins
    char_p = safe_pct(tp, tp + fp_total)
    char_r = safe_pct(tp, tp + fn)
    char_f1 = None
    if char_p is not None and char_r is not None and (char_p + char_r) > 0:
        char_f1 = 2 * char_p * char_r / (char_p + char_r)

    spec = safe_div(tn, tn + fp_align)
    sens = safe_div(tp, tp + fn)
    char_bal_acc = None
    if spec is not None and sens is not None:
        char_bal_acc = (spec + sens) / 2.0 * 100.0

    char_mcc = None
    numerator = (tp * tn) - (fp_align * fn)
    denominator = (tp + fp_align) * (tp + fn) * (tn + fp_align) * (tn + fn)
    if denominator > 0:
        char_mcc = numerator / math.sqrt(denominator)

    return {
        'N_sent': num_sent,
        'N_clean_sent': num_clean_sent,
        'N_error_sent': num_error_sent,
        'Char_P': char_p,
        'Char_R': char_r,
        'Char_F1': char_f1,
        'Char_BalAcc': char_bal_acc,
        'Char_MCC': char_mcc,
        'Sent_FA': safe_pct(sent_fa, num_clean_sent),
        'Sent_EM': safe_pct(sent_em, num_error_sent),
    }


def sanitize_filename(text):
    text = normalize_text(text)
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ('-', '_', '.'):
            allowed.append(ch)
        else:
            allowed.append('_')
    sanitized = ''.join(allowed).strip('._')
    return sanitized or 'run'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_output_dir(args):
    model_tag = sanitize_filename(args.model_name)
    if args.output_dir:
        base_dir = os.path.abspath(args.output_dir)
    else:
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ocr_tool_eval_outputs',
        )
    return os.path.join(base_dir, model_tag)


def discover_lmdb_dirs(root_path):
    root_path = os.path.normpath(root_path)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f'LMDB path does not exist: {root_path}')
    if not os.path.isdir(root_path):
        raise NotADirectoryError(f'LMDB path is not a directory: {root_path}')

    lmdb_dirs = []
    for dirpath, _, filenames in os.walk(root_path, followlinks=True):
        if 'data.mdb' in filenames:
            lmdb_dirs.append(os.path.normpath(dirpath))
    if not lmdb_dirs:
        raise FileNotFoundError(f'No LMDB directories found under: {root_path}')
    lmdb_dirs.sort()
    return lmdb_dirs


class HierarchicalLMDBReader:

    def __init__(self, root_path):
        self.root_path = os.path.normpath(root_path)
        self.entries = []
        total_samples = 0

        for lmdb_dir in discover_lmdb_dirs(self.root_path):
            env = lmdb.open(
                lmdb_dir,
                max_readers=1,
                readonly=True,
                create=False,
                readahead=False,
                meminit=False,
                lock=False,
            )
            with env.begin(write=False) as txn:
                raw_num = txn.get(b'num-samples')
                if raw_num is None:
                    env.close()
                    continue
                if isinstance(raw_num, bytes):
                    raw_num = raw_num.decode('utf-8', errors='ignore')
                num_samples = int(raw_num)

            rel_dir = os.path.relpath(lmdb_dir, self.root_path).replace('\\', '/')
            self.entries.append({
                'dirpath': lmdb_dir,
                'rel_dir': rel_dir,
                'env': env,
                'num_samples': num_samples,
            })
            total_samples += num_samples

        if not self.entries:
            raise RuntimeError(f'No readable LMDB entries found under: {self.root_path}')
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def close(self):
        for entry in self.entries:
            entry['env'].close()

    def iter_samples(self, dataset_name, limit=0):
        yielded = 0
        for entry_idx, entry in enumerate(self.entries):
            rel_tag = entry['rel_dir']
            if rel_tag == '.':
                rel_tag = f'leaf_{entry_idx:04d}'

            with entry['env'].begin(write=False) as txn:
                for sample_idx in range(1, entry['num_samples'] + 1):
                    sample_key = f'{dataset_name}:{rel_tag}:{sample_idx:09d}'
                    label = ''
                    image = None
                    read_error = ''
                    try:
                        label_key = f'label-{sample_idx:09d}'.encode()
                        image_key = f'image-{sample_idx:09d}'.encode()
                        label_bin = txn.get(label_key)
                        image_bin = txn.get(image_key)
                        if label_bin is None or image_bin is None:
                            raise KeyError('missing label or image record')
                        label = label_bin.decode('utf-8', errors='ignore')
                        with Image.open(io.BytesIO(image_bin)) as pil_image:
                            image = pil_image.convert('RGB')
                    except Exception as exc:
                        read_error = str(exc)

                    yield {
                        'sample_key': sample_key,
                        'label': label,
                        'image': image,
                        'read_error': read_error,
                    }

                    yielded += 1
                    if limit > 0 and yielded >= limit:
                        return


def pil_to_numpy_rgb(image):
    if image is None:
        return None
    return np.asarray(image.convert('RGB'))


def load_prediction_cache(cache_path):
    cache = {}
    if not os.path.exists(cache_path):
        return cache

    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_key = record.get('sample_key')
            if sample_key:
                cache[sample_key] = record
    return cache


def append_prediction_cache(cache_path, record):
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def enum_from_value(enum_cls, raw_value):
    raw_key = str(raw_value).replace('-', '').replace('_', '').lower()
    for member in enum_cls:
        name_key = str(member.name).replace('_', '').lower()
        value_key = str(member.value).replace('-', '').replace('_', '').lower()
        if raw_key == name_key or raw_key == value_key:
            return member
    raise ValueError(f'{raw_value} is not a valid value for {enum_cls.__name__}')


def unpack_rapidocr_output(output):
    if isinstance(output, tuple) and len(output) == 2:
        core, maybe_elapse = output
        if hasattr(core, 'txts') or hasattr(core, 'scores') or hasattr(core, 'word_results'):
            return core, maybe_elapse
    return output, getattr(output, 'elapse', None)


def as_sequence(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_score(score):
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def flatten_text_tokens(value):
    tokens = []
    if value is None:
        return tokens
    if isinstance(value, str):
        token = normalize_text(value)
        if token:
            tokens.append(token)
        return tokens
    if isinstance(value, (list, tuple)):
        for item in value:
            tokens.extend(flatten_text_tokens(item))
    return tokens


def flatten_scores(value):
    scores = []
    if value is None:
        return scores
    if isinstance(value, (list, tuple)):
        for item in value:
            scores.extend(flatten_scores(item))
        return scores
    scores.append(normalize_score(value))
    return scores


def make_char_entries(token, score, source):
    token = normalize_text(token)
    if not token:
        return []
    normalized_score = normalize_score(score)
    return [
        {
            'char': ch,
            'score': normalized_score,
            'source': source,
        }
        for ch in token
    ]


def extract_char_conf_entries(word_results):
    entries = []

    def visit(item):
        if item is None:
            return

        if hasattr(item, 'words') and hasattr(item, 'confs'):
            tokens = flatten_text_tokens(getattr(item, 'words', None))
            confs = flatten_scores(getattr(item, 'confs', None))
            if not tokens:
                return
            if len(confs) == len(tokens):
                for token, score in zip(tokens, confs):
                    entries.extend(make_char_entries(token, score, 'word_info_confs'))
                return
            if len(confs) == 1:
                for token in tokens:
                    entries.extend(make_char_entries(token, confs[0], 'word_info_single_conf'))
                return
            for token in tokens:
                entries.extend(make_char_entries(token, None, 'word_info_no_conf'))
            return

        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], str)
        ):
            entries.extend(make_char_entries(item[0], item[1], 'legacy_word_result'))
            return

        if isinstance(item, (list, tuple)):
            for sub_item in item:
                visit(sub_item)

    visit(word_results)
    return entries


def build_char_conf_debug(pred_text, word_results, fallback_score=None, source='word_result'):
    entries = extract_char_conf_entries(word_results)
    if entries:
        return entries

    pred_text = normalize_text(pred_text)
    if pred_text:
        for ch in pred_text:
            entries.append({
                'char': ch,
                'score': normalize_score(fallback_score),
                'source': 'line_score_fallback',
            })
    return entries


def format_char_conf_debug(entries, max_items=200):
    if not entries:
        return 'N/A'
    chunks = []
    for idx, item in enumerate(entries):
        if idx >= max_items:
            chunks.append('...')
            break
        score = item.get('score')
        score_text = 'None' if score is None else f'{score:.4f}'
        source = item.get('source', '')
        chunks.append(f"{item.get('char', '')}:{score_text}({source})")
    return ' '.join(chunks)


def build_confdet_text(pred_text, word_results, conf_thr):
    pred_text = normalize_text(pred_text)
    if not pred_text:
        return DEFAULT_FAILED_PRED

    char_entries = extract_char_conf_entries(word_results)
    if not char_entries:
        return pred_text

    if len(char_entries) != len(pred_text):
        return pred_text

    return ''.join(
        'X' if item.get('score') is not None and item.get('score') < conf_thr else item.get('char', '')
        for item in char_entries
    )


def extract_prediction_from_legacy_result(result, conf_thr):
    rows = as_sequence(result)
    text_parts = []
    scores = []
    word_results = []

    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue

        if isinstance(row[0], str):
            text = normalize_text(row[0])
            score = normalize_score(row[1])
            if text:
                text_parts.append(text)
            if score is not None:
                scores.append(score)
            continue

        if len(row) >= 3 and isinstance(row[1], str):
            text = normalize_text(row[1])
            score = normalize_score(row[2])
            if text:
                text_parts.append(text)
            if score is not None:
                scores.append(score)

            if len(row) >= 5:
                char_boxes = as_sequence(row[3])
                char_texts = as_sequence(row[4])
                for char_text, char_box in zip(char_texts, char_boxes):
                    char_text = normalize_text(char_text)
                    if char_text:
                        word_results.append((char_text, score, char_box))

    pred_text = ''.join(text_parts) or DEFAULT_FAILED_PRED
    pred_score = (sum(scores) / len(scores)) if scores else None
    pred_confdet = build_confdet_text(pred_text, word_results, conf_thr)
    char_conf_debug = build_char_conf_debug(pred_text, word_results, pred_score, 'legacy_word_result')
    return pred_text, pred_confdet, pred_score, None, char_conf_debug


def extract_prediction_from_output(output, conf_thr):
    core, elapse = unpack_rapidocr_output(output)
    if isinstance(core, (list, tuple)):
        pred_text, pred_confdet, pred_score, _, char_conf_debug = extract_prediction_from_legacy_result(
            core,
            conf_thr,
        )
        return pred_text, pred_confdet, pred_score, elapse, char_conf_debug

    txts = as_sequence(getattr(core, 'txts', None))
    scores = as_sequence(getattr(core, 'scores', None))
    word_results = getattr(core, 'word_results', None)

    pred_text = ''.join(normalize_text(txt) for txt in txts) or DEFAULT_FAILED_PRED

    valid_scores = [normalize_score(score) for score in scores]
    valid_scores = [score for score in valid_scores if score is not None]
    if len(valid_scores) == 1:
        pred_score = valid_scores[0]
    elif valid_scores:
        pred_score = sum(valid_scores) / len(valid_scores)
    else:
        pred_score = None

    pred_confdet = build_confdet_text(pred_text, word_results, conf_thr)
    char_conf_debug = build_char_conf_debug(pred_text, word_results, pred_score)
    return pred_text, pred_confdet, pred_score, elapse, char_conf_debug


class RapidOCRPredictor:

    def __init__(self, args, cache_path):
        self.args = args
        self.cache_path = cache_path
        self.cache = load_prediction_cache(cache_path)
        self.engine = None
        self.rapidocr_module = None
        self.stats = {
            'cache_hits': 0,
            'predict_calls': 0,
            'predict_errors': 0,
        }

    def _build_params(self):
        if self.rapidocr_module is None:
            raise RuntimeError('RapidOCR module is not initialized.')

        engine_type = enum_from_value(self.rapidocr_module.EngineType, self.args.backend)
        det_lang = enum_from_value(self.rapidocr_module.LangDet, self.args.lang_type)
        cls_lang = enum_from_value(self.rapidocr_module.LangCls, 'ch')
        rec_lang = enum_from_value(self.rapidocr_module.LangRec, self.args.lang_type)
        model_type = enum_from_value(self.rapidocr_module.ModelType, self.args.model_type)
        ocr_version = enum_from_value(self.rapidocr_module.OCRVersion, self.args.ocr_version)
        cls_model_type = enum_from_value(
            self.rapidocr_module.ModelType,
            self.args.cls_model_type,
        )
        cls_ocr_version = enum_from_value(
            self.rapidocr_module.OCRVersion,
            self.args.cls_ocr_version,
        )

        params = {
            'Det.engine_type': engine_type,
            'Global.min_height': self.args.min_height,
            'Global.width_height_ratio': self.args.width_height_ratio,
            'Det.lang_type': det_lang,
            'Det.model_type': model_type,
            'Det.ocr_version': ocr_version,
            'Cls.engine_type': engine_type,
            'Cls.lang_type': cls_lang,
            'Cls.model_type': cls_model_type,
            'Cls.ocr_version': cls_ocr_version,
            'Rec.engine_type': engine_type,
            'Rec.lang_type': rec_lang,
            'Rec.model_type': model_type,
            'Rec.ocr_version': ocr_version,
            'Rec.rec_batch_num': self.args.batch_size,
        }
        return params

    def _get_engine(self):
        if self.engine is None:
            RapidOCR = None
            import_errors = []
            import_targets = ['rapidocr']
            if self.args.backend == 'onnxruntime':
                import_targets.append('rapidocr_onnxruntime')
            elif self.args.backend == 'openvino':
                import_targets.append('rapidocr_openvino')
            elif self.args.backend == 'paddle':
                import_targets.append('rapidocr_paddle')
            elif self.args.backend == 'torch':
                import_targets.append('rapidocr_torch')

            script_dir = os.path.dirname(os.path.abspath(__file__))
            cwd = os.path.abspath(os.getcwd())
            removed_paths = []
            for path in list(sys.path):
                abs_path = os.path.abspath(path or '.')
                if abs_path in (script_dir, cwd):
                    removed_paths.append(path)
                    sys.path.remove(path)

            try:
                for module_name in import_targets:
                    try:
                        module = importlib.import_module(module_name)
                        module_file = os.path.abspath(getattr(module, '__file__', ''))
                        if module_name == 'rapidocr' and module_file == os.path.abspath(__file__):
                            raise RuntimeError(
                                f'module name conflict: imported current script {module_file}'
                            )
                        RapidOCR = getattr(module, 'RapidOCR')
                        self.rapidocr_module = module
                        break
                    except Exception as exc:
                        import_errors.append(f'{module_name}: {exc}')
            finally:
                for path in reversed(removed_paths):
                    sys.path.insert(0, path)

            if RapidOCR is None:
                raise RuntimeError(
                    'RapidOCR import failed. Tried: ' + ' | '.join(import_errors)
                )
            self.engine = RapidOCR(params=self._build_params())
        return self.engine

    def predict(self, sample_key, image):
        cached = self.cache.get(sample_key)
        if (
            cached
            and cached.get('status') == 'ok'
            and cached.get('cache_format_version') == DEFAULT_CACHE_FORMAT_VERSION
            and 'pred_text' in cached
            and 'pred_confdet' in cached
        ):
            self.stats['cache_hits'] += 1
            return (
                cached.get('pred_text', DEFAULT_FAILED_PRED),
                cached.get('pred_confdet', DEFAULT_FAILED_PRED),
                cached.get('pred_score'),
                cached.get('char_conf_debug', []),
                True,
                cached.get('latency_sec'),
                'ok',
                '',
            )

        try:
            engine = self._get_engine()
            image_array = pil_to_numpy_rgb(image)
            output = engine(
                image_array,
                use_det=self.args.use_det,
                use_cls=self.args.use_cls,
                use_rec=True,
                return_word_box=True,
                return_single_char_box=self.args.return_single_char_box,
            )
            pred_text, pred_confdet, pred_score, latency_sec, char_conf_debug = extract_prediction_from_output(
                output,
                self.args.conf_thr,
            )
            self.stats['predict_calls'] += 1

            record = {
                'sample_key': sample_key,
                'status': 'ok',
                'cache_format_version': DEFAULT_CACHE_FORMAT_VERSION,
                'pred_text': pred_text,
                'pred_confdet': pred_confdet,
                'pred_score': pred_score,
                'char_conf_debug': char_conf_debug,
                'latency_sec': latency_sec,
                'model_name': self.args.model_name,
            }
            self.cache[sample_key] = record
            append_prediction_cache(self.cache_path, record)
            return pred_text, pred_confdet, pred_score, char_conf_debug, False, latency_sec, 'ok', ''
        except Exception as exc:
            self.stats['predict_errors'] += 1
            error_message = str(exc)
            error_record = {
                'sample_key': sample_key,
                'status': 'error',
                'cache_format_version': DEFAULT_CACHE_FORMAT_VERSION,
                'pred_text': DEFAULT_FAILED_PRED,
                'pred_confdet': DEFAULT_FAILED_PRED,
                'pred_score': None,
                'char_conf_debug': [],
                'latency_sec': None,
                'model_name': self.args.model_name,
                'error': error_message,
            }
            self.cache[sample_key] = error_record
            append_prediction_cache(self.cache_path, error_record)
            return (
                DEFAULT_FAILED_PRED,
                DEFAULT_FAILED_PRED,
                None,
                [],
                False,
                None,
                'error',
                error_message,
            )


def evaluate_dataset(dataset_name, dataset_path, predictor, output_rows, det_inputs_text, det_inputs_confdet, limit=0):
    reader = HierarchicalLMDBReader(dataset_path)

    text_src_neds = []
    text_tgt_neds = []
    confdet_src_neds = []
    confdet_tgt_neds = []
    text_src_match = 0
    text_tgt_match = 0
    confdet_src_match = 0
    confdet_tgt_match = 0
    num_samples = 0

    pbar = None
    try:
        progress_total = min(len(reader), limit) if limit > 0 else len(reader)
        pbar = tqdm(
            total=progress_total,
            desc=f'eval {dataset_name}',
            leave=True,
            dynamic_ncols=True,
            miniters=1,
            unit='sample',
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar}| '
                '{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )
        for sample in reader.iter_samples(dataset_name=dataset_name, limit=limit):
            sample_key = sample['sample_key']
            try:
                gt_text = sample['label']
                gt_src_text, gt_tgt_text = split_src_tgt(gt_text)
                gt_src_norm = replace_punctuation(gt_src_text)
                gt_tgt_norm = replace_punctuation(gt_tgt_text)

                if sample['image'] is None:
                    tqdm.write(
                        f'[WARN] Failed to read sample {sample_key}: {sample["read_error"]}'
                    )
                    continue

                pred_text, pred_confdet, pred_score, char_conf_debug, cached, latency_sec, status, error_message = predictor.predict(
                    sample_key, sample['image']
                )
                pred_text_norm = replace_punctuation(pred_text)
                pred_confdet_norm = replace_punctuation(pred_confdet)

                if predictor.args.verbose:
                    tqdm.write(
                        f'gt_src: {gt_src_norm}, gt_tgt: {gt_tgt_norm}, '
                        f'pred_text: {pred_text_norm}, pred_confdet: {pred_confdet_norm}'
                    )
                if predictor.args.debug_char_conf:
                    tqdm.write(
                        'char_conf: ' + format_char_conf_debug(
                            char_conf_debug,
                            predictor.args.debug_char_limit,
                        )
                    )

                text_src_ned = calc_ned(pred_text_norm, gt_src_norm)
                text_tgt_ned = calc_ned(pred_text_norm, gt_tgt_norm)
                confdet_src_ned = calc_ned(pred_confdet_norm, gt_src_norm)
                confdet_tgt_ned = calc_ned(pred_confdet_norm, gt_tgt_norm)

                text_src_neds.append(text_src_ned)
                text_tgt_neds.append(text_tgt_ned)
                confdet_src_neds.append(confdet_src_ned)
                confdet_tgt_neds.append(confdet_tgt_ned)

                if int(text_src_ned) == 1:
                    text_src_match += 1
                if int(text_tgt_ned) == 1:
                    text_tgt_match += 1
                if int(confdet_src_ned) == 1:
                    confdet_src_match += 1
                if int(confdet_tgt_ned) == 1:
                    confdet_tgt_match += 1

                det_inputs_text['gts'].append(gt_src_norm)
                det_inputs_text['preds'].append(pred_text_norm)
                det_inputs_confdet['gts'].append(gt_src_norm)
                det_inputs_confdet['preds'].append(pred_confdet_norm)

                output_rows.append({
                    'img_name': sample_key.replace(':', '_').replace('/', '_'),
                    'sample_key': sample_key,
                    'type': dataset_name,
                    'label_src': gt_src_norm,
                    'label_tgt': gt_tgt_norm,
                    'pred_text': pred_text_norm,
                    'pred_confdet': pred_confdet_norm,
                    'pred_score': '' if pred_score is None else float(pred_score),
                    'char_conf_debug': json.dumps(char_conf_debug, ensure_ascii=False),
                    'NED_text_src': float(text_src_ned),
                    'NED_text_tgt': float(text_tgt_ned),
                    'NED_confdet_src': float(confdet_src_ned),
                    'NED_confdet_tgt': float(confdet_tgt_ned),
                    'cached': cached,
                    'status': status,
                    'latency_sec': '' if latency_sec is None else float(latency_sec),
                    'error': error_message,
                })

                num_samples += 1
                pbar.set_postfix_str(f'ok={num_samples}')
            except Exception as exc:
                tqdm.write(f'[WARN] Failed to process sample {sample_key}: {exc}')
                pbar.set_postfix_str(f'ok={num_samples}')
                continue
            finally:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        reader.close()

    return {
        'dataset_name': dataset_name,
        'num': num_samples,
        'text_src_acc': (text_src_match / num_samples) if num_samples else 0.0,
        'text_tgt_acc': (text_tgt_match / num_samples) if num_samples else 0.0,
        'text_src_ned': mean_or_zero(text_src_neds),
        'text_tgt_ned': mean_or_zero(text_tgt_neds),
        'confdet_src_acc': (confdet_src_match / num_samples) if num_samples else 0.0,
        'confdet_tgt_acc': (confdet_tgt_match / num_samples) if num_samples else 0.0,
        'confdet_src_ned': mean_or_zero(confdet_src_neds),
        'confdet_tgt_ned': mean_or_zero(confdet_tgt_neds),
    }


def summarize_rows(dataset_name, rows):
    num_samples = len(rows)
    return {
        'dataset_name': dataset_name,
        'num': num_samples,
        'text_src_acc': (sum(1 for row in rows if int(float(row['NED_text_src'])) == 1) / num_samples) if num_samples else 0.0,
        'text_tgt_acc': (sum(1 for row in rows if int(float(row['NED_text_tgt'])) == 1) / num_samples) if num_samples else 0.0,
        'text_src_ned': mean_or_zero([float(row['NED_text_src']) for row in rows]),
        'text_tgt_ned': mean_or_zero([float(row['NED_text_tgt']) for row in rows]),
        'confdet_src_acc': (sum(1 for row in rows if int(float(row['NED_confdet_src'])) == 1) / num_samples) if num_samples else 0.0,
        'confdet_tgt_acc': (sum(1 for row in rows if int(float(row['NED_confdet_tgt'])) == 1) / num_samples) if num_samples else 0.0,
        'confdet_src_ned': mean_or_zero([float(row['NED_confdet_src']) for row in rows]),
        'confdet_tgt_ned': mean_or_zero([float(row['NED_confdet_tgt']) for row in rows]),
    }


def print_summary(summary):
    print(
        f"{summary['dataset_name']}:\t "
        f"text_src_acc: {100 * summary['text_src_acc']:6g}, "
        f"text_src_NED:{100 * summary['text_src_ned']:6g}, "
        f"text_tgt_acc: {100 * summary['text_tgt_acc']:6g}, "
        f"text_tgt_NED:{100 * summary['text_tgt_ned']:6g}"
    )
    print(
        f"{summary['dataset_name']}:\t "
        f"confdet_src_acc: {100 * summary['confdet_src_acc']:6g}, "
        f"confdet_src_NED:{100 * summary['confdet_src_ned']:6g}, "
        f"confdet_tgt_acc: {100 * summary['confdet_tgt_acc']:6g}, "
        f"confdet_tgt_NED:{100 * summary['confdet_tgt_ned']:6g}"
    )


def write_csv(csv_path, rows):
    fieldnames = [
        'img_name',
        'sample_key',
        'type',
        'label_src',
        'label_tgt',
        'pred_text',
        'pred_confdet',
        'pred_score',
        'char_conf_debug',
        'NED_text_src',
        'NED_text_tgt',
        'NED_confdet_src',
        'NED_confdet_tgt',
        'cached',
        'status',
        'latency_sec',
        'error',
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx_if_possible(xlsx_path, rows):
    try:
        import pandas as pd
    except ImportError:
        print('[WARN] pandas is not installed, skip XLSX export.')
        return False

    df = pd.DataFrame(rows)
    df.to_excel(xlsx_path, index=False)
    return True


def build_dataset_jobs(args):
    jobs = []
    if not args.only_faked:
        jobs.append(('test_correct', args.correct_lmdb_path))
    jobs.append(('test_fakedv2', args.faked_lmdb_path))
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RapidOCR on Visual-C3 fake-character textline data.'
    )
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['onnxruntime', 'openvino', 'paddle', 'torch'])
    parser.add_argument('--ocr_version', type=str, default=DEFAULT_OCR_VERSION)
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=['mobile', 'server'])
    parser.add_argument('--cls_ocr_version', type=str, default=DEFAULT_CLS_OCR_VERSION)
    parser.add_argument('--cls_model_type', type=str, default=DEFAULT_CLS_MODEL_TYPE,
                        choices=['mobile', 'server'])
    parser.add_argument('--lang_type', type=str, default=DEFAULT_LANG_TYPE)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--conf_thr', type=float, default=DEFAULT_CONF_THR)
    parser.add_argument('--use_det', action='store_true',
                        help='Enable text detection explicitly.')
    parser.add_argument('--use_cls', action='store_true',
                        help='Enable orientation classification.')
    parser.add_argument('--return_single_char_box', action='store_true', default=True,
                        help='Request single-char boxes when possible.')
    parser.add_argument('--min_height', type=int, default=30)
    parser.add_argument('--width_height_ratio', type=float, default=8.0)
    parser.add_argument('--correct_lmdb_path', type=str, default=DEFAULT_CORRECT_LMDB)
    parser.add_argument('--faked_lmdb_path', type=str, default=DEFAULT_FAKED_LMDB)
    parser.add_argument(
        '--only_faked',
        action='store_true',
        help='Evaluate only test_fakedv2. By default test_correct is also evaluated.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help='Root output directory. Results are saved under <output_dir>/<model_name>/',
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default=None,
        help='JSONL cache file. Default: <output_dir>/pred_cache.jsonl',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Evaluate at most this many samples for each dataset. 0 means all samples.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-sample gt/pred details.',
    )
    parser.add_argument(
        '--debug_char_conf',
        action='store_true',
        help='Print per-character confidence debug info.',
    )
    parser.add_argument(
        '--debug_char_limit',
        type=int,
        default=200,
        help='Maximum number of char-confidence entries to print per sample.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)

    cache_path = args.cache_path or os.path.join(output_dir, 'pred_cache.jsonl')
    predictor = RapidOCRPredictor(args=args, cache_path=cache_path)

    output_rows = []
    dataset_summaries = []
    det_inputs_text = {'gts': [], 'preds': []}
    det_inputs_confdet = {'gts': [], 'preds': []}

    for dataset_name, dataset_path in build_dataset_jobs(args):
        summary = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            predictor=predictor,
            output_rows=output_rows,
            det_inputs_text=det_inputs_text,
            det_inputs_confdet=det_inputs_confdet,
            limit=args.limit,
        )
        dataset_summaries.append(summary)
        print_summary(summary)

    overall_summary = summarize_rows('total', output_rows)
    print_summary(overall_summary)

    det_text = calculate_cuo_metric_compact(
        det_inputs_text['gts'],
        det_inputs_text['preds'],
        x_token='X',
    )
    det_confdet = calculate_cuo_metric_compact(
        det_inputs_confdet['gts'],
        det_inputs_confdet['preds'],
        x_token='X',
    )

    print('\nCuo detection metrics (text):')
    if det_text:
        print(
            f"N_sent={det_text['N_sent']} | "
            f"clean={det_text['N_clean_sent']} | "
            f"error={det_text['N_error_sent']}"
        )
        print(
            f"Char_P={fmt(det_text['Char_P'])}%  "
            f"Char_R={fmt(det_text['Char_R'])}%  "
            f"Char_F1={fmt(det_text['Char_F1'])}%"
        )
        print(
            f"Char_BalAcc={fmt(det_text['Char_BalAcc'])}%  "
            f"Char_MCC={fmt(det_text['Char_MCC'])}"
        )
        print(
            f"Sent_FA={fmt(det_text['Sent_FA'])}%  "
            f"Sent_EM={fmt(det_text['Sent_EM'])}%"
        )
    else:
        print('No sentences available.')

    print('\nCuo detection metrics (confdet):')
    if det_confdet:
        print(
            f"N_sent={det_confdet['N_sent']} | "
            f"clean={det_confdet['N_clean_sent']} | "
            f"error={det_confdet['N_error_sent']}"
        )
        print(
            f"Char_P={fmt(det_confdet['Char_P'])}%  "
            f"Char_R={fmt(det_confdet['Char_R'])}%  "
            f"Char_F1={fmt(det_confdet['Char_F1'])}%"
        )
        print(
            f"Char_BalAcc={fmt(det_confdet['Char_BalAcc'])}%  "
            f"Char_MCC={fmt(det_confdet['Char_MCC'])}"
        )
        print(
            f"Sent_FA={fmt(det_confdet['Sent_FA'])}%  "
            f"Sent_EM={fmt(det_confdet['Sent_EM'])}%"
        )
    else:
        print('No sentences available.')

    csv_path = os.path.join(output_dir, 'predictions.csv')
    xlsx_path = os.path.join(output_dir, 'predictions.xlsx')
    stats_path = os.path.join(output_dir, 'statistics.txt')

    write_csv(csv_path, output_rows)
    xlsx_written = write_xlsx_if_possible(xlsx_path, output_rows)

    stats_lines = [
        f'model_name: {args.model_name}',
        f'backend: {args.backend}',
        f'ocr_version: {args.ocr_version}',
        f'model_type: {args.model_type}',
        f'cls_ocr_version: {args.cls_ocr_version}',
        f'cls_model_type: {args.cls_model_type}',
        f'lang_type: {args.lang_type}',
        f'batch_size: {args.batch_size}',
        f'conf_thr: {args.conf_thr}',
        f'debug_char_conf: {args.debug_char_conf}',
        f'debug_char_limit: {args.debug_char_limit}',
        f'use_det: {args.use_det}',
        f'use_cls: {args.use_cls}',
        f'return_single_char_box: {args.return_single_char_box}',
        f'min_height: {args.min_height}',
        f'width_height_ratio: {args.width_height_ratio}',
        f'correct_lmdb_path: {args.correct_lmdb_path}',
        f'faked_lmdb_path: {args.faked_lmdb_path}',
        f'only_faked: {args.only_faked}',
        f'limit_per_dataset: {args.limit}',
        f'cache_path: {cache_path}',
        f'cache_hits: {predictor.stats["cache_hits"]}',
        f'predict_calls: {predictor.stats["predict_calls"]}',
        f'predict_errors: {predictor.stats["predict_errors"]}',
        '',
    ]

    for summary in dataset_summaries:
        stats_lines.append(
            f"{summary['dataset_name']} text: "
            f"src_acc={100 * summary['text_src_acc']:.4f}%, "
            f"src_NED={100 * summary['text_src_ned']:.4f}%, "
            f"tgt_acc={100 * summary['text_tgt_acc']:.4f}%, "
            f"tgt_NED={100 * summary['text_tgt_ned']:.4f}%"
        )
        stats_lines.append(
            f"{summary['dataset_name']} confdet: "
            f"src_acc={100 * summary['confdet_src_acc']:.4f}%, "
            f"src_NED={100 * summary['confdet_src_ned']:.4f}%, "
            f"tgt_acc={100 * summary['confdet_tgt_acc']:.4f}%, "
            f"tgt_NED={100 * summary['confdet_tgt_ned']:.4f}%"
        )

    stats_lines.extend([
        '',
        (
            f"total text: src_acc={100 * overall_summary['text_src_acc']:.4f}%, "
            f"src_NED={100 * overall_summary['text_src_ned']:.4f}%, "
            f"tgt_acc={100 * overall_summary['text_tgt_acc']:.4f}%, "
            f"tgt_NED={100 * overall_summary['text_tgt_ned']:.4f}%"
        ),
        (
            f"total confdet: src_acc={100 * overall_summary['confdet_src_acc']:.4f}%, "
            f"src_NED={100 * overall_summary['confdet_src_ned']:.4f}%, "
            f"tgt_acc={100 * overall_summary['confdet_tgt_acc']:.4f}%, "
            f"tgt_NED={100 * overall_summary['confdet_tgt_ned']:.4f}%"
        ),
        '',
    ])

    if det_text:
        stats_lines.extend([
            'Cuo detection metrics (text):',
            (
                f"N_sent={det_text['N_sent']}, "
                f"N_clean_sent={det_text['N_clean_sent']}, "
                f"N_error_sent={det_text['N_error_sent']}"
            ),
            (
                f"Char_P={fmt(det_text['Char_P'])}%, "
                f"Char_R={fmt(det_text['Char_R'])}%, "
                f"Char_F1={fmt(det_text['Char_F1'])}%"
            ),
            (
                f"Char_BalAcc={fmt(det_text['Char_BalAcc'])}%, "
                f"Char_MCC={fmt(det_text['Char_MCC'])}"
            ),
            f"Sent_FA={fmt(det_text['Sent_FA'])}%, Sent_EM={fmt(det_text['Sent_EM'])}%",
            '',
        ])

    if det_confdet:
        stats_lines.extend([
            'Cuo detection metrics (confdet):',
            (
                f"N_sent={det_confdet['N_sent']}, "
                f"N_clean_sent={det_confdet['N_clean_sent']}, "
                f"N_error_sent={det_confdet['N_error_sent']}"
            ),
            (
                f"Char_P={fmt(det_confdet['Char_P'])}%, "
                f"Char_R={fmt(det_confdet['Char_R'])}%, "
                f"Char_F1={fmt(det_confdet['Char_F1'])}%"
            ),
            (
                f"Char_BalAcc={fmt(det_confdet['Char_BalAcc'])}%, "
                f"Char_MCC={fmt(det_confdet['Char_MCC'])}"
            ),
            f"Sent_FA={fmt(det_confdet['Sent_FA'])}%, Sent_EM={fmt(det_confdet['Sent_EM'])}%",
        ])

    with open(stats_path, 'w', encoding='utf-8') as f:
        for line in stats_lines:
            f.write(line + '\n')

    print(f'\nPredictions saved to {csv_path}')
    if xlsx_written:
        print(f'Predictions saved to {xlsx_path}')
    print(f'Statistics saved to {stats_path}')


if __name__ == '__main__':
    main()
