import argparse
import base64
import csv
import io
import json
import math
import os
import time

import lmdb
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


DEFAULT_API_KEY = r'sk-318be36ce78749cdb458a1c5b1989060'
DEFAULT_MODEL_NAME = 'deepseek-v3.2'
DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DEFAULT_FAILED_PRED = '#'
DEFAULT_CACHE_FORMAT_VERSION = 3
DEFAULT_REQUEST_TIMEOUT_SEC = 3.0
DEFAULT_CORRECT_LMDB = r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_correct'
DEFAULT_FAKED_LMDB = r'/ipfs/lirunrui/lmdb_dataset/visual_c3_new_textline/test_lmdb/test_fakedv2'
DEFAULT_PROMPT = (
    'A faked character is a non-existent Chinese character formed by an incorrect '
    'combination of radicals or components. '
    'Recognize the handwritten text and return only a JSON object with exactly two string fields: '
    '{"src":"...","tgt":"..."}. '
    'The "src" field must preserve reading order and replace every fake character with a single uppercase X. '
    'The "tgt" field must preserve reading order, must not contain X, and should use the intended normal '
    'character at fake-character positions. '
    'Do not output markdown, code fences, comments, or any extra keys. '
    'Example: {"src":"text_with_X_marks","tgt":"corrected_text_without_X"}'
)


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


def clean_prediction_text(text):
    text = normalize_text(text)
    if text.startswith('```'):
        lines = []
        for line in text.splitlines():
            if line.strip().startswith('```'):
                continue
            lines.append(line)
        text = '\n'.join(lines).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text.strip()


def normalize_prediction_line(text):
    text = normalize_text(text).replace('\n', '').strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
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
            'doubao_eval_outputs',
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


def pil_image_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f'data:image/png;base64,{image_b64}'


def ensure_min_height(image, min_height):
    if image is None or min_height <= 0:
        return image
    if image.height >= min_height:
        return image
    new_width = max(1, int(round(image.width * (float(min_height) / float(image.height)))))
    return image.resize((new_width, min_height), Image.BICUBIC)


def extract_message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(normalize_text(item.get('text', '')))
            else:
                parts.append(normalize_text(item))
        return ''.join(parts)
    return normalize_text(content)


def extract_json_payload(text):
    text = clean_prediction_text(text)
    if not text:
        return None

    candidates = [text]
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidates.append(text[start_idx:end_idx + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def parse_json_prediction(text):
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        return DEFAULT_FAILED_PRED, DEFAULT_FAILED_PRED

    pred_src = (
        payload.get('pred_src')
        or payload.get('src')
        or payload.get('with_x')
        or payload.get('src_text')
    )
    pred_tgt = (
        payload.get('pred_tgt')
        or payload.get('tgt')
        or payload.get('without_x')
        or payload.get('tgt_text')
    )

    pred_src = normalize_prediction_line(pred_src) or DEFAULT_FAILED_PRED
    pred_tgt = normalize_prediction_line(pred_tgt) or DEFAULT_FAILED_PRED
    return pred_src, pred_tgt


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


class DoubaoPredictor:

    def __init__(self, args, cache_path):
        self.args = args
        self.cache_path = cache_path
        self.cache = load_prediction_cache(cache_path)
        self.client = None
        self.stats = {
            'cache_hits': 0,
            'api_calls': 0,
            'api_errors': 0,
        }

    def _get_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    'openai package is missing. Install it with: pip install openai'
                ) from exc
            api_key = self.args.api_key or os.getenv('ARK_API_KEY', '')
            if not api_key:
                raise RuntimeError(
                    'ARK api key is missing. Set ARK_API_KEY or pass --api_key.'
                )
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.args.base_url,
                # timeout=self.args.request_timeout_sec,
                # max_retries=0,
            )
        return self.client

    def _call_api_once(self, image):
        client = self._get_client()
        messages = [
            {
                'role': 'user',
                'content': [{
                    "type": "image_url",
                    "image_url": {"url":pil_image_to_data_url(image)},
                    # 输入图像的最小像素阈值，小于该值图像会放大，直到总像素大于min_pixels
                    "min_pixels": 32 * 32 * 3,
                    # 输入图像的最大像素阈值，超过该值图像会缩小，直到总像素低于max_pixels
                    "max_pixels": 32 * 32 * 8192,
                },{
                    "type": "text",
                    "text": self.args.prompt,
                }],
            }
        ]
        completion = client.chat.completions.create(
            model=self.args.model_name,
            messages=messages,
            # timeout=self.args.request_timeout_sec,
        )
        return extract_message_text(completion.choices[0].message.content)

    def predict(self, sample_key, image):
        cached = self.cache.get(sample_key)
        if (
            cached
            and cached.get('status') == 'ok'
            and cached.get('cache_format_version') == DEFAULT_CACHE_FORMAT_VERSION
            and 'pred_src' in cached
            and 'pred_tgt' in cached
        ):
            self.stats['cache_hits'] += 1
            return (
                cached.get('pred_src', DEFAULT_FAILED_PRED),
                cached.get('pred_tgt', DEFAULT_FAILED_PRED),
                True,
                cached.get('latency_sec'),
                'ok',
                '',
            )

        work_image = image
        attempt = 1
        max_attempts = None if self.args.retry_forever else max(1, self.args.max_retries)
        while True:
            try:
                start_time = time.time()
                response_text = self._call_api_once(work_image)
                # print(f'response_text:{response_text}')
                pred_src, pred_tgt = parse_json_prediction(response_text)
                latency_sec = time.time() - start_time
                self.stats['api_calls'] += 1
                record = {
                    'sample_key': sample_key,
                    'status': 'ok',
                    'cache_format_version': DEFAULT_CACHE_FORMAT_VERSION,
                    'pred_src': pred_src,
                    'pred_tgt': pred_tgt,
                    'raw_response': response_text,
                    'latency_sec': latency_sec,
                    'model_name': self.args.model_name,
                }
                self.cache[sample_key] = record
                append_prediction_cache(self.cache_path, record)
                if self.args.sleep_sec > 0:
                    time.sleep(self.args.sleep_sec)
                return pred_src, pred_tgt, False, latency_sec, 'ok', ''
            except Exception as exc:
                self.stats['api_errors'] += 1
                last_error = str(exc)
                tqdm.write(f'Attempt {attempt} failed.')
                tqdm.write(f'Sample: {sample_key}')
                tqdm.write(f'Error: {last_error}')
                if work_image is not None and work_image.height < self.args.min_image_height:
                    resized_image = ensure_min_height(work_image, self.args.min_image_height)
                    if resized_image.size != work_image.size:
                        work_image = resized_image
                        tqdm.write(
                            f'Resized image for retry: width={work_image.width}, '
                            f'height={work_image.height}'
                        )
                if max_attempts is not None and attempt >= max_attempts:
                    tqdm.write(
                        f'Giving up sample after {max_attempts} attempts: {sample_key}'
                    )
                    error_record = {
                        'sample_key': sample_key,
                        'status': 'error',
                        'cache_format_version': DEFAULT_CACHE_FORMAT_VERSION,
                        'pred_src': DEFAULT_FAILED_PRED,
                        'pred_tgt': DEFAULT_FAILED_PRED,
                        'raw_response': '',
                        'latency_sec': None,
                        'model_name': self.args.model_name,
                        'error': last_error,
                    }
                    self.cache[sample_key] = error_record
                    append_prediction_cache(self.cache_path, error_record)
                    return (
                        DEFAULT_FAILED_PRED,
                        DEFAULT_FAILED_PRED,
                        False,
                        None,
                        'error',
                        last_error,
                    )
                attempt += 1
                if self.args.retry_sleep_sec > 0:
                    time.sleep(self.args.retry_sleep_sec)


def evaluate_dataset(dataset_name, dataset_path, predictor, output_rows, det_inputs, limit=0):
    reader = HierarchicalLMDBReader(dataset_path)

    src_neds = []
    tgt_neds = []
    src_match = 0
    tgt_match = 0
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

                pred_src, pred_tgt, cached, latency_sec, status, api_error = predictor.predict(
                    sample_key, sample['image']
                )
                error_message = api_error

                pred_src_norm = replace_punctuation(pred_src)
                pred_tgt_norm = replace_punctuation(pred_tgt)

                if predictor.args.verbose:
                    tqdm.write(
                        f'gt_src: {gt_src_norm}, gt_tgt: {gt_tgt_norm}, '
                        f'pred_src: {pred_src_norm}, pred_tgt: {pred_tgt_norm}'
                    )

                src_ned = calc_ned(pred_src_norm, gt_src_norm)
                tgt_ned = calc_ned(pred_tgt_norm, gt_tgt_norm)

                src_neds.append(src_ned)
                tgt_neds.append(tgt_ned)
                if int(src_ned) == 1:
                    src_match += 1
                if int(tgt_ned) == 1:
                    tgt_match += 1

                det_inputs['gts'].append(gt_src_norm)
                det_inputs['preds'].append(pred_src_norm)

                output_rows.append({
                    'img_name': sample_key.replace(':', '_').replace('/', '_'),
                    'sample_key': sample_key,
                    'type': dataset_name,
                    'label_src': gt_src_norm,
                    'label_tgt': gt_tgt_norm,
                    'pred_src': pred_src_norm,
                    'pred_tgt': pred_tgt_norm,
                    'NED_src': float(src_ned),
                    'NED_tgt': float(tgt_ned),
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

    summary = {
        'dataset_name': dataset_name,
        'num': num_samples,
        'src_acc': (src_match / num_samples) if num_samples else 0.0,
        'tgt_acc': (tgt_match / num_samples) if num_samples else 0.0,
        'src_ned': mean_or_zero(src_neds),
        'tgt_ned': mean_or_zero(tgt_neds),
    }
    return summary


def summarize_rows(dataset_name, rows):
    src_neds = [float(row['NED_src']) for row in rows]
    tgt_neds = [float(row['NED_tgt']) for row in rows]
    src_match = sum(1 for row in rows if int(float(row['NED_src'])) == 1)
    tgt_match = sum(1 for row in rows if int(float(row['NED_tgt'])) == 1)
    num_samples = len(rows)
    return {
        'dataset_name': dataset_name,
        'num': num_samples,
        'src_acc': (src_match / num_samples) if num_samples else 0.0,
        'tgt_acc': (tgt_match / num_samples) if num_samples else 0.0,
        'src_ned': mean_or_zero(src_neds),
        'tgt_ned': mean_or_zero(tgt_neds),
    }


def print_summary(summary):
    print(
        f"{summary['dataset_name']}:\t "
        f"text_src_acc: {100 * summary['src_acc']:6g}, "
        f"text_src_NED:{100 * summary['src_ned']:6g}, "
        f"text_tgt_acc: {100 * summary['tgt_acc']:6g}, "
        f"text_tgt_NED:{100 * summary['tgt_ned']:6g}"
    )


def write_csv(csv_path, rows):
    fieldnames = [
        'img_name',
        'sample_key',
        'type',
        'label_src',
        'label_tgt',
        'pred_src',
        'pred_tgt',
        'NED_src',
        'NED_tgt',
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
        description='Evaluate Doubao OCR on Visual-C3 textline fake-character data.'
    )
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY)
    parser.add_argument('--base_url', type=str, default=DEFAULT_BASE_URL)
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
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
        default=r'/lirunrui/OpenOCR/outputs/output_new_visualC3_textline/eval_LLM_and_OCRtools',
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
        '--max_retries',
        type=int,
        default=3,
        help='Maximum retries per sample when --retry_forever is not set.',
    )
    parser.add_argument(
        '--retry_forever',
        action='store_true',
        help='Retry a failed sample indefinitely.',
    )
    parser.add_argument('--retry_sleep_sec', type=float, default=3.0)
    parser.add_argument(
        '--request_timeout_sec',
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help='Timeout for a single API request in seconds.',
    )
    parser.add_argument('--sleep_sec', type=float, default=0.0)
    parser.add_argument('--min_image_height', type=int, default=14)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-sample gt/pred details.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)

    cache_path = args.cache_path or os.path.join(output_dir, 'pred_cache.jsonl')
    predictor = DoubaoPredictor(args=args, cache_path=cache_path)

    output_rows = []
    dataset_summaries = []
    det_inputs = {'gts': [], 'preds': []}

    for dataset_name, dataset_path in build_dataset_jobs(args):
        summary = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            predictor=predictor,
            output_rows=output_rows,
            det_inputs=det_inputs,
            limit=args.limit,
        )
        dataset_summaries.append(summary)
        print_summary(summary)

    overall_summary = summarize_rows('total (text)', output_rows)
    print_summary(overall_summary)

    det_metric = calculate_cuo_metric_compact(det_inputs['gts'], det_inputs['preds'], x_token='X')
    print('\nCuo detection metrics (src/pred_x):')
    if det_metric:
        print(
            f"N_sent={det_metric['N_sent']} | "
            f"clean={det_metric['N_clean_sent']} | "
            f"error={det_metric['N_error_sent']}"
        )
        print(
            f"Char_P={fmt(det_metric['Char_P'])}%  "
            f"Char_R={fmt(det_metric['Char_R'])}%  "
            f"Char_F1={fmt(det_metric['Char_F1'])}%"
        )
        print(
            f"Char_BalAcc={fmt(det_metric['Char_BalAcc'])}%  "
            f"Char_MCC={fmt(det_metric['Char_MCC'])}"
        )
        print(
            f"Sent_FA={fmt(det_metric['Sent_FA'])}%  "
            f"Sent_EM={fmt(det_metric['Sent_EM'])}%"
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
        f'correct_lmdb_path: {args.correct_lmdb_path}',
        f'faked_lmdb_path: {args.faked_lmdb_path}',
        f'only_faked: {args.only_faked}',
        f'limit_per_dataset: {args.limit}',
        f'cache_path: {cache_path}',
        f'cache_hits: {predictor.stats["cache_hits"]}',
        f'api_calls: {predictor.stats["api_calls"]}',
        f'api_errors: {predictor.stats["api_errors"]}',
        f'max_retries: {args.max_retries}',
        f'retry_forever: {args.retry_forever}',
        f'request_timeout_sec: {args.request_timeout_sec}',
        '',
    ]

    for summary in dataset_summaries:
        stats_lines.append(
            f"{summary['dataset_name']}: "
            f"src_acc={100 * summary['src_acc']:.4f}%, "
            f"src_NED={100 * summary['src_ned']:.4f}%, "
            f"tgt_acc={100 * summary['tgt_acc']:.4f}%, "
            f"tgt_NED={100 * summary['tgt_ned']:.4f}%"
        )

    stats_lines.extend([
        '',
        (
            f"total: src_acc={100 * overall_summary['src_acc']:.4f}%, "
            f"src_NED={100 * overall_summary['src_ned']:.4f}%, "
            f"tgt_acc={100 * overall_summary['tgt_acc']:.4f}%, "
            f"tgt_NED={100 * overall_summary['tgt_ned']:.4f}%"
        ),
        '',
    ])

    if det_metric:
        stats_lines.extend([
            'Cuo detection metrics (src/pred_x):',
            (
                f"N_sent={det_metric['N_sent']}, "
                f"N_clean_sent={det_metric['N_clean_sent']}, "
                f"N_error_sent={det_metric['N_error_sent']}"
            ),
            (
                f"Char_P={fmt(det_metric['Char_P'])}%, "
                f"Char_R={fmt(det_metric['Char_R'])}%, "
                f"Char_F1={fmt(det_metric['Char_F1'])}%"
            ),
            (
                f"Char_BalAcc={fmt(det_metric['Char_BalAcc'])}%, "
                f"Char_MCC={fmt(det_metric['Char_MCC'])}"
            ),
            f"Sent_FA={fmt(det_metric['Sent_FA'])}%, Sent_EM={fmt(det_metric['Sent_EM'])}%",
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
