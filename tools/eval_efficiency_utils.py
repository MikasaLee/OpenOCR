import contextlib
import io
import time

import numpy as np
import torch
import torch.nn as nn


def sync_device(device):
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        torch.cuda.synchronize()


def split_batch_tensors(batch, device, max_tensor_items=5):
    tensor_items = []
    for item in batch[:max_tensor_items]:
        if not torch.is_tensor(item):
            break
        tensor_items.append(item)
    batch_tensor = [t.to(device) for t in tensor_items]
    batch_numpy = [t.numpy() for t in tensor_items]
    return batch_tensor, batch_numpy


class GTCOnlyDecoder(nn.Module):

    def __init__(self, gtc_decoder):
        super().__init__()
        self.gtc_decoder = gtc_decoder

    def forward(self, x, data=None):
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        return {'gtc_pred': self.gtc_decoder(x, data=data)}


def replace_gtc_decoder_with_gtc_only(model):
    base_model = getattr(model, 'module', model)
    decoder = getattr(base_model, 'decoder', None)
    if decoder is None or not hasattr(decoder, 'gtc_decoder'):
        return False
    base_model.decoder = GTCOnlyDecoder(decoder.gtc_decoder)
    return True


def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'params': int(total),
        'trainable_params': int(trainable),
    }


def _profile_flops_with_torch(model, sample_input, device):
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        return None, f'torch.profiler unavailable: {exc}'

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and sample_input.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    try:
        sync_device(device)
        with torch.no_grad():
            with profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = model(sample_input)
        sync_device(device)
        flops = 0
        for item in prof.key_averages():
            flops += int(getattr(item, 'flops', 0) or 0)
        if flops > 0:
            return {
                'flops': int(flops),
                'macs': None,
                'profile_method': 'torch.profiler',
            }, None
        return None, 'torch.profiler returned zero FLOPs'
    except Exception as exc:
        return None, f'torch.profiler failed: {exc}'


def _profile_macs_with_thop(model, sample_input, device):
    try:
        from thop import profile
    except Exception as exc:
        return None, f'thop unavailable: {exc}'

    try:
        sync_device(device)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            macs, _params = profile(model, inputs=(sample_input, ), verbose=False)
        sync_device(device)
        return {
            'flops': None,
            'macs': int(macs),
            'profile_method': 'thop',
        }, None
    except Exception as exc:
        return None, f'thop failed: {exc}'


def profile_model_compute(model, sample_input, device):
    stats = count_model_params(model)
    stats.update({
        'input_shape': tuple(int(x) for x in sample_input.shape),
        'flops': None,
        'macs': None,
        'profile_method': None,
        'profile_error': None,
    })

    sample = sample_input[:1].detach().contiguous()
    torch_stats, torch_error = _profile_flops_with_torch(model, sample, device)
    if torch_stats is not None:
        stats.update(torch_stats)
        return stats

    thop_stats, thop_error = _profile_macs_with_thop(model, sample, device)
    if thop_stats is not None:
        stats.update(thop_stats)
        return stats

    stats['profile_error'] = '; '.join([msg for msg in [torch_error, thop_error] if msg])
    return stats


def init_efficiency_stats(dataset_name, num_batches, cfg_batch_size, model):
    stats = count_model_params(model)
    stats.update({
        'dataset_name': dataset_name,
        'num_batches': int(num_batches),
        'cfg_batch_size': int(cfg_batch_size) if cfg_batch_size is not None else None,
        'num_samples': 0,
        'total_data': 0.0,
        'total_model': 0.0,
        'total_post': 0.0,
        'total_dump': 0.0,
        'model_batch_times': [],
        'input_shape': None,
        'flops': None,
        'macs': None,
        'profile_method': None,
        'profile_error': None,
    })
    return stats


def maybe_profile_first_batch(stats, model, batch_tensor, device):
    if stats.get('input_shape') is not None or not batch_tensor:
        return
    profile_stats = profile_model_compute(model, batch_tensor[0], device)
    for key in [
            'input_shape',
            'flops',
            'macs',
            'profile_method',
            'profile_error',
    ]:
        stats[key] = profile_stats.get(key)


def update_efficiency_stats(stats, num_samples, data_time, model_time, post_time, dump_time):
    stats['num_samples'] += int(num_samples)
    stats['total_data'] += float(data_time)
    stats['total_model'] += float(model_time)
    stats['total_post'] += float(post_time)
    stats['total_dump'] += float(dump_time)
    stats['model_batch_times'].append(float(model_time))


def _safe_div(a, b):
    return None if b == 0 else a / b


def finalize_efficiency_stats(stats):
    summary = dict(stats)
    num_batches = summary['num_batches']
    num_samples = summary['num_samples']
    model_batch_times = np.array(summary.pop('model_batch_times', []), dtype=np.float64)

    summary['total_eval'] = summary['total_data'] + summary['total_model'] + summary['total_post']
    summary['total_loop'] = summary['total_eval'] + summary['total_dump']
    summary['avg_batch_size'] = _safe_div(num_samples, num_batches)
    summary['throughput'] = _safe_div(num_samples, summary['total_eval'])
    summary['model_throughput'] = _safe_div(num_samples, summary['total_model'])
    summary['fps'] = summary['model_throughput']
    summary['throughput_with_dump'] = _safe_div(num_samples, summary['total_loop'])
    summary['avg_batch_e2e'] = _safe_div(summary['total_eval'], num_batches)
    summary['avg_batch_loop'] = _safe_div(summary['total_loop'], num_batches)
    summary['avg_batch_data'] = _safe_div(summary['total_data'], num_batches)
    summary['avg_batch_model'] = _safe_div(summary['total_model'], num_batches)
    summary['avg_batch_post'] = _safe_div(summary['total_post'], num_batches)
    summary['avg_batch_dump'] = _safe_div(summary['total_dump'], num_batches)
    summary['avg_model_sample_ms'] = None if num_samples == 0 else summary['total_model'] / num_samples * 1000.0
    summary['avg_post_sample_ms'] = None if num_samples == 0 else summary['total_post'] / num_samples * 1000.0
    summary['avg_e2e_sample_ms'] = None if num_samples == 0 else summary['total_eval'] / num_samples * 1000.0
    if model_batch_times.size > 0:
        summary['model_batch_p50_ms'] = float(np.percentile(model_batch_times * 1000.0, 50))
        summary['model_batch_p90_ms'] = float(np.percentile(model_batch_times * 1000.0, 90))
    else:
        summary['model_batch_p50_ms'] = None
        summary['model_batch_p90_ms'] = None
    return summary


def merge_efficiency_stats(dst, src):
    if dst is None:
        merged = init_efficiency_stats('total', 0, src.get('cfg_batch_size'), _NoParamModel(src))
    else:
        merged = dst

    merged['num_batches'] += src['num_batches']
    merged['num_samples'] += src['num_samples']
    merged['total_data'] += src['total_data']
    merged['total_model'] += src['total_model']
    merged['total_post'] += src['total_post']
    merged['total_dump'] += src['total_dump']
    merged['model_batch_times'].extend(src.get('model_batch_times', []))
    merged['params'] = src.get('params', merged.get('params'))
    merged['trainable_params'] = src.get('trainable_params', merged.get('trainable_params'))
    if merged.get('cfg_batch_size') != src.get('cfg_batch_size'):
        merged['cfg_batch_size'] = None

    for key in ['flops', 'macs']:
        src_value = src.get(key)
        if src_value is None or src.get('num_samples', 0) == 0:
            continue
        old_weight = merged.get(f'_{key}_weight', 0)
        old_value = merged.get(key)
        new_weight = old_weight + src['num_samples']
        if old_value is None or old_weight == 0:
            merged[key] = src_value
        else:
            merged[key] = (old_value * old_weight + src_value * src['num_samples']) / new_weight
        merged[f'_{key}_weight'] = new_weight

    if merged.get('input_shape') is None:
        merged['input_shape'] = src.get('input_shape')
    elif merged.get('input_shape') != src.get('input_shape'):
        merged['input_shape'] = 'various'
    if merged.get('profile_method') is None:
        merged['profile_method'] = src.get('profile_method')
    elif merged.get('profile_method') != src.get('profile_method'):
        merged['profile_method'] = 'mixed'
    if merged.get('profile_error') is None:
        merged['profile_error'] = src.get('profile_error')
    return merged


class _NoParamModel:
    def __init__(self, stats):
        self._params = int(stats.get('params', 0) or 0)
        self._trainable_params = int(stats.get('trainable_params', 0) or 0)

    def parameters(self):
        return []


def _format_number(value, scale=1.0, suffix='', precision=2):
    if value is None:
        return 'N/A'
    return f'{float(value) / scale:.{precision}f}{suffix}'


def format_efficiency_line(tag, summary):
    fps = _format_number(summary.get('fps'), precision=2)
    fps_e2e = _format_number(summary.get('throughput'), precision=2)
    e2e_ms = _format_number(summary.get('avg_e2e_sample_ms'), suffix=' ms', precision=3)
    model_ms = _format_number(summary.get('avg_model_sample_ms'), suffix=' ms', precision=3)
    post_ms = _format_number(summary.get('avg_post_sample_ms'), suffix=' ms', precision=3)
    p50 = _format_number(summary.get('model_batch_p50_ms'), suffix=' ms', precision=3)
    p90 = _format_number(summary.get('model_batch_p90_ms'), suffix=' ms', precision=3)
    flops = _format_number(summary.get('flops'), scale=1e9, suffix=' GFLOPs', precision=3)
    macs = _format_number(summary.get('macs'), scale=1e9, suffix=' GMACs', precision=3)
    params = _format_number(summary.get('params'), scale=1e6, suffix=' M', precision=3)
    trainable = _format_number(summary.get('trainable_params'), scale=1e6, suffix=' M', precision=3)
    return (
        f"[EFF-SUMMARY][{tag}] samples={summary.get('num_samples')}, "
        f"FPS={fps}, FPS_e2e={fps_e2e}, "
        f"e2e={e2e_ms}/line, model={model_ms}/line, post={post_ms}/line, "
        f"model_batch_p50={p50}, model_batch_p90={p90}, params={params}, trainable={trainable}, "
        f"flops={flops}, macs={macs}, input_shape={summary.get('input_shape')}, "
        f"profile={summary.get('profile_method') or 'N/A'}"
    )


def print_efficiency_summary(summary):
    dataset_name = summary['dataset_name']
    avg_bs = _format_number(summary.get('avg_batch_size'), precision=2)
    cfg_bs = 'N/A' if summary.get('cfg_batch_size') is None else str(summary['cfg_batch_size'])
    print(f"[EFF][{dataset_name}] samples={summary['num_samples']}, batches={summary['num_batches']}, cfg_bs={cfg_bs}, avg_bs={avg_bs}")
    print(
        f"[EFF][{dataset_name}] total_eval={summary['total_eval']:.4f}s, "
        f"total_model={summary['total_model']:.4f}s, total_post={summary['total_post']:.4f}s, "
        f"total_data_prep={summary['total_data']:.4f}s, total_dump={summary['total_dump']:.4f}s"
    )
    print(format_efficiency_line(dataset_name, summary))
    if summary.get('profile_error'):
        print(f"[EFF][{dataset_name}] profile_warning={summary['profile_error']}")
