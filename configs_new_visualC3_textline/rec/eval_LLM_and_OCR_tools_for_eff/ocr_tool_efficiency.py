import math

import numpy as np


def init_runtime_stats(dataset_name):
    return {
        'dataset_name': dataset_name,
        'records': [],
        'num_cached': 0,
        'num_errors': 0,
    }


def add_runtime_record(stats, wall_time_sec, engine_time_sec=None, cached=False, status='ok'):
    if cached:
        stats['num_cached'] += 1
        return
    if status != 'ok':
        stats['num_errors'] += 1
        return
    if wall_time_sec is None:
        return

    stats['records'].append({
        'wall_time_sec': float(wall_time_sec),
        'engine_time_sec': None if engine_time_sec is None else float(engine_time_sec),
    })


def _mean_or_none(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean())


def _percentile_or_none(values, q):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _slice_records(records, warmup=0, max_samples=0):
    start = max(int(warmup), 0)
    sliced = records[start:]
    if max_samples > 0:
        sliced = sliced[:int(max_samples)]
    return sliced


def finalize_runtime_stats(stats, warmup=20, max_samples=100):
    measured_records = _slice_records(stats['records'], warmup=warmup, max_samples=max_samples)
    wall_times = [item['wall_time_sec'] for item in measured_records if item.get('wall_time_sec') is not None]
    engine_times = [item['engine_time_sec'] for item in measured_records if item.get('engine_time_sec') is not None]

    total_runtime_sec = float(sum(wall_times)) if wall_times else None
    throughput = None
    if total_runtime_sec is not None and total_runtime_sec > 0:
        throughput = len(wall_times) / total_runtime_sec

    total_engine_sec = float(sum(engine_times)) if engine_times else None

    return {
        'dataset_name': stats['dataset_name'],
        'num_candidates': len(stats['records']),
        'num_measured': len(wall_times),
        'warmup': max(int(warmup), 0),
        'fixed_first_n_samples': max(int(warmup), 0) + len(measured_records),
        'avg_runtime_sec': _mean_or_none(wall_times),
        'p50_runtime_sec': _percentile_or_none(wall_times, 50),
        'p90_runtime_sec': _percentile_or_none(wall_times, 90),
        'total_runtime_sec': total_runtime_sec,
        'throughput_lines_per_sec': throughput,
        'avg_engine_latency_sec': _mean_or_none(engine_times),
        'total_engine_latency_sec': total_engine_sec,
        'num_engine_measured': len(engine_times),
        'num_cached': int(stats['num_cached']),
        'num_errors': int(stats['num_errors']),
    }


def aggregate_runtime_summaries(summaries, dataset_name='total'):
    valid = [item for item in summaries if item and item.get('num_measured', 0) > 0]
    if not valid:
        return {
            'dataset_name': dataset_name,
            'num_candidates': 0,
            'num_measured': 0,
            'warmup': None,
            'fixed_first_n_samples': 0,
            'avg_runtime_sec': None,
            'p50_runtime_sec': None,
            'p90_runtime_sec': None,
            'total_runtime_sec': None,
            'throughput_lines_per_sec': None,
            'avg_engine_latency_sec': None,
            'total_engine_latency_sec': None,
            'num_engine_measured': 0,
            'num_cached': sum(item.get('num_cached', 0) for item in summaries if item),
            'num_errors': sum(item.get('num_errors', 0) for item in summaries if item),
        }

    total_measured = sum(item['num_measured'] for item in valid)
    total_runtime_sec = sum(item['total_runtime_sec'] for item in valid if item.get('total_runtime_sec') is not None)
    avg_runtime_sec = None
    throughput = None
    if total_runtime_sec > 0:
        avg_runtime_sec = total_runtime_sec / total_measured
        throughput = total_measured / total_runtime_sec

    total_engine_measured = sum(item.get('num_engine_measured', 0) for item in valid)
    total_engine_sec = sum(
        item.get('total_engine_latency_sec', 0.0) or 0.0
        for item in valid
    )
    avg_engine_latency_sec = None
    if total_engine_measured > 0:
        avg_engine_latency_sec = total_engine_sec / total_engine_measured

    return {
        'dataset_name': dataset_name,
        'num_candidates': sum(item.get('num_candidates', 0) for item in valid),
        'num_measured': int(total_measured),
        'warmup': valid[0].get('warmup'),
        'fixed_first_n_samples': sum(item.get('fixed_first_n_samples', 0) for item in valid),
        'avg_runtime_sec': avg_runtime_sec,
        'p50_runtime_sec': None,
        'p90_runtime_sec': None,
        'total_runtime_sec': total_runtime_sec if total_runtime_sec > 0 else None,
        'throughput_lines_per_sec': throughput,
        'avg_engine_latency_sec': avg_engine_latency_sec,
        'total_engine_latency_sec': total_engine_sec if total_engine_measured > 0 else None,
        'num_engine_measured': int(total_engine_measured),
        'num_cached': sum(item.get('num_cached', 0) for item in summaries if item),
        'num_errors': sum(item.get('num_errors', 0) for item in summaries if item),
    }


def _fmt_time_sec(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _fmt_rate(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def print_runtime_summary(summary):
    name = summary['dataset_name']
    print(
        f"[EFF][{name}] avg_runtime_per_line={_fmt_time_sec(summary.get('avg_runtime_sec'))} s/line, "
        f"p50={_fmt_time_sec(summary.get('p50_runtime_sec'))} s, "
        f"p90={_fmt_time_sec(summary.get('p90_runtime_sec'))} s, "
        f"throughput={_fmt_rate(summary.get('throughput_lines_per_sec'))} lines/s"
    )
    print(
        f"[EFF][{name}] measured={summary.get('num_measured', 0)}, "
        f"warmup={summary.get('warmup')}, "
        f"fixed_first_n={summary.get('fixed_first_n_samples', 0)}, "
        f"cached={summary.get('num_cached', 0)}, "
        f"errors={summary.get('num_errors', 0)}"
    )
    if summary.get('avg_engine_latency_sec') is not None:
        print(
            f"[EFF][{name}] engine_latency_per_line="
            f"{_fmt_time_sec(summary.get('avg_engine_latency_sec'))} s/line"
        )


def print_runtime_table_summary(summary):
    name = summary['dataset_name']
    runtime = _fmt_time_sec(summary.get('avg_runtime_sec'))
    print(f"[EFF-SUMMARY][{name}] runtime_per_line={runtime} s/line")
