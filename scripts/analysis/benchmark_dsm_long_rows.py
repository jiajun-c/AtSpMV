#!/usr/bin/env python3
"""Reproduce single-SpMV FP64 long-row DSM comparisons on a free Hopper GPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[202610301, 202610302, 202610303])
    parser.add_argument('--shape', action='append', default=[], help='ROWS,LENGTH or mixed:ROWS,LENGTH')
    parser.add_argument('--matrix', type=Path, action='append', default=[])
    args = parser.parse_args()
    if not args.shape and not args.matrix:
        parser.error('at least one --shape or --matrix is required')
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    cases = [(s.replace(',', '-').replace(':', '-'), s.split(',')) for s in args.shape]
    cases += [(p.stem, [str(p.resolve()), '0']) for p in args.matrix]
    if len({name for name, _ in cases}) != len(cases):
        parser.error('case names must be unique')
    gpu_command = ['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used', '--format=csv']
    metadata = {
        'sources': {p: sha(root / p) for p in ['include/dsmLongRow.cuh', 'bench/dsm_long_row_bench.cu', 'scripts/analysis/benchmark_dsm_long_rows.py']},
        'executable': str(args.exe.resolve()), 'executable_sha256': sha(args.exe),
        'inputs': {str(p.resolve()): sha(p) for p in args.matrix},
        'quack_reference': '77a750c0d9548d4a6e23c0f64062d61b3ea420ff',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_state_before': subprocess.check_output(gpu_command, text=True),
        'seeds': args.seeds, 'cases': cases,
        'timing': '8 independent beta=0 SpMV calls per CUDA graph; 3 warmup and 15 shuffled interleaved measurement batches; per-call GPU event time. Includes output clear, DSM initialization/communication, and second-stage kernels. No cache flush. Excludes allocation, row partition/sort/upload, AtBSR packing, cuSPARSE preprocessing, and graph setup.',
        'validation': 'Every candidate vs CPU long-double reference, two nonuniform X vectors, poisoned output before direct and graph checks. Graph replay includes repeated overwrite.',
        'selection': 'Sweep threads 128/256/512/1024, rows/group 1/4/8, split 2/4/8 (non-DSM additionally16/32), long-row cutoffs 0/1024/4096; deduplicate identical row partitions. Skip cutoff0 for >16384 rows. AtSpmv sweeps all six shapes and both balancing modes. Report per-kind best median across seeds; not an online selector or unbiased holdout.',
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    records = []
    summary = []
    keys = ['kind', 'threads', 'parts', 'rows_per_group', 'threshold', 'tile', 'balanced']
    for name, parameters in cases:
        samples = []
        for seed in args.seeds:
            filename = f'{name}-{seed}.csv'
            command = [str(args.exe.resolve()), *parameters, str(seed)]
            with (args.output / filename).open('w') as output:
                subprocess.run(command, stdout=output, check=True)
            rows = list(csv.DictReader((args.output / filename).open()))
            if not rows:
                raise RuntimeError(f'empty result: {filename}')
            samples += rows
            records.append({'case': name, 'seed': seed, 'command': command, 'csv': filename, 'candidates': len(rows), 'max_abs_error': max(float(r['max_abs_error']) for r in rows)})
            (args.output / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        groups = {}
        for row in samples:
            key = tuple(row[k] for k in keys)
            groups.setdefault(key, []).append(row)
        configs = []
        for key, rows in groups.items():
            if len(rows) != len(args.seeds):
                raise RuntimeError(f'inconsistent sweep for {name}: {key}')
            configs.append({**dict(zip(keys, key)), 'median_us': statistics.median(float(r['median_us']) for r in rows), 'per_seed_us': {r['seed']: float(r['median_us']) for r in rows}})
        best = {kind: min((c for c in configs if c['kind'] == kind), key=lambda c: c['median_us']) for kind in {c['kind'] for c in configs}}
        nondsm = min((c for k, c in best.items() if not k.startswith('dsm')), key=lambda c: c['median_us'])
        target = best['dsm_async']
        paired = {str(seed): min(float(r['median_us']) for r in samples if r['seed'] == str(seed) and not r['kind'].startswith('dsm')) / target['per_seed_us'][str(seed)] for seed in args.seeds}
        summary.append({'case': name, 'best': best, 'best_non_dsm_kind': nondsm['kind'], 'speedup': nondsm['median_us'] / target['median_us'], 'fixed_dsm_vs_best_non_dsm_each_seed': paired})
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(name, {k: round(v['median_us'], 3) for k, v in best.items()}, 'speedup', round(summary[-1]['speedup'], 3), flush=True)
    metadata['gpu_state_after'] = subprocess.check_output(gpu_command, text=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
