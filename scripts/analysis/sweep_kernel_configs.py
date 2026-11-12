#!/usr/bin/env python3
"""Offline paired kernel experiments; never changes runtime defaults or model fits."""
import argparse
import csv
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--matrix', action='append', default=[])
    parser.add_argument('--manifest', help='Existing matrix manifest; requires --names')
    parser.add_argument('--names', nargs='+', help='Matrix names to read from the manifest')
    parser.add_argument('--configs', help='JSON list of [cap, warps, prefetch, cta_reduce]')
    parser.add_argument('--full', action='store_true', help='all 120 configurations')
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--cache', choices=['warm', 'cold'], default='warm')
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.names):
        parser.error('--manifest and --names must be supplied together')
    if args.manifest:
        matrices = {m['matrix']: m['path'] for m in json.loads(Path(args.manifest).read_text())['matrices']}
        args.matrix.extend(matrices[name] for name in args.names)
    baseline = (8, 4, 0, 0)
    configs = {baseline}
    if args.configs:
        configs.update(map(tuple, json.loads(Path(args.configs).read_text())))
    elif args.full:
        configs.update(itertools.product([4, 8, 16, 32], [2, 4, 8], [0, 1, 2, 4, 8], [0, 1]))
    else:
        configs.update((c, 4, 0, 0) for c in [4, 16, 32])
        configs.update((8, w, 0, 0) for w in [2, 8])
        configs.update((8, 4, p, r) for p in [1, 2, 4, 8] for r in [0, 1])
        configs.add((8, 4, 0, 1))
    jobs = list(itertools.product(args.matrix or ['--synthetic'], sorted(configs)))
    random.Random(args.seed).shuffle(jobs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open('w', newline='') as stream:
        writer = None
        for number, (matrix, config) in enumerate(jobs, 1):
            env = os.environ.copy()
            for key in list(env):
                if key.startswith(('ATSPMV_TUNE_', 'ATSPMV_MODEL_')):
                    del env[key]
            env.update(zip(['ATSPMV_TUNE_CAP', 'ATSPMV_TUNE_WARPS',
                            'ATSPMV_TUNE_PREFETCH', 'ATSPMV_TUNE_CTA_REDUCE'], map(str, config)))
            env['ATSPMV_BENCH_SEED'] = str(args.seed)
            env['ATSPMV_BENCH_CACHE'] = args.cache
            result = subprocess.run([args.binary, matrix, str(args.repeat)], env=env,
                                    text=True, capture_output=True, check=True)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if not rows or any(row['correct'] != '1' for row in rows):
                raise RuntimeError(f'Invalid results: {matrix}, {config}')
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            records.extend(rows)
            print(f'{number}/{len(jobs)} {Path(matrix).name} {config}', flush=True)
    # Compare each configuration at the SAME layout selected by the original
    # model. This does not claim that an oracle choice is available at runtime.
    base = {r['matrix']: r for r in records if configuration(r) == baseline and r['selected'] == '1'}
    summary = []
    for config in sorted(configs):
        by_matrix = {}
        for r in records:
            b = base[r['matrix']]
            if (configuration(r) == config and r['m'] == b['m'] and
                    r['balanced'] == b['balanced'] and int(r['nnz']) > 0):
                by_matrix[r['matrix']] = float(b['measured_us']) / float(r['measured_us'])
        summary.append(dict(config=config, speedups=by_matrix,
                            geomean=math.exp(statistics.mean(map(math.log, by_matrix.values()))),
                            worst=min(by_matrix.values())))
    summary.sort(key=lambda r: -r['geomean'])
    output.with_suffix('.json').write_text(json.dumps(dict(
        comparison='fixed baseline model-selected tile and mode; baseline/tuned time',
        seed=args.seed, cache=args.cache, checks=len(records), configs=summary), indent=2) + '\n')
    for row in summary[:10]:
        print(row)


def configuration(row):
    return tuple(int(row[k]) for k in ['cap', 'warps_per_block', 'prefetch', 'cta_reduce'])


if __name__ == '__main__':
    main()
