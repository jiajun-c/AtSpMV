#!/usr/bin/env python3
"""Select family-disjoint Matrix Market splits and collect all 12 tile timings."""
import argparse
import concurrent.futures
import csv
import json
import hashlib
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time


def inventory(root):
    # Only primary SuiteSparse matrices, not RHS/solution files.
    for folder in sorted(root.iterdir()):
        path = folder / (folder.name + '.mtx')
        if not path.is_file() or path.stat().st_size > 600_000_000:
            continue
        try:
            with path.open() as f:
                header = next(f).lower()
                if 'coordinate' not in header or 'complex' in header:
                    continue
                dimensions = None
                for line in f:
                    if not line.startswith('%'):
                        dimensions = tuple(map(int, line.split()))
                        break
                if dimensions is None or len(dimensions) != 3:
                    continue
                rows, cols, nnz = dimensions
            if not (1000 <= nnz <= 20_000_000 and rows > 0 and cols > 0):
                continue
            family = re.split(r'[0-9]', folder.name.lower())[0].rstrip('_-') or folder.name
            yield dict(matrix=folder.name, family=family, path=str(path.resolve()),
                       rows=rows, cols=cols, file_nnz=nnz, bytes=path.stat().st_size)
        except (ValueError, StopIteration, OSError):
            continue


def make_manifest(root, output):
    rng = random.Random(20261027)
    items = list(inventory(root))
    rng.shuffle(items)
    unique = {}
    for item in items:
        unique.setdefault(item['family'], item)
    bins = {}
    for item in unique.values():
        bins.setdefault(int(math.log10(item['file_nnz'])), []).append(item)
    chosen = []
    # Round robin across logarithmic nnz strata, preserving family isolation.
    while bins and len(chosen) < 242:
        for key in list(sorted(bins)):
            if bins[key] and len(chosen) < 242:
                chosen.append(bins[key].pop())
            if not bins[key]:
                del bins[key]
    for i, item in enumerate(chosen):
        # Assign within the interleaved strata before any timing is observed.
        item['split'] = ('train', 'train', 'train', 'dev', 'train', 'test')[i % 6]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'seed': 20261027, 'matrices': chosen}, indent=2) + '\n')
    print(f'Selected {len(chosen)} family-disjoint matrices', flush=True)


def collect(manifest, binary, directory, splits, devices, seed, cache_mode, overrides=None):
    items = [m for m in json.loads(manifest.read_text())['matrices'] if m['split'] in splits]
    directory.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(overrides.read_text()) if overrides else {}
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    def worker(device, work):
        for item in work:
            target = directory / (item['matrix'] + '.csv')
            meta = directory / (item['matrix'] + '.json')
            if meta.exists() and json.loads(meta.read_text()).get('status') == 'ok':
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), ATSPMV_BENCH_SEED=str(seed), ATSPMV_BENCH_CACHE=cache_mode,
                       ATSPMV_MODEL_CACHE=cache_mode)
            prior_attempt = json.loads(meta.read_text()) if meta.exists() else None
            benchmark_path = overrides.get(item['matrix'], {}).get('path', item['path'])
            start = time.monotonic()
            try:
                result = subprocess.run([str(binary), benchmark_path, '64'], env=env,
                                        capture_output=True, text=True, timeout=240)
                rows = list(csv.DictReader(result.stdout.splitlines()))
                ok = result.returncode == 0 and len(rows) == 12 and all(r['correct'] == '1' for r in rows)
                target.write_text(result.stdout)
                (directory / (item['matrix'] + '.log')).write_text(result.stderr)
                status = 'ok' if ok else 'failed'
                error = result.stderr[-1000:] if not ok else ''
            except subprocess.TimeoutExpired:
                status, error = 'timeout', 'exceeded 240 seconds'
            meta.write_text(json.dumps(dict(item, status=status, error=error, gpu=device, seed=seed, cache_mode=cache_mode,
                                           binary_sha256=binary_sha256, benchmark_path=benchmark_path, prior_attempt=prior_attempt,
                                           seconds=time.monotonic()-start), indent=2)+'\n')
            print(f"{item['split']} {item['matrix']} {status} {time.monotonic()-start:.1f}s", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, items[i::len(devices)]) for i, device in enumerate(devices)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev'])
    parser.add_argument('--overrides', type=Path, help='Audited replacement paths for damaged input files')
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--cache', choices=['warm','cold'], default='warm')
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    if args.root:
        if args.manifest.exists():
            raise SystemExit('Refusing to overwrite a predeclared split manifest')
        make_manifest(args.root, args.manifest)
    if args.binary:
        collect(args.manifest, args.binary.resolve(), args.output, args.splits, args.devices, args.seed, args.cache, args.overrides)

if __name__ == '__main__':
    main()
