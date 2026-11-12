#!/usr/bin/env python3
"""Paired fixed-width versus adaptive packed SpTRSV experiments."""
import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

MODES = ('adaptive', '1', '2', '4', '8', '16', '32')
NAMES = ('grid1_dual', 'bp_1000', 'nos6', 'circuit204', 'str_200')
p = argparse.ArgumentParser()
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--factors', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'manifest.json').write_text(json.dumps({'names': NAMES, 'modes': MODES, 'threads': 512, 'rounds': 2}, indent=2) + '\n')
for name in NAMES:
    for rnd in range(2):
        for mode in (MODES if rnd == 0 else MODES[::-1]):
            dest = a.output / f'{name}-{mode}-{rnd}.json'
            if dest.exists():
                try:
                    if json.loads(dest.read_text()).get('returncode') == 0:
                        continue
                except (OSError, ValueError):
                    pass
            env = os.environ.copy()
            for key in ('ATSPTRSV_FIXED_TILE_K', 'ATSPTRSV_TILE_K2', 'ATSPTRSV_NARROW_K32', 'ATSPTRSV_THREADS_512_1024', 'ATSPTRSV_THREADS_NARROW', 'ATSPTRSV_REVERSE_THREADS'):
                env.pop(key, None)
            env['ATSPTRSV_CACHED_ONLY'] = '1'
            if mode != 'adaptive':
                env['ATSPTRSV_FIXED_TILE_K'] = mode
            result = subprocess.run([str(a.exe), str(a.factors / (name + '-L.mtx')), '--factor'], env=env, capture_output=True, text=True, timeout=120)
            dest.write_text(json.dumps({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
            print(name, rnd, mode, result.returncode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)

rows = []
for name in NAMES:
    for mode in MODES:
        samples = []
        for rnd in range(2):
            record = json.loads((a.output / f'{name}-{mode}-{rnd}.json').read_text())
            variants = {row['variant']: row for row in csv.DictReader(io.StringIO(record['stdout']))}
            stats = re.search(r'tile_shape_stats,slots=(\d+),strict_nnz=(\d+),shared_bytes=(\d+),shared_limit=(\d+)', record['stderr'])
            if not stats or 'cusparse' not in variants:
                raise RuntimeError(f'missing stats or baseline: {name} {mode} {rnd}')
            slots, nnz, shared, limit = map(int, stats.groups())
            baseline = variants['cusparse']
            tile = variants.get('cached-tiles-512')
            for row in (baseline, tile):
                if row and (not math.isfinite(float(row['max_error'])) or float(row['max_error']) > 1e-10 * (0.5 + 100/101) or float(row['backward_error']) > 1e-12):
                    raise RuntimeError(f'wrong result: {name} {mode} {rnd} {row}')
            samples.append((baseline, tile, slots, nnz, shared, limit))
        base = statistics.median(float(s[0]['solve_us']) for s in samples)
        valid = all(s[1] is not None for s in samples)
        times = [float(s[1]['solve_us']) for s in samples] if valid else []
        ratios = [float(s[0]['solve_us']) / float(s[1]['solve_us']) for s in samples] if valid else []
        rows.append({'matrix': name, 'n': samples[0][0]['n'], 'nnz': samples[0][0]['nnz'], 'levels': samples[0][0]['levels'], 'shape': mode, 'status': 'ok' if valid else 'shared_memory_limit', 'cusparse_us': base, 'tile_us': statistics.median(times) if valid else '', 'speedup': math.exp(statistics.mean(map(math.log, ratios))) if valid else '', 'slots': samples[0][2], 'strict_nnz': samples[0][3], 'density': samples[0][3] / samples[0][2] if samples[0][2] else 1., 'shared_bytes': samples[0][4], 'limit_bytes': samples[0][5], 'max_forward_error': max(float(s[1]['max_error']) for s in samples) if valid else '', 'max_backward_error': max(float(s[1]['backward_error']) for s in samples) if valid else ''})
with (a.output / 'analysis.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
