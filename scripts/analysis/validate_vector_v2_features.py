#!/usr/bin/env python3
"""Compare runtime features with those stored in the training dataset."""
import argparse
import csv
from pathlib import Path
import subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--binary', required=True)
p.add_argument('--limit', type=int, default=100)
a = p.parse_args()
with a.dataset.open(newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['width'] == '1']
rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
max_error = 0.0
for row in rows:
    name = row['dataset']
    with (a.matrix_root / name / (name + '.mtx')).open() as f:
        header = f.readline().lower().split()
        line = next(line for line in f if line.strip() and not line.startswith('%'))
        m, n, _ = map(int, line.split())
        counts = [0] * m
        for line in f:
            if not line.strip() or line.startswith('%'):
                continue
            i, j = (int(v) - 1 for v in line.split()[:2])
            counts[i] += 1
            if header[4] != 'general' and i != j:
                counts[j] += 1
    payload = f'{m} {n}\n' + ' '.join(map(str, counts)) + '\n'
    run = subprocess.run([a.binary], input=payload, text=True, capture_output=True, check=True)
    actual = np.array([float(v) for v in run.stdout.split()])
    expected = np.array([float(row[f'x{i}']) for i in range(38)])
    error = np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected)))
    if error >= 1e-12:
        raise RuntimeError(f'{name}: feature mismatch {error}')
    max_error = max(max_error, error)
print(f'{len(rows)} matrices match; maximum relative/scaled error {max_error:.3g}')
