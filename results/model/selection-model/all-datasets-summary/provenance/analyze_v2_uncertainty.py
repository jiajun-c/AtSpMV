#!/usr/bin/env python3
"""Bootstrap paired v1/v2 results within approximate matrix families."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def family(name):
    return re.sub(r'[\d_\-]+$', '', name.lower()) or name.lower()


def analyze(rows, iterations, seed):
    result = {}
    for split in ('all', 'train', 'dev', 'holdout', 'not_in_training_table'):
        subset = [r for r in rows if split == 'all' or r['training_split'] == split]
        groups = {}
        for r in subset:
            groups.setdefault(family(r['dataset']), []).append(r)
        if not groups:
            continue
        # Per-group sums and counts retain the per-matrix weighting when families
        # of different sizes are sampled with replacement.
        aggregates = []
        for members in groups.values():
            speed = [float(r['v2_speedup_vs_v1']) for r in members]
            old = [float(r['v1_all_width_ratio_mape']) for r in members]
            new = [float(r['v2_all_width_ratio_mape']) for r in members]
            if not all(math.isfinite(v) and v > 0 for v in speed):
                raise ValueError('invalid speedup')
            if not all(math.isfinite(v) and v >= 0 for v in old + new):
                raise ValueError('invalid prediction error')
            aggregates.append((len(members), sum(map(math.log, speed)),
                               sum(old) - sum(new)))
        values = np.asarray(aggregates, dtype=float)
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(values), size=(iterations, len(values)))
        sampled = values[draws].sum(axis=1)
        speed = np.exp(sampled[:, 1] / sampled[:, 0])
        improvement = sampled[:, 2] / sampled[:, 0]
        total = values.sum(axis=0)
        result[split] = {
            'cases': len(subset), 'families': len(groups),
            'paired_speedup': float(np.exp(total[1] / total[0])),
            'paired_speedup_ci95': np.quantile(speed, [.025, .975]).tolist(),
            'width_ratio_mape_reduction': float(total[2] / total[0]),
            'width_ratio_mape_reduction_ci95': np.quantile(improvement, [.025, .975]).tolist(),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261029)
    a = p.parse_args()
    if a.iterations <= 0:
        p.error('iterations must be positive')
    data = a.input.read_bytes()
    rows = list(csv.DictReader(data.decode('utf-8').splitlines()))
    if not rows or len({r['dataset'] for r in rows}) != len(rows):
        raise ValueError('empty input or duplicate datasets')
    report = {'input': str(a.input), 'input_sha256': hashlib.sha256(data).hexdigest(),
              'iterations': a.iterations, 'seed': a.seed,
              'method': 'Name-based family bootstrap of per-matrix paired results. '
                        'Intervals exclude timing noise, selection of models after exploration, '
                        'family misclassification, and sampling bias. Input may be incomplete.',
              'splits': analyze(rows, a.iterations, a.seed)}
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
