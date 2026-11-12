#!/usr/bin/env python3
"""Extract tile-slot density from the mixed-tile benchmark's raw CSVs."""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

SHAPES = (1, 2, 4, 8, 16, 32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path, help='experiment directory containing raw/*.csv')
    args = parser.parse_args()
    rows = []
    for path in sorted((args.directory / 'raw').glob('*.csv')):
        candidates = {}
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                candidates.setdefault(row['candidate'], row)
        mixed = candidates.get('mixed')
        if mixed is None:
            raise ValueError(f'{path}: missing mixed candidate')
        fixed = {}
        for name, row in candidates.items():
            match = re.fullmatch(r'm(1|2|4|8|16|32)_b0', name)
            if match:
                fixed[int(match.group(1))] = int(row['slots'])
        if set(fixed) != set(SHAPES):
            raise ValueError(f'{path}: incomplete fixed-shape candidates')
        nnz, slots = int(mixed['nnz']), int(mixed['slots'])
        if nnz <= 0 or slots <= 0:
            raise ValueError(f'{path}: density undefined for empty matrix')
        best = min(SHAPES, key=fixed.get)
        if slots > fixed[best]:
            raise ValueError(f'{path}: mixed slots exceed best fixed shape')
        rows.append({'dataset': path.stem, 'nnz': nnz, 'mixed_slots': slots,
                     'mixed_density': nnz / slots, 'best_shape_m': best,
                     'best_fixed_slots': fixed[best], 'best_fixed_density': nnz / fixed[best],
                     'density_ratio': fixed[best] / slots,
                     **{f'm{m}_slots': fixed[m] for m in SHAPES}})
    if not rows:
        raise ValueError('no raw CSVs found')
    rows.sort(key=lambda row: row['density_ratio'], reverse=True)
    output = args.directory / 'density.csv'
    with output.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ratios = [row['density_ratio'] for row in rows]
    print(f'cases: {len(rows)}; strictly better: {sum(x > 1 for x in ratios)}')
    print(f'density ratio: geomean={math.exp(statistics.mean(map(math.log, ratios))):.6f}, '
          f'median={statistics.median(ratios):.6f}')
    print(f'mean density: mixed={statistics.mean(r["mixed_density"] for r in rows):.6f}, '
          f'best_fixed={statistics.mean(r["best_fixed_density"] for r in rows):.6f}')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
