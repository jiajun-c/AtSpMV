#!/usr/bin/env python3
"""Freeze a structure-stratified matrix sample before observing new timings."""
import argparse
import csv
import json
from pathlib import Path
import random
import re

from benchmark_dsm_atbsr_full import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix-root', type=Path, default=Path('/staff/chengjiajun/workspace/database/suitsparse'))
    parser.add_argument('--inventory', type=Path, default=Path('results/analysis/longest_row_ratio.csv'))
    parser.add_argument('--exclude', type=Path, default=Path('results/model/grouped-columns/h100-prior24/selection.json'))
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026111101)
    args = parser.parse_args()
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('sample already frozen; use a new output directory')
    excluded = {m['dataset'] for m in json.loads(args.exclude.read_text())['matrices']}
    pools = {(size, long): [] for size in range(4) for long in (False, True)}
    for row in csv.DictReader(args.inventory.open()):
        if row['status'] != 'ok' or row['dataset'] in excluded:
            continue
        nnz, rows, columns = (int(row[k]) for k in ('nnz', 'rows', 'columns'))
        if not (1000 <= nnz <= 20000000 and 0 < rows <= 2000000 and 0 < columns <= 2000000):
            continue
        name = row['dataset']; path = args.matrix_root/name/(name+'.mtx')
        if not path.is_file():
            continue
        size = 0 if nnz < 100000 else 1 if nnz < 1000000 else 2 if nnz < 5000000 else 3
        long = int(row['longest_row_nnz']) >= 1024
        pools[size,long].append({**row, 'path': str(path.resolve()), 'size_bin': size, 'has_long_row': long})
    rng = random.Random(args.seed); selected = []; families = {}
    for size in range(4):
        for long, count in ((False,13), (True,12)):
            pool = sorted(pools[size,long], key=lambda r:r['dataset']); rng.shuffle(pool)
            chosen = []
            for row in pool:
                family = re.sub(r'\d+', '', row['dataset']).strip('_-.') or row['dataset']
                if families.get(family,0) >= 3:
                    continue
                with Path(row['path']).open() as f:
                    header = f.readline().lower()
                if 'coordinate' not in header or 'complex' in header:
                    continue
                row['approximate_family'] = family
                families[family] = families.get(family,0)+1; chosen.append(row)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'insufficient eligible families in stratum {size}/{long}: {len(chosen)}')
            selected.extend(chosen)
    rng.shuffle(selected)
    configs = [[kind,-1,-1,8,128,1,1024] for kind in
               ['atbsr_control','atbsr_grouped_model','auto_refined_v2','auto_refined_v2_grouped']]
    for row in selected:
        row['configs'] = configs
    manifest = {'scope':'Predeclared 100-matrix structural sample: 25 per nnz bin (1k-100k,100k-1m,1m-5m,5m-20m); each 13 short/12 long-row cases. Approximate name-family cap 3. Excludes previous24, no performance used for selection. Not a random full-library sample or training holdout.',
                'trials':9,'gpu':args.gpu,'seed':args.seed,
                'inventory_sha256':sha(args.inventory),'exclude_sha256':sha(args.exclude),
                'row_column_limit':2000000,'long_row_threshold':1024,'family_cap':3,'matrices':selected}
    write_json(output/'manifest.json',manifest)
    print(f'Frozen {len(selected)} matrices in {output}/manifest.json')


if __name__ == '__main__':
    main()
