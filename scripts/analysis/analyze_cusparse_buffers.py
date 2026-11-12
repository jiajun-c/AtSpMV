#!/usr/bin/env python3
"""Summarize interleaved cuSPARSE timing across identical buffer copies."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('files', nargs='+', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = {'method': 'Median warm Graph time per CSR/X/Y allocation combination; effects are geometric means of paired ratios over the other two factors. Two allocations per factor, one allocation history per process; cannot isolate why addresses matter.', 'runs': []}
for path in args.files:
    groups = collections.defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = tuple(int(row[f'{name}_buffer']) for name in ('matrix', 'x', 'y'))
            value = float(row['us'])
            if any(x not in (0, 1) for x in key) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{path}: invalid timing record')
            groups[key].append(value)
    if len(groups) != 8 or len({len(v) for v in groups.values()}) != 1:
        raise ValueError(f'{path}: incomplete or unbalanced factorial design')
    med = {k: statistics.median(v) for k, v in groups.items()}
    effects = {}
    for index, name in enumerate(('matrix', 'x', 'y')):
        ratios = []
        for key in med:
            if key[index] == 0:
                alternate = list(key);alternate[index] = 1
                ratios.append(med[tuple(alternate)]/med[key])
        effects[name] = math.exp(statistics.mean(map(math.log, ratios)))
    report['runs'].append({'file': str(path), 'trials_per_combination': len(next(iter(groups.values()))),
        'median_us': {''.join(map(str,k)):v for k,v in sorted(med.items())},
        'max_over_min': max(med.values())/min(med.values()), 'buffer_1_over_0': effects})
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
