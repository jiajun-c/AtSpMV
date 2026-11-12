#!/usr/bin/env python3
"""Break down measured Plan speedups by matrix size and long-row structure."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--features', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
features = {}
with args.features.open() as f:
    for row in csv.DictReader(f):
        if int(row['threshold']) != 1024:
            continue
        name = row['dataset']
        if name in features:
            raise ValueError(f'duplicate features for {name}')
        features[name] = row
with (args.directory/'analysis.csv').open() as f:
    rows = list(csv.DictReader(f))
groups = collections.defaultdict(list)
missing = []
for row in rows:
    name = row['dataset']
    feature = features.get(name)
    if feature is None:
        missing.append(name)
        continue
    nnz = int(row['nnz']);nrows = int(feature['rows']);longs = int(feature['long_rows'])
    if nnz != int(feature['nnz']) or nrows <= 0 or longs < 0 or longs > nrows:
        raise ValueError(f'inconsistent matrix features for {name}')
    speed = float(row['speedup_vs_extended_baseline'])
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f'invalid speedup for {name}')
    scale = '<10k' if nnz < 10_000 else '10k-100k' if nnz < 100_000 else '100k-1M' if nnz <= 1_000_000 else '>1M'
    fraction = longs/nrows
    structure = 'no rows >=1024' if longs == 0 else '<1% rows >=1024' if fraction < .01 else '1%-10% rows >=1024' if fraction < .1 else '>=10% rows >=1024'
    selected = row['selected']
    for dimension, label in [('nnz', scale), ('long_rows', structure), ('selection', selected)]:
        groups[dimension, label].append((speed, name))
result = {'analyzed_cases': len(rows), 'matched_cases': len(rows)-len(missing), 'missing_features': missing,
          'method': 'Per-case counterbalanced warm Graph speedup against the fastest of six fixed nonhybrid candidates, grouped by matrix structure and selected backend. Sample is an incomplete randomized prefix; groups can be small and post-selection. Setup excluded.',
          'groups': {}}
for (dimension, label), samples in sorted(groups.items()):
    values = sorted(v for v,_ in samples)
    result['groups'].setdefault(dimension, {})[label] = {
        'cases': len(values), 'geomean_speedup': math.exp(statistics.mean(map(math.log, values))),
        'median_speedup': statistics.median(values), 'p10_speedup': values[int((len(values)-1)*.1)],
        'wins': sum(x > 1 for x in values), 'minimum_speedup': values[0],
        'worst_dataset': min(samples)[1]}
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
