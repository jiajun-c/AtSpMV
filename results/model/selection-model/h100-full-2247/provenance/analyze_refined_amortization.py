#!/usr/bin/env python3
"""Estimate paired Plan setup plus repeated warm-Graph SpMV costs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def geo(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
root = a.directory
manifest = json.loads((root / 'manifest.json').read_text())
metadata = json.loads((root / 'metadata.json').read_text())
labels = ('auto_model', 'auto_vector_model', 'auto_model_refined', 'auto_tuned_mixed')
counts, cases = Counter(), []
for item in manifest:
    name = item['dataset']
    status_path = root / 'status' / f'{name}.json'
    if not status_path.exists():
        counts['pending'] += 1
        continue
    status = json.loads(status_path.read_text())['status']
    counts[status] += 1
    if status != 'ok':
        continue
    samples = defaultdict(list)
    with (root / 'raw' / f'{name}.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['backend'] in labels:
                samples[(row['order'], row['backend'])].append(float(row['us']))
    prep, order = {}, None
    for line in (root / 'raw' / f'{name}.log').read_text().splitlines():
        if line.startswith('BENCH_ORDER,'):
            order = line.split(',', 1)[1]
        elif line.startswith('PLAN_PREP,'):
            _, label, elapsed = next(csv.reader([line]))
            if label in labels:
                if (order, label) in prep:
                    raise ValueError(f'{name}: duplicate preparation time')
                prep[(order, label)] = float(elapsed)
    orders = sorted({o for o, _ in samples})
    expected = {(o, label) for o in orders for label in labels}
    if len(orders) != 2 or set(samples) != expected or set(prep) != expected:
        raise ValueError(f'{name}: incomplete paired preparation or timing')
    if any(len(v) != metadata['trials'] or any(not math.isfinite(x) or x <= 0 for x in v) for v in samples.values()):
        raise ValueError(f'{name}: invalid timings')
    if any(not math.isfinite(x) or x <= 0 for x in prep.values()):
        raise ValueError(f'{name}: invalid preparation times')
    times = {key: statistics.median(value) for key, value in samples.items()}
    cases.append((name, orders, prep, times))

report = {'manifest_total': len(manifest), 'status_counts': dict(counts),
          'scope': 'Paired Plan-reported setup plus N * independently measured warm-Graph SpMV; excludes input reading, caller allocation and Graph creation. Construction order, aliasing, shared warmup and timing noise affect estimates.',
          'cases': len(cases), 'comparisons': {}}
for base in ('auto_model', 'auto_vector_model', 'auto_tuned_mixed'):
    item = {}
    for n in (1, 100, 1000, 10000, 100000, 1000000, 10000000):
        speedups = [geo((prep[(o, base)] + n * times[(o, base)]) /
                        (prep[(o, 'auto_model_refined')] + n * times[(o, 'auto_model_refined')])
                        for o in orders) for _, orders, prep, times in cases]
        if speedups:
            item[str(n)] = {'geomean_speedup': geo(speedups),
                            'median_speedup': statistics.median(speedups),
                            'cases_above_one': sum(value > 1 for value in speedups)}
    report['comparisons'][base] = item
(root / 'refined-amortization.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
