#!/usr/bin/env python3
"""Analyze candidate-family coverage in frozen initialization timing logs.

This is a replay of tuning samples, not an independent Plan performance test.
"""
import argparse
from collections import Counter
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
counts, fastest, rows = Counter(), Counter(), []
for entry in manifest:
    name = entry['dataset']
    path = root / 'status' / f'{name}.json'
    if not path.exists():
        counts['pending'] += 1
        continue
    state = json.loads(path.read_text())['status']
    counts[state] += 1
    if state != 'ok':
        continue
    plans, current, block_name, current_plan = [], None, None, None
    for line in (root / 'raw' / f'{name}.log').read_text().splitlines():
        if line.startswith('PLAN_PREP,'):
            current_plan = next(csv.reader([line]))[1]
        elif line.startswith('TUNING_BEGIN'):
            current, block_name = [], current_plan
        elif line.startswith('TUNING,') and current is not None:
            fields = next(csv.reader([line]))
            current.append((fields[1], float(fields[9])))
        elif line == 'TUNING_END' and current is not None:
            if block_name == 'auto_tuned_mixed':
                plans.append(current)
            current = None
    if len(plans) != 2:
        raise ValueError(f'{name}: expected two mixed tuning blocks, got {len(plans)}')
    vector_ratios, mixed_ratios = [], []
    for candidates in plans:
        if any(not math.isfinite(t) or t <= 0 for _, t in candidates):
            raise ValueError(f'{name}: invalid time')
        vector = [t for backend, t in candidates if backend == 'csr_vector']
        mixed = [t for backend, t in candidates if backend == 'mixed_tile']
        if len(vector) != 6 or len(mixed) != 5:
            raise ValueError(f'{name}: missing CSR vector or mixed candidates')
        best_name, best = min(candidates, key=lambda candidate: candidate[1])
        fastest[best_name] += 1
        vector_ratios.append(min(vector) / best)
        mixed_ratios.append(min(vector + mixed) / best)
    rows.append({'dataset': name, 'nnz': entry['nnz'],
                 'vector_over_tuning_best': geo(vector_ratios),
                 'vector_or_mixed_over_tuning_best': geo(mixed_ratios)})
summary = {'manifest_total': len(manifest), 'status_counts': dict(counts),
           'scope': 'Retrospective coverage of initialization medians in two orders; no independent remeasurement, measured margin, or Plan setup costs.',
           'fastest_candidate_backend_per_order': dict(fastest)}
if rows:
    for key in ('vector_over_tuning_best', 'vector_or_mixed_over_tuning_best'):
        values = [row[key] for row in rows]
        summary[key] = {'cases': len(values), 'geomean': geo(values),
                        'within_5pct': sum(value <= 1.05 for value in values),
                        'over_10pct_slower': sum(value > 1.10 for value in values)}
    with (root / 'refinement-coverage.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
(root / 'refinement-coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
