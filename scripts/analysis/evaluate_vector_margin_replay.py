#!/usr/bin/env python3
"""Retrospectively replay vector prediction margins against paired Plan timings."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--replay', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
a = p.parse_args()
root = a.directory
metadata = json.loads((root / 'metadata.json').read_text())
if not (metadata.get('compare_vector_model') and metadata.get('alias_identical_graphs')):
    raise ValueError('requires a paired vector experiment with identical graphs aliased')
manifest = json.loads((root / 'manifest.json').read_text())
margins = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
results = {margin: [] for margin in margins}
for entry in manifest:
    name = entry['dataset']
    status = json.loads((root / 'status' / f'{name}.json').read_text())
    if status['status'] != 'ok':
        raise ValueError(f'{name}: experiment is incomplete')
    matrix = a.matrix_root / name / (name + '.mtx')
    digest = hashlib.sha256()
    with matrix.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != status['matrix_sha256']:
        raise ValueError(f'{name}: input matrix has changed')
    output = subprocess.check_output([str(a.replay.resolve()), str(matrix)], text=True)
    choices = {row['mode']: row for row in csv.DictReader(output.splitlines())}
    base, vector = choices['base'], choices['vector']
    predicted_base = float(base['selected_ratio'])
    predicted_vector = float(vector['best_vector_ratio'])
    if not (math.isfinite(predicted_base) and math.isfinite(predicted_vector) and
            predicted_base > 0 and predicted_vector > 0):
        raise ValueError(f'{name}: invalid model prediction')
    groups = defaultdict(list)
    with (root / 'raw' / f'{name}.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['backend'] in ('auto_model', 'auto_vector_model'):
                groups[(row['order'], row['backend'])].append(float(row['us']))
    orders = sorted({order for order, _ in groups})
    expected = {(order, label) for order in orders for label in ('auto_model', 'auto_vector_model')}
    if len(orders) != 2 or set(groups) != expected or any(len(v) != metadata['trials'] for v in groups.values()):
        raise ValueError(f'{name}: incomplete paired measurements')
    times = {key: statistics.median(v) for key, v in groups.items()}
    vector_speed = geo(times[(order, 'auto_model')] / times[(order, 'auto_vector_model')] for order in orders)
    for margin in margins:
        chosen = vector['backend'] == 'csr_vector' and predicted_vector < predicted_base * (1 - margin)
        results[margin].append((name, vector_speed if chosen else 1.0, chosen))
report = {'cases': len(manifest),
          'scope': 'Retrospective thresholds on previously inspected samples. Reuses measured Plan timings; not an independent experiment or automatic default recommendation.',
          'margins': {}}
for margin, records in results.items():
    values = [speed for _, speed, _ in records]
    report['margins'][str(margin)] = {
        'geomean_speedup_vs_base': geo(values),
        'selected_vector_cases': sum(chosen for _, _, chosen in records),
        'slower_than_base_by_5pct_cases': sum(speed < 1 / 1.05 for speed in values),
        'faster_than_base_by_5pct_cases': sum(speed >= 1.05 for speed in values),
    }
(root / 'vector-margin-retrospective.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
