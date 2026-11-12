#!/usr/bin/env python3
"""Account for all discovered matrices and the H100 model's calibrated scope."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--catalog', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--sweep-index', type=Path, required=True)
p.add_argument('--recovered', type=Path, help='Validated structural records from alternate local copies')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
catalog = read(a.catalog)
features = {r['dataset']: r for r in read(a.features) if r['status'] == 'ok'}
swept = {r['dataset'] for r in read(a.sweep_index)}
recovered = {r['dataset'] for r in read(a.recovered) if r['status'] == 'ok'} if a.recovered else set()
rows = []
for item in catalog:
    name = item['dataset']
    f = features.get(name)
    field = item['header'].lower().split()[3]
    if field == 'complex':
        scope = 'unsupported_complex'
    elif f is None:
        scope = 'unknown_structure'
    elif int(f['nnz']) > 1000000 or int(f['rows']) > 349651 or int(f['columns']) > 1785345:
        scope = 'outside_calibration_scope'
    else:
        scope = 'within_calibration_scope'
    rows.append({'dataset': name, 'scope': scope, 'in_h100_sweep': name in swept,
                 'input_source': 'alternate_local_copy' if name in recovered else 'primary',
                 'rows': f['rows'] if f else item['rows'], 'columns': f['columns'] if f else item['columns'],
                 'exact_nnz': f['nnz'] if f else '', 'header': item['header']})
a.output.mkdir(parents=True, exist_ok=True)
temporary = a.output / 'inventory.csv.tmp'
with temporary.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
os.replace(temporary, a.output / 'inventory.csv')
summary = {'total': len(rows), 'scope_counts': dict(Counter(r['scope'] for r in rows)),
           'sweep_coverage': dict(Counter((r['scope'] + ('/included' if r['in_h100_sweep'] else '/not_included')) for r in rows)),
           'inputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.catalog, a.features, a.sweep_index) + ((a.recovered,) if a.recovered else ())},
           'note': 'Scope uses exact structural statistics when available; unknown is not assumed to be in scope. Hardware and kernel configuration must also match the model.'}
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
