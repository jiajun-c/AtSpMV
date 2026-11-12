#!/usr/bin/env python3
"""Classify obvious input failures without reading very large matrices."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--catalog', type=Path, required=True)
p.add_argument('--small-audit', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
catalog = {r['dataset']: r for r in read(a.catalog)}
scanned = {r['dataset']: r['status'] for r in read(a.small_audit)}
rows = []
for r in read(a.inventory):
    if r['scope'] != 'unknown_structure':
        continue
    name = r['dataset']
    item = catalog[name]
    declared = int(item['nnz'])
    path = a.matrix_root / name / (name + '.mtx')
    if not path.is_file():
        status = 'missing'
        size = ''
    else:
        size = path.stat().st_size
        if name in scanned:
            status = scanned[name]
        elif declared > 2147483647:
            status = 'unsupported_int32_nnz'
        # Even "1 1\n" uses four bytes per coordinate, except possibly the
        # final newline. This is a necessary bound, not a validity check.
        elif size < 4 * declared - 1:
            status = 'definitely_truncated'
        else:
            status = 'unverified'
    rows.append({'dataset': name, 'status': status, 'declared_nnz': declared, 'file_bytes': size})
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
from collections import Counter
print(dict(Counter(r['status'] for r in rows)))
