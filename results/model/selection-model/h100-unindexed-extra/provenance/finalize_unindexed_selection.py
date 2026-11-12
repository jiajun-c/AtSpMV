#!/usr/bin/env python3
"""Prepare an exact-nnz benchmark index from completed unindexed scans."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scan', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with (a.scan/'provenance/input-status.csv').open(newline='') as f:
    names = [r['dataset'] for r in csv.DictReader(f) if r['status'] == 'unverified']
records = []
for name in names:
    path = a.scan/'status'/(name+'.json')
    if not path.exists():
        raise RuntimeError(f'{name}: scan has not completed')
    record = json.loads(path.read_text())
    if record['status'] == 'ok':
        matrix = a.matrix_root/name/(name+'.mtx')
        stat = matrix.stat()
        if stat.st_size != record['file_bytes'] or stat.st_mtime_ns != record['file_mtime_ns']:
            raise RuntimeError(f'{name}: matrix changed after scan')
        f = record['structure']
        if int(f['nnz']) <= 1000000 and int(f['rows']) <= 349651 and int(f['columns']) <= 1785345:
            raise RuntimeError(f'{name}: unexpectedly belongs to model scope')
        records.append(f)
a.output.mkdir(parents=True, exist_ok=True)
fields = ['dataset','rows','columns','nnz','longest_row','longest_row_nnz','ratio','status']
with (a.output/'index.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
(a.output/'scan-summary.json').write_text(json.dumps({'scanned': len(names), 'benchmark_candidates': len(records)}, indent=2)+'\n')
print(f'{len(names)} scans accounted for; {len(records)} indexable matrices')
