#!/usr/bin/env python3
"""Check a host-only model selector against frozen H100 Plan decisions."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--replay', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.directory / 'manifest.json').read_text())
errors, fallback, checked = [], Counter(), 0
for item in manifest:
    name = item['dataset']
    status = json.loads((a.directory / 'status' / f'{name}.json').read_text())
    if status['status'] != 'ok':
        continue
    matrix = a.matrix_root / name / (name + '.mtx')
    digest = hashlib.sha256()
    with matrix.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != status.get('matrix_sha256'):
        errors.append(f'{name}: matrix content differs from frozen experiment')
        continue
    expected = {}
    with (a.directory / 'raw' / f'{name}.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row['backend'] in ('auto_model', 'auto_vector_model'):
                expected.setdefault(row['backend'], set()).add(row['config'])
    output = subprocess.check_output([str(a.replay.resolve()), str(matrix)], text=True)
    for row in csv.DictReader(output.splitlines()):
        label = 'auto_vector_model' if row['mode'] == 'vector' else 'auto_model'
        versions = expected.get(label, set())
        if len(versions) != 1:
            errors.append(f'{name}/{label}: missing or inconsistent frozen configurations')
            continue
        old = next(iter(versions)).split('/')
        if row['selected'] == '0':
            fallback[label] += 1
            if old[0] != 'atbsr':
                errors.append(f'{name}/{label}: expected AtBSR fallback, found {old[0]}')
        else:
            actual = [row['backend'], row['tile'], row['balanced'], row['parts'], row['threads'],
                      row['rows_per_group'], row['threshold'], 'vw=' + row['width']]
            if old != actual:
                errors.append(f'{name}/{label}: {old} != {actual}')
        checked += 1
report = {'cases': len(manifest), 'decisions_checked': checked, 'default_fallbacks': dict(fallback),
          'mismatches': errors, 'scope': 'Host model decision replay against frozen GPU Plan selections; fallback tile chosen by AtBSR is not checked.'}
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors) or checked != 2 * len(manifest))
