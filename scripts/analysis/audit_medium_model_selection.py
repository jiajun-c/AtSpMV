#!/usr/bin/env python3
"""Compare frozen Python replay decisions with the real host selector."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replay', type=Path, required=True)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--matrix-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--split', default='holdout')
args = parser.parse_args()
with args.replay.open(newline='') as stream:
    rows = [r for r in csv.DictReader(stream) if r['split'] == args.split]
issues = []
for row in rows:
    name = row['dataset']
    matrix = args.matrix_root / name / (name + '.mtx')
    result = subprocess.run([str(args.binary.resolve()), str(matrix), '--medium'],
                            check=True, capture_output=True, text=True)
    modes = {r['mode']: r for r in csv.DictReader(io.StringIO(result.stdout))}
    actual, base = modes['medium'], modes['vector_v2']
    expected_medium = row['selected_medium'].lower() == 'true'
    if expected_medium:
        expected = ['hybrid_dsm_atbsr', int(row['medium_tile']), 0,
                    int(row['medium_parts']), 256, 4, 128, 0]
        ratio = float(row['predicted_medium_ratio'])
    else:
        expected = json.loads(row['existing_config'])
        ratio = float(row['predicted_existing_ratio'])
    if expected is None:
        if actual['selected'] != '0':
            issues.append(f'{name}: expected default, got {actual}')
    else:
        keys = ('tile', 'balanced', 'parts', 'threads', 'rows_per_group', 'threshold', 'width')
        config = [actual['backend']] + [int(actual[k]) for k in keys]
        if actual['selected'] != '1' or config != expected:
            issues.append(f'{name}: expected {expected}, got {config}')
    if abs(float(actual['selected_ratio']) - ratio) > 1e-10:
        issues.append(f'{name}: prediction mismatch')
    if not expected_medium and actual != dict(base, mode='medium'):
        issues.append(f'{name}: unchanged baseline mismatch')
args.output.parent.mkdir(parents=True, exist_ok=True)
report = {'split': args.split, 'cases': len(rows), 'issues': issues,
          'scope': 'Host decision parity against a previously inspected retrospective replay.'}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if issues:
    raise SystemExit(1)
