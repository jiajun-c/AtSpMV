#!/usr/bin/env python3
"""Check Python's existing-selector replay against the C++ v2 decision."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--replay', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--split', default='holdout')
a = p.parse_args()
with a.replay.open(newline='') as f:
    cases = [r for r in csv.DictReader(f) if r['split'] == a.split]
issues = []
for r in cases:
    name = r['dataset']
    path = a.matrix_root/name/(name+'.mtx')
    proc = subprocess.run([str(a.binary.resolve()),str(path.resolve()),'--v2'],capture_output=True,text=True,check=True)
    modes = {x['mode']:x for x in csv.DictReader(io.StringIO(proc.stdout))}
    actual = modes['vector_v2']
    predicted = json.loads(r['existing_config'])
    if predicted is None:
        if actual['selected'] != '0': issues.append(f'{name}: expected default, got {actual}')
    else:
        config = [actual['backend']]+[int(actual[k]) for k in ('tile','balanced','parts','threads','rows_per_group','threshold','width')]
        if actual['selected'] != '1' or predicted != config:
            issues.append(f'{name}: expected {predicted}, got {config}')
    if abs(float(actual['selected_ratio']) - float(r['predicted_existing_ratio'])) > 1e-10:
        issues.append(f'{name}: ratio mismatch: Python {r["predicted_existing_ratio"]}, C++ {actual["selected_ratio"]}')
summary = {'split':a.split,'cases':len(cases),'issues':issues,
           'scope':'Host selector parity on current matrix files; does not validate GPU timing or model quality.'}
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if issues: raise SystemExit(1)
