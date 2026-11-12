#!/usr/bin/env python3
"""Check real-matrix host Plan predictions against the frozen Python replay."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--replay',type=Path,required=True)
p.add_argument('--status',type=Path,required=True)
p.add_argument('--matrix-root',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--split',default='holdout')
a=p.parse_args()
with a.replay.open(newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']==a.split]
issues=[];maximum_error=0.
for row in rows:
    name=row['dataset'];matrix=a.matrix_root/name/(name+'.mtx')
    state=json.loads((a.status/(name+'.json')).read_text())
    h=hashlib.sha256()
    with matrix.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=state['matrix_sha256']:
        issues.append(f'{name}: matrix changed');continue
    result=subprocess.run([str(a.binary.resolve()),str(matrix.resolve())],text=True,capture_output=True,check=True)
    values=list(csv.DictReader(io.StringIO(result.stdout)))
    if len(values)!=1 or values[0]['same_selection']!='1':issues.append(f'{name}: selection changed');continue
    expected=float(row['predicted_reference_us']);actual=float(values[0]['reference_us'])
    error=abs(actual/expected-1);maximum_error=max(error,maximum_error)
    if error>=1e-12:issues.append(f'{name}: prediction mismatch {error}')
report={'split':a.split,'cases':len(rows),'max_relative_error':maximum_error,'issues':issues,
        'scope':'Host-only comparison on hash-verified real matrices; GPU integration is tested separately.'}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if issues:raise SystemExit(1)
