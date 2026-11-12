#!/usr/bin/env python3
"""Compare frozen sklearn and exported C++ absolute reference predictions."""
import argparse
import csv
from pathlib import Path
import subprocess
import joblib
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
a=p.parse_args()
with a.dataset.open(newline='') as f:
    rows=list(csv.DictReader(f))
x=np.array([[float(r[f'x{i}']) for i in range(38)] for r in rows])
expected=np.exp(joblib.load(a.model).predict(x))
payload=''.join(' '.join(format(v,'.17g') for v in row)+'\n' for row in x)
run=subprocess.run([str(a.binary.resolve())],input=payload,text=True,capture_output=True,check=True)
actual=np.array([float(line) for line in run.stdout.splitlines()])
if len(actual)!=len(expected):raise ValueError('prediction count mismatch')
error=np.max(np.abs(actual/expected-1))
if error>=1e-12:raise ValueError(f'export mismatch: {error}')
print(f'{len(x)} predictions match; maximum relative error {error:.3g}')
