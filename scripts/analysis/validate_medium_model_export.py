#!/usr/bin/env python3
"""Compare frozen model and exported C++ predictions on all fixture inputs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with a.dataset.open(newline='') as stream:
    x = np.asarray([[float(r[f'x{i}']) for i in range(30)] for r in csv.DictReader(stream)])
expected = np.exp(joblib.load(a.model).predict(x))
data = ''.join(' '.join(format(v, '.17g') for v in row) + '\n' for row in x)
result = subprocess.run([str(a.binary.resolve())], input=data, text=True, capture_output=True, check=True)
actual = np.asarray([float(line) for line in result.stdout.splitlines()])
if actual.shape != expected.shape:
    raise ValueError('prediction count mismatch')
error = float(np.max(np.abs(actual / expected - 1)))
report = {'inputs': len(x), 'max_relative_error': error}
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if error > 1e-12:
    raise SystemExit(1)
