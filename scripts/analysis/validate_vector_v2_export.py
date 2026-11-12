#!/usr/bin/env python3
"""Check the frozen sklearn forest and exported C++ forest on real inputs."""
import argparse
import csv
import subprocess
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--binary', required=True)
args = parser.parse_args()
with open(args.dataset, newline='') as f:
    rows = list(csv.DictReader(f))
inputs = np.array([[float(r[f'x{i}']) for i in range(39)] for r in rows])
expected = np.exp(joblib.load(args.model).predict(inputs))
payload = ''.join(' '.join(format(v, '.17g') for v in row) + '\n' for row in inputs)
run = subprocess.run([args.binary], input=payload, text=True, capture_output=True, check=True)
actual = np.array([float(x) for x in run.stdout.splitlines()])
if len(actual) != len(expected):
    raise RuntimeError('prediction count mismatch')
error = np.max(np.abs(actual - expected) / expected)
if error >= 1e-12:
    raise RuntimeError(f'export mismatch: {error}')
print(f'{len(inputs)} predictions match; maximum relative error {error:.3g}')
