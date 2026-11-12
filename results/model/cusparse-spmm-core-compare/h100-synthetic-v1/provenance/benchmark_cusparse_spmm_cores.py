#!/usr/bin/env python3
"""Resume an idle-GPU cuSPARSE CSR/Blocked-ELL comparison."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
cases = [(blocks, density, n, order) for blocks in (2, 8) for density in (5, 25, 100)
         for n in (8, 32, 128) for order in ('forward', 'reverse')]
random.Random(20261102).shuffle(cases)
for i, (blocks, density, n, order) in enumerate(cases, 1):
    dest = a.out / f'b{blocks}-d{density}-n{n}-{order}.json'
    if dest.exists():
        continue
    run = subprocess.run([str(a.exe.resolve()), str(blocks), str(density), str(n), 'all', order],
                         capture_output=True, text=True, timeout=300)
    rows = list(csv.DictReader(io.StringIO(run.stdout))) if run.returncode == 0 else []
    expected = {'csr_alg2', 'csr_alg3', 'blocked_ell'}
    if run.returncode == 0 and ({r['method'] for r in rows} != expected or len(rows) != 3):
        raise ValueError(f'invalid output for {dest.name}: {run.stdout!r}')
    record = {'case': [blocks, density, n, order], 'returncode': run.returncode,
              'stderr': run.stderr, 'rows': rows}
    with tempfile.NamedTemporaryFile('w', dir=a.out, delete=False) as tmp:
        json.dump(record, tmp, indent=2)
        tmp.write('\n')
        name = tmp.name
    os.replace(name, dest)
    print(f'{i}/{len(cases)} {dest.name} exit={run.returncode} '
          f'status={[r["status"] for r in rows]}', flush=True)
    if run.returncode or any(r['status'] == 'incorrect' for r in rows):
        raise RuntimeError(f'execution or correctness failure: {dest}')
print('suite complete', flush=True)
