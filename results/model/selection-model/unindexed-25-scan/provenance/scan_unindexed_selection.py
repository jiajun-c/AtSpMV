#!/usr/bin/env python3
"""Resumably scan unindexed matrices at low I/O priority, one at a time."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--scanner', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--timeout', type=int, default=7200)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'status').mkdir(exist_ok=True)
with a.input.open(newline='') as f:
    rows = sorted((r for r in csv.DictReader(f) if r['status'] == 'unverified'), key=lambda r: int(r['file_bytes']))
meta = {'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'scanner_sha256': hashlib.sha256(a.scanner.read_bytes()).hexdigest(),
        'timeout_seconds': a.timeout, 'matrix_root': str(a.matrix_root.resolve())}
meta_path = a.output / 'metadata.json'
if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
    raise RuntimeError('scan inputs changed; use a new output directory')
meta_path.write_text(json.dumps(meta, indent=2) + '\n')
for i, row in enumerate(rows, 1):
    name = row['dataset']
    status_path = a.output / 'status' / (name + '.json')
    if status_path.exists():
        continue
    path = a.matrix_root / name / (name + '.mtx')
    stat = path.stat()
    result = {'dataset': name, 'file_bytes': stat.st_size, 'file_mtime_ns': stat.st_mtime_ns}
    list_path = a.output / 'current-list.txt'
    list_path.write_text(name + '\n')
    started = time.monotonic()
    try:
        run = subprocess.run(['nice', '-n', '15', 'ionice', '-c', '2', '-n', '7', str(a.scanner.resolve()),
                              str(list_path.resolve()), str(a.matrix_root.resolve())],
                             capture_output=True, text=True, timeout=a.timeout)
        parsed = list(csv.DictReader(io.StringIO(run.stdout)))
        if run.returncode or len(parsed) != 1 or parsed[0]['dataset'] != name:
            result.update(status='scanner_error', returncode=run.returncode, stderr=run.stderr[-2000:])
        else:
            result.update(structure=parsed[0], status=parsed[0]['status'])
            if result['status'] == 'ok' and int(parsed[0]['nnz']) > 2147483647:
                result['status'] = 'unsupported_int32_nnz'
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
    result['elapsed_s'] = round(time.monotonic() - started, 2)
    tmp = status_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(tmp, status_path)
    print(f'{i}/{len(rows)} {name}: {result["status"]} ({result["elapsed_s"]} s)', flush=True)
print('scan completed', flush=True)
