#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_sptrsv_threads.py'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root/'cases').mkdir()
    (root/'manifest.json').write_text(json.dumps({'names': ['valid', 'bad', 'skipped']}))
    for order in ('forward', 'reverse'):
        stdout = ('variant,solve_us\n'
                  'cached-tiles-128,5\ncached-tiles-256,10\ncached-tiles-512,20\n')
        (root/'cases'/f'valid-{order}.json').write_text(json.dumps({'returncode': 0, 'stdout': stdout}))
        (root/'cases'/f'bad-{order}.json').write_text(json.dumps({'returncode': 1}))
    (root/'cases'/'skipped-skip.json').write_text('{}')
    subprocess.run([sys.executable, str(script), str(root)], check=True, text=True, capture_output=True)
    with (root/'analysis.csv').open() as f: rows = {r['name']: r for r in csv.DictReader(f)}
    assert rows['valid']['status']=='ok' and float(rows['valid']['speedup_128'])==4
    assert rows['bad']['status']=='failed' and rows['skipped']['status']=='factor_missing'
    for order in ('forward', 'reverse'):
        stdout = 'variant,solve_us\ncached-tiles-512,20\ncached-tiles-1024,10\n'
        (root/'cases'/f'valid-{order}.json').write_text(json.dumps({'returncode': 0, 'stdout': stdout}))
    subprocess.run([sys.executable, str(script), str(root), '--include-1024'], check=True, text=True, capture_output=True)
    with (root/'analysis.csv').open() as f: rows = {r['name']: r for r in csv.DictReader(f)}
    assert rows['valid']['status']=='ok' and float(rows['valid']['speedup_1024'])==2
