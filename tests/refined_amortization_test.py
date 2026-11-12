#!/usr/bin/env python3
"""Check paired setup-plus-execution accounting and missing data detection."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_refined_amortization.py'
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    (root / 'raw').mkdir(); (root / 'status').mkdir()
    (root / 'manifest.json').write_text(json.dumps([{'dataset': 'a'}]))
    (root / 'metadata.json').write_text(json.dumps({'trials': 2}))
    (root / 'status/a.json').write_text(json.dumps({'status': 'ok'}))
    labels = {'auto_model': (10, 4), 'auto_vector_model': (12, 3),
              'auto_model_refined': (20, 2), 'auto_tuned_mixed': (30, 1)}
    with (root / 'raw/a.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=['order', 'backend', 'us'])
        writer.writeheader()
        for order in ('first', 'second'):
            for label, (_, time) in labels.items():
                for _ in range(2): writer.writerow({'order': order, 'backend': label, 'us': time})
    with (root / 'raw/a.log').open('w') as target:
        for order in ('first', 'second'):
            target.write(f'BENCH_ORDER,{order}\n')
            for label, (setup, _) in labels.items(): target.write(f'PLAN_PREP,{label},{setup}\n')
    subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)
    report = json.loads((root / 'refined-amortization.json').read_text())
    assert math.isclose(report['comparisons']['auto_model']['1']['geomean_speedup'], 14/22)
    assert math.isclose(report['comparisons']['auto_tuned_mixed']['100']['geomean_speedup'], 130/220)
    log = (root / 'raw/a.log').read_text().replace('PLAN_PREP,auto_model_refined,20\n', '')
    (root / 'raw/a.log').write_text(log)
    assert subprocess.run([sys.executable, str(script), str(root)], capture_output=True).returncode != 0
print('refined amortization analysis passed')
