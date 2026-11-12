#!/usr/bin/env python3
"""Check candidate matching and per-order initialization drift ratios."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory)
    (p/'raw').mkdir()
    (p/'status').mkdir()
    (p/'metadata.json').write_text(json.dumps({'selection': 'auto_tuned', 'extended_baselines': True}))
    (p/'status/example.json').write_text(json.dumps({'dataset': 'example', 'status': 'ok'}))
    (p/'raw/example.log').write_text(
        'BENCH_ORDER,baseline_first\nTUNING_BEGIN selected=atbsr\n'
        'TUNING,atbsr,3,0,8,128,1,1024,0,4\n'
        'TUNING,atbsr,3,1,8,128,1,1024,0,20\n'
        'TUNING,cusparse,-1,-1,8,128,1,1024,0,10\nTUNING_END\n'
        'BENCH_ORDER,selection_first\nTUNING_BEGIN selected=cusparse\n'
        'TUNING,atbsr,3,0,8,128,1,1024,0,5\n'
        'TUNING,cusparse,-1,-1,8,128,1,1024,0,10\nTUNING_END\n')
    records = []
    for order, atbsr, sparse in [('baseline_first', 8, 11), ('selection_first', 5, 12)]:
        for backend, value in [('atbsr_m8', atbsr), ('cusparse', sparse)]:
            for trial in range(3):
                records.append({'order': order, 'backend': backend, 'us': value})
    with (p/'raw/example.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    subprocess.run([sys.executable, str(root/'scripts/analysis/analyze_tuning_drift.py'), str(p)], check=True, capture_output=True)
    report = json.loads((p/'tuning-drift.json').read_text())
    assert report['records'] == 4
    assert report['candidates']['atbsr_m8']['observations'] == 2
    assert abs(report['candidates']['atbsr_m8']['geomean_ratio']-2**.5) < 1e-12
    assert abs(report['candidates']['cusparse']['median_ratio']-1.15) < 1e-12
    assert report['candidates']['cusparse']['above_1_10'] == 1
print('tuning drift analysis passed')
