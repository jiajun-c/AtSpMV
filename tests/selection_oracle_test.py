#!/usr/bin/env python3
"""Check paired selection aggregation, optional candidates and completeness."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_selection_oracle.py'
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    (root / 'raw').mkdir()
    (root / 'status').mkdir()
    (root / 'manifest.json').write_text(json.dumps([{'dataset': 'a', 'nnz': 10}, {'dataset': 'pending', 'nnz': 10}]))
    (root / 'metadata.json').write_text(json.dumps({'trials': 2, 'compare_vector_model': True, 'compare_refined_model': True}))
    (root / 'status/a.json').write_text(json.dumps({'status': 'ok'}))
    times = {'auto_model': 4.0, 'auto_tuned': 2.0, 'auto_tuned_mixed': 2.0,
             'atbsr': 4.0, 'cusparse': 4.0, 'auto_vector_model': 2.1, 'auto_model_refined': 2.0}
    with (root / 'raw/a.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=['order', 'backend', 'trial', 'us', 'config'])
        writer.writeheader()
        for order in ('0', '1'):
            for name, value in times.items():
                for trial in range(2):
                    writer.writerow({'order': order, 'backend': name, 'trial': trial, 'us': value, 'config': name})
    subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)
    report = json.loads((root / 'oracle-analysis.json').read_text())
    assert report['status_counts'] == {'ok': 1, 'pending': 1}
    assert abs(report['geomean_model_over_observed_best'] - 2.0) < 1e-12
    assert report['vector_model_within_5pct_cases'] == 1
    assert report['refined_model_within_5pct_cases'] == 1
    raw = (root / 'raw/a.csv').read_text()
    (root / 'raw/a.csv').write_text(raw.replace('auto_model_refined', 'missing'))
    assert subprocess.run([sys.executable, str(script), str(root)], capture_output=True).returncode != 0
    (root / 'raw/a.csv').write_text(raw)
    (root / 'metadata.json').write_text(json.dumps({'trials': 2, 'compare_vector_model': True, 'compare_refined_model': True, 'vector_candidates':True}))
    with (root / 'raw/a.csv').open('a', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=['order', 'backend', 'trial', 'us', 'config'])
        for order in ('0','1'):
            for width in (1,2,4,8,16,32):
                for trial in range(2):
                    writer.writerow({'order':order,'backend':f'csr_vec{width}','trial':trial,'us':1,'config':str(width)})
    subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)
    report = json.loads((root/'oracle-analysis.json').read_text())
    assert abs(report['geomean_vector_model_over_observed_best']-2.1)<1e-12
print('selection oracle analysis passed')
