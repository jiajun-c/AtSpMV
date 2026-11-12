#!/usr/bin/env python3
"""Check the optional forward-error filter on both benchmark variants."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_sptrsv_plan.py'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'cases').mkdir()
    (root / 'manifest.json').write_text(json.dumps({'names': ['good', 'bad_default', 'bad_model']}))
    for name in ('good', 'bad_default', 'bad_model'):
        for order in ('forward', 'reverse'):
            de = '1e-2' if name == 'bad_default' else '1e-13'
            me = '1e-2' if name == 'bad_model' else '1e-13'
            stdout = ('mode,backend,rows,solve_us,setup_us,max_error\n'
                      f'default,cusparse,128,10,20,{de}\n'
                      f'model,cached,128,5,21,{me}\n')
            (root / 'cases' / f'{name}-{order}.json').write_text(json.dumps({'returncode': 0, 'stdout': stdout}))
    subprocess.run([sys.executable, str(script), str(root), '--forward-tolerance', '1e-10'],
                   check=True, text=True, capture_output=True)
    with (root / 'analysis-forward.csv').open() as f:
        rows = {row['name']: row['status'] for row in csv.DictReader(f)}
    assert rows == {'good': 'ok', 'bad_default': 'forward_error_exceeds_tolerance',
                    'bad_model': 'forward_error_exceeds_tolerance'}
    summary = json.loads((root / 'summary-forward.json').read_text())
    assert summary['status_counts']['ok'] == 1 and summary['solve_geomean'] == 2.0
