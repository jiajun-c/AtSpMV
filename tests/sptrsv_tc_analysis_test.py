#!/usr/bin/env python3
"""The paired SOTA analysis must reject incorrect or invalid measurements."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_sptrsv_tc.py'
names = ['good', 'plan_wrong', 'plan_nan', 'plan_zero_time', 'tc_wrong', 'missing']
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'cases').mkdir()
    (root / 'manifest.json').write_text(json.dumps({'names': names}))
    for name in names[:-1]:
        for rnd in range(2):
            error = {'plan_wrong': '0.01', 'plan_nan': 'nan'}.get(name, '1e-13')
            plan_time = '0' if name == 'plan_zero_time' else '5'
            stdout = ('mode,backend,rows,solve_us,max_error\n'
                      f'model-host,cached,128,{plan_time},{error}\n')
            (root / 'cases' / f'{name}-plan-{rnd}.json').write_text(json.dumps({'returncode': 0, 'stdout': stdout}))
            tc_error = '0.01' if name == 'tc_wrong' else '1e-13'
            stdout = f'time_cuda_solve = 0.01 ms\nVALIDATION finite=1 max_error={tc_error} reference_scale=1.49\n'
            (root / 'cases' / f'{name}-tc-{rnd}.json').write_text(json.dumps({'returncode': 0, 'stdout': stdout}))
    (root / 'cases' / 'missing-skip.json').write_text('{}')
    subprocess.run([sys.executable, str(script), str(root), '--aligned'], check=True, capture_output=True, text=True)
    with (root / 'analysis.csv').open() as f:
        rows = {row['name']: row for row in csv.DictReader(f)}
    expected = {'good': 'ok', 'plan_wrong': 'plan_incorrect', 'plan_nan': 'plan_incorrect',
                'plan_zero_time': 'plan_output_invalid', 'tc_wrong': 'tc_incorrect', 'missing': 'factor_missing'}
    assert {name: row['status'] for name, row in rows.items()} == expected
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['valid_pairs'] == 1 and summary['plan_wins'] == 1
    assert summary['geomean_tc_over_plan'] == 2.0
