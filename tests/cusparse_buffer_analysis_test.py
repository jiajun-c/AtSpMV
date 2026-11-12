#!/usr/bin/env python3
"""Verify paired factorial effects, and reject incomplete experiments."""
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script = Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_cusparse_buffers.py'
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory)
    rows = []
    for matrix, x, y in itertools.product(range(2), repeat=3):
        for trial in range(3):
            rows.append({'matrix_buffer': matrix, 'x_buffer': x, 'y_buffer': y, 'trial': trial,
                         'us': 4*(2**matrix)*(3**x)*(5**y)})
    path = p/'input.csv'
    def write(records):
        with path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(records)
    write(rows)
    cmd=[sys.executable,str(script),str(path),'--output',str(p/'report.json')]
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads((p/'report.json').read_text())['runs'][0]
    assert result['trials_per_combination']==3 and result['max_over_min']==30
    for key, expected in [('matrix',2),('x',3),('y',5)]:
        assert abs(result['buffer_1_over_0'][key]-expected)<1e-12
    write(rows[:-3])
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('cuSPARSE buffer analysis passed')
