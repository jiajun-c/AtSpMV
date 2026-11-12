#!/usr/bin/env python3
"""Validate per-order amortized totals and grouping by matrix size."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_amortization_segments.py'
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory);(p/'raw').mkdir()
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_tuned'}))
    (p/'analysis.csv').write_text('dataset,nnz\nexample,100\n')
    (p/'raw/example.log').write_text('BENCH_ORDER,baseline_first\nPLAN_PREP,atbsr,10\nPLAN_PREP,auto_tuned,110\nBENCH_ORDER,selection_first\nPLAN_PREP,atbsr,10\nPLAN_PREP,auto_tuned,130\n')
    rows=[]
    for order in ('baseline_first','selection_first'):
        for backend,value in [('atbsr',5),('auto_tuned',3)]:
            for trial in range(3):rows.append({'order':order,'backend':backend,'us':value})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(script),str(p),'--output',str(p/'out.json')],check=True,capture_output=True)
    group=json.loads((p/'out.json').read_text())['groups']['<10k']
    assert group['cases']==1 and group['median_selection_setup_ms']==.12 and group['median_default_setup_ms']==.01
    assert abs(group['reuse_scenarios']['1']['geomean_speedup']-((15/113)*(15/133))**.5)<1e-12
    assert group['reuse_scenarios']['100']['wins']==1
print('plan amortization segmentation passed')
