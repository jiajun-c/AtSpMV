#!/usr/bin/env python3
"""Exercise paired aggregation, fixed oracle, completeness and layout invariants."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_mixed_tile.py'
with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/'raw').mkdir();(root/'status').mkdir()
    (root/'manifest.json').write_text(json.dumps([{'dataset':'a'},{'dataset':'pending'}]))
    (root/'metadata.json').write_text(json.dumps({'trials':2}))
    (root/'status/a.json').write_text(json.dumps({'status':'ok'}))
    fields=['candidate','trial','us','slots','stored_bytes','nnz']+[f'groups_m{m}' for m in (1,2,4,8,16,32)]+['order']
    with (root/'raw/a.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for order in ('0','1'):
            for name in ['mixed','mixed_specialized','mixed_bal4','mixed_bal8','mixed_bal16','default']+[f'm{m}_b{b}' for m in (1,2,4,8,16,32) for b in (0,1)]+[f'm{m}_b1_cap{cap}' for m in (1,2,4,8,16,32) for cap in (4,16)]:
                for trial in (0,1):
                    r=dict(candidate=name,trial=trial,us=1 if name.startswith('mixed') else 2,slots=64 if name.startswith('mixed') else 128,
                           stored_bytes=800 if name.startswith('mixed') else 1200,nnz=60,order=order)
                    r.update({f'groups_m{m}':0 for m in (1,2,4,8,16,32)});w.writerow(r)
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True)
    report=json.loads((root/'analysis.json').read_text())
    assert report['status_counts']=={'ok':1,'pending':1} and not report['complete']
    assert abs(report['speedup_vs_fastest_fixed']['geomean']-2)<1e-12
    assert report['storage_slot_strictly_lower_cases']==1 and report['mixed_faster_cases']==1
    raw=(root/'raw/a.csv').read_text();(root/'raw/a.csv').write_text(raw.replace(',64,800,',',256,800,'))
    assert subprocess.run([sys.executable,str(script),str(root)],capture_output=True).returncode!=0
print('mixed tile analysis passed')
