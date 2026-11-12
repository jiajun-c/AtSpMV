#!/usr/bin/env python3
"""Guard bootstrap coverage and exclusion reporting."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/plan_speedup_uncertainty.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'status').mkdir()
    (p/'manifest.json').write_text(json.dumps([{'dataset':'a'},{'dataset':'b'}]))
    (p/'status/a.json').write_text(json.dumps({'status':'ok'}))
    with (p/'analysis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','speedup_vs_default']);w.writeheader();w.writerow({'dataset':'a','speedup_vs_default':2})
    cmd=[sys.executable,str(script),str(p),'--iterations','10']
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    subprocess.run(cmd+['--allow-partial'],check=True,capture_output=True)
    result=json.loads((p/'uncertainty.json').read_text())
    assert not result['complete_success'] and result['status_counts']['pending']==1 and result['metrics']['speedup_vs_default']['ci95']==[2,2]
    (p/'status/b.json').write_text(json.dumps({'status':'error'}))
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads((p/'uncertainty.json').read_text())
    assert not result['complete_success'] and result['failures']==[{'dataset':'b','status':'error'}]
    (p/'analysis.csv').write_text('dataset,speedup_vs_default\na,0\n')
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('plan uncertainty coverage passed')
