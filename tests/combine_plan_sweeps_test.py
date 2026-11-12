#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

repo=Path(__file__).resolve().parents[1];script=repo/'scripts/analysis/combine_plan_sweeps.py'
with tempfile.TemporaryDirectory() as d:
    root=Path(d);small=root/'small';large=root/'large';out=root/'out'
    for part,names in ((small,['a']),(large,['b','c'])):
        (part/'provenance').mkdir(parents=True)
        (part/'provenance/spmv_hybrid_validation').write_bytes(b'test executable')
        with (part/'index.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['dataset','nnz']);w.writeheader();w.writerows({'dataset':n,'nnz':3} for n in names)
    (small/'status').mkdir();(small/'raw').mkdir()
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta={'exe_sha256':h(small/'provenance/spmv_hybrid_validation'),'index_sha256':h(small/'index.csv'),
          'selection':'auto_tuned','counterbalance':True,'extended_baselines':True,'trials':3}
    (small/'metadata.json').write_text(json.dumps(meta));(small/'manifest.json').write_text(json.dumps([{'dataset':'a','nnz':3}]))
    (small/'status/a.json').write_text(json.dumps({'status':'ok'}))
    rows=[]
    for order in ('baseline_first','selection_first'):
        for backend,us in (('atbsr',10),('auto_tuned',8),('cusparse',20),('atbsr_m8',12),('atbsr_m1_balanced',13),('csr_dsm_fixed',14),('csr_two_stage_fixed',15)):
            for trial in range(3):rows.append({'backend':backend,'config':'atbsr/0' if backend=='auto_tuned' else 'x','order':order,'trial':trial,'us':us})
    with (small/'raw/a.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(script),str(small),str(large),'--output',str(out)],check=True,capture_output=True)
    report=json.loads((out/'analysis.json').read_text())
    assert report['status_counts']=={'ok':1,'pending':2} and not report['suite_complete']
    assert report['speedup_vs_extended_baseline']['geomean']==1.25
    with (large/'index.csv').open('a') as f:f.write('a,3\n')
    assert subprocess.run([sys.executable,str(script),str(small),str(large),'--output',str(out)],capture_output=True).returncode!=0
print('combined sweep analysis passed')
