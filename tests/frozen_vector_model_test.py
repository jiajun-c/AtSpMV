#!/usr/bin/env python3
"""Test frozen model evaluation, provenance checks, and family exclusion."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
script=root/'scripts/analysis/evaluate_frozen_vector_model.py'
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory);q=p/'provenance';q.mkdir();(p/'raw').mkdir();(p/'status').mkdir()
    with (q/'training.csv').open('w',newline='') as f:
        fields=['dataset','width']+[f'x{i}' for i in range(15)]+['ratio'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(20):
            r={'dataset':f'train{i}','width':1,'ratio':.5};r.update({f'x{j}':0 for j in range(15)});w.writerow(r)
    digest=hashlib.sha256((q/'training.csv').read_bytes()).hexdigest()
    (q/'training-result.json').write_text(json.dumps({'selected_on_development':'extra:1.0','training_data_sha256':digest}))
    (p/'index.csv').write_text('dataset\ntestcase\nnotdone\n')
    index_sha=hashlib.sha256((p/'index.csv').read_bytes()).hexdigest()
    (p/'selection.json').write_text(json.dumps({'frozen_training_sha256':digest,'index_sha256':index_sha}))
    fields=['dataset','threshold','rows','cols','nnz','long_rows','long_nnz','longest','longest_short','long_length_squared','short_tiles_3','short_max_tiles_3']
    with (q/'features-exact.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(zip(fields,['testcase',1024,1,1,1,0,0,1,1,0,1,1])))
    with (q/'unified-reference.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','candidate','prediction_ratio']);w.writeheader()
        for i in range(5):w.writerow({'dataset':'testcase','candidate':i,'prediction_ratio':1})
    (p/'status/testcase.json').write_text(json.dumps({'status':'ok'}))
    with (p/'raw/testcase.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['order','backend','us']);w.writeheader()
        for order in ('baseline_first','selection_first'):
            for backend in ('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse','auto_model'):
                w.writerow({'order':order,'backend':backend,'us':10})
            for width in (1,2,4,8,16,32):w.writerow({'order':order,'backend':f'csr_vec{width}','us':5})
    subprocess.run([sys.executable,str(script),str(p)],check=True,capture_output=True)
    result=json.loads((p/'frozen-model-evaluation.json').read_text())
    assert result['cases']==1 and not result['complete_success'] and result['status_counts']['pending']==1
    assert result['selected_vector']==1 and abs(result['speedup_vs_model']['geomean']-2)<1e-12
    (p/'index.csv').write_text('dataset\ntrain999\n')
    (p/'selection.json').write_text(json.dumps({'frozen_training_sha256':digest,'index_sha256':hashlib.sha256((p/'index.csv').read_bytes()).hexdigest()}))
    assert subprocess.run([sys.executable,str(script),str(p)],capture_output=True).returncode!=0
print('frozen vector model evaluation passed')
