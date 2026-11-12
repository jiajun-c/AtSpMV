#!/usr/bin/env python3
"""Validate reference-time and all-width error against a tiny fixture."""
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile

import joblib
from sklearn.dummy import DummyRegressor

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary);run=d/'run';(run/'status').mkdir(parents=True);(run/'raw').mkdir();hist=d/'historical';hist.mkdir()
    (run/'manifest.json').write_text('[{"dataset":"matrix","nnz":"1"}]')
    (run/'metadata.json').write_text('{"log_model_predictions":true,"vector_candidates":true,"trials":2}')
    (run/'status/matrix.json').write_text('{"status":"ok","matrix_sha256":"same"}')
    (hist/'matrix.json').write_text('{"matrix_sha256":"same"}')
    def write(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    write(d/'training.csv',[{'dataset':'matrix','split':'holdout','matrix_sha256':'same','x2':'0','x5':'0'}])
    write(d/'splits.csv',[{'dataset':'matrix','split':'holdout'}])
    features=[]
    for threshold in (128,256,1024,4096):
        r={'dataset':'matrix','threshold':threshold,'rows':1,'cols':1,'nnz':1,'long_rows':0,'long_nnz':0,
           'longest':1,'longest_short':1,'long_length_squared':0}
        r.update({f'short_tiles_{i}':1 for i in range(6)});r.update({f'short_max_tiles_{i}':1 for i in range(6)})
        features.append(r)
    write(d/'features.csv',features)
    rows=[];log=[]
    for order in ('baseline_first','selection_first'):
        log.extend([f'BENCH_ORDER,{order}', 'MODEL_REFERENCE,auto_vector_model_v2,1,0'])
        for width in (1,2,4,8,16,32):
            log.append(f'MODEL_PREDICTION,auto_vector_model_v2,csr_vector,-1,-1,8,128,1,1024,{width},1,1')
        for backend in ('atbsr',*[f'csr_vec{w}' for w in (1,2,4,8,16,32)]):
            for trial in range(2):rows.append({'order':order,'backend':backend,'us':2})
    (run/'raw/matrix.log').write_text('\n'.join(log)+'\n');write(run/'raw/matrix.csv',rows)
    model=DummyRegressor(strategy='constant',constant=math.log(2)).fit([[0]*38],[0]);joblib.dump(model,d/'model.joblib')
    cmd=['python3',str(root/'scripts/analysis/evaluate_h100_absolute_reference.py'),'--run',str(run),'--model',str(d/'model.joblib'),
         '--training',str(d/'training.csv'),'--splits',str(d/'splits.csv'),'--features',str(d/'features.csv'),
         '--historical-status',str(hist),'--output',str(d/'out')]
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads((d/'out/reference-evaluation.json').read_text())['holdout']
    assert result['cases']==1 and result['old_reference_mape']==.5 and result['new_reference_mape']==0
    assert result['old_vector_absolute_mape']==.5 and result['new_vector_absolute_mape']==0
    assert result['new_reference_p90_ape']==0 and result['new_reference_over_20pct']==0
    assert json.loads((d/'out/reference-evaluation.json').read_text())['large_short_posthoc']['evaluation_cases']==0
    (hist/'matrix.json').unlink()
    (d/'additional.json').write_text('{"matrix":"same"}')
    subprocess.run(cmd+['--additional-feature-hashes',str(d/'additional.json')],check=True,capture_output=True)
    assert json.loads((d/'out/reference-evaluation.json').read_text())['holdout']['cases']==1
print('H100 absolute reference evaluation passed')
