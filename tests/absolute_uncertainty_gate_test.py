#!/usr/bin/env python3
"""A frozen permissive gate must retain predictions and exclude prior cases."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import joblib
from sklearn.ensemble import ExtraTreesRegressor

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary)
    def write(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    rows=[{'dataset':'old','split':'dev','nnz':'1','old_reference_mape':'.1','new_reference_mape':'.2'},
          {'dataset':'new','split':'holdout','nnz':'1','old_reference_mape':'.3','new_reference_mape':'.05'}]
    write(d/'evaluation.csv',rows);write(d/'exclude.csv',[{'dataset':'old'}])
    features=[]
    for name in ('old','new'):
        for threshold in (128,256,1024,4096):
            r={'dataset':name,'threshold':threshold,'rows':1,'cols':1,'nnz':1,'long_rows':0,'long_nnz':0,
               'longest':1,'longest_short':1,'long_length_squared':0}
            r.update({f'short_tiles_{i}':1 for i in range(6)});r.update({f'short_max_tiles_{i}':1 for i in range(6)});features.append(r)
    write(d/'features.csv',features)
    model=ExtraTreesRegressor(n_estimators=2,random_state=1).fit([[0]*38,[1]*38],[0,1]);joblib.dump(model,d/'model.joblib')
    subprocess.run(['python3',str(root/'scripts/analysis/explore_absolute_uncertainty_gate.py'),
        '--evaluation',str(d/'evaluation.csv'),'--exclude-cases',str(d/'exclude.csv'),'--threshold','1000',
        '--features',str(d/'features.csv'),'--model',str(d/'model.joblib'),'--output',str(d/'out')],check=True,capture_output=True)
    report=json.loads((d/'out/report.json').read_text())
    assert report['metrics']['all']['selected_gate']['cases']==1
    assert report['metrics']['all']['selected_gate']['fallbacks']==0
    assert report['metrics']['all']['selected_gate']['mape']==.05
print('absolute uncertainty replay passed')
