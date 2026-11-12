#!/usr/bin/env python3
"""Fit an experimental H100 default-AtBSR absolute-time reference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--features',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
meta=json.loads((a.run/'metadata.json').read_text())
with a.features.open(newline='') as f:
    features={r['dataset']:r for r in csv.DictReader(f) if r['width']=='1'}
rows=[]
for name,r in sorted(features.items()):
    status_path=a.run/'status'/(name+'.json')
    if not status_path.exists():continue
    status=json.loads(status_path.read_text())
    if status['status']!='ok':continue
    samples={}
    with (a.run/'raw'/(name+'.csv')).open(newline='') as f:
        for record in csv.DictReader(f):
            if record['backend']=='atbsr':samples.setdefault(record['order'],[]).append(float(record['us']))
    if set(samples)!={'baseline_first','selection_first'} or any(len(v)!=meta['trials'] or any(not math.isfinite(x) or x<=0 for x in v) for v in samples.values()):
        raise ValueError(f'{name}: invalid paired default AtBSR timing')
    time_us=math.exp(statistics.mean(math.log(statistics.median(v)) for v in samples.values()))
    rows.append({'dataset':name,'split':r['split'],'matrix_sha256':status['matrix_sha256'],
                 'time_us':time_us,**{f'x{i}':float(r[f'x{i}']) for i in range(38)}})
training=[r for r in rows if r['split']=='train']
if len(training)<100:raise ValueError('insufficient completed training cases')
a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'training-data.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
model=ExtraTreesRegressor(n_estimators=100,max_depth=14,min_samples_leaf=3,random_state=20261029,n_jobs=4)
model.fit(np.array([[r[f'x{i}'] for i in range(38)] for r in training]),np.log([r['time_us'] for r in training]))
joblib.dump(model,a.output/'model.joblib')
report={'method':'Exploratory absolute default-AtBSR warm Graph reference; frozen group split inherited from vector model. Labels are medians per construction order, geometrically combined. No production update. Timings include internal Graph event nodes and may differ with device, allocation, cache and host contention.',
        'source_metadata_sha256':hashlib.sha256((a.run/'metadata.json').read_bytes()).hexdigest(),
        'features_sha256':hashlib.sha256(a.features.read_bytes()).hexdigest(),
        'training_data_sha256':hashlib.sha256((a.output/'training-data.csv').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((a.output/'model.joblib').read_bytes()).hexdigest(),
        'sklearn_version':sklearn.__version__,'metrics':{}}
for split in ('train','dev','holdout'):
    group=[r for r in rows if r['split']==split]
    if not group:continue
    x=np.array([[r[f'x{i}'] for i in range(38)] for r in group]);actual=np.array([r['time_us'] for r in group])
    pred=np.exp(model.predict(x));errors=np.abs(pred/actual-1)
    report['metrics'][split]={'cases':len(group),'mape':float(np.mean(errors)),
        'median_ape':float(np.median(errors)),'over_20pct':int(np.sum(errors>.2)),
        'geomean_predicted_over_actual':float(np.exp(np.mean(np.log(pred/actual))))}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
