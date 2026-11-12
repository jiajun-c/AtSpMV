#!/usr/bin/env python3
"""Evaluate one previously selected vector model on frozen paired test data."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from fit_hybrid_selector import vector

WIDTHS=(1,2,4,8,16,32)
BASELINES=('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse')
def family(name):return re.sub(r'[\d_\-]+$','',name.lower()) or name.lower()
def geo(values):return math.exp(statistics.mean(map(math.log,values)))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
args=parser.parse_args();root=args.directory;frozen=root/'provenance'
selection=json.loads((root/'selection.json').read_text())
training_info=json.loads((frozen/'training-result.json').read_text())
training_path=frozen/'training.csv'
if sha(training_path)!=selection['frozen_training_sha256'] or sha(training_path)!=training_info['training_data_sha256']:
    raise ValueError('frozen training data hash mismatch')
if training_info['selected_on_development']!='extra:1.0':raise ValueError('unexpected preselected model')
if sha(root/'index.csv')!=selection['index_sha256']:raise ValueError('test index hash mismatch')
for field,name in (('features_sha256','features-exact.csv'),('references_sha256','unified-reference.csv'),('training_result_sha256','training-result.json')):
    if field in selection and sha(frozen/name)!=selection[field]:raise ValueError(f'{name} hash mismatch')
training=list(csv.DictReader(training_path.open()))
train_groups={family(r['dataset']) for r in training}
index=list(csv.DictReader((root/'index.csv').open()))
if len({r['dataset'] for r in index})!=len(index) or len({family(r['dataset']) for r in index})!=len(index):
    raise ValueError('test index contains duplicates or related name groups')
if train_groups & {family(r['dataset']) for r in index}:raise ValueError('train/test name family overlap')
prior=frozen/'prior-evaluation-manifest.json'
if prior.exists():
    if sha(prior)!=selection['prior_evaluation_sha256']:raise ValueError('prior evaluation manifest hash mismatch')
    prior_groups={family(r['dataset']) for r in json.loads(prior.read_text())}
    if prior_groups & {family(r['dataset']) for r in index}:raise ValueError('prior evaluation/test name family overlap')
xkeys=sorted((k for k in training[0] if re.fullmatch(r'x\d+',k)),key=lambda k:int(k[1:]))
model=ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1)
model.fit(np.array([[float(r[k]) for k in xkeys] for r in training]),np.log([float(r['ratio']) for r in training]))
features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((frozen/'features-exact.csv').open())}
references=collections.defaultdict(list)
for r in csv.DictReader((frozen/'unified-reference.csv').open()):references[r['dataset']].append(r)
def fallback(name):
    rows=references[name];non=[r for r in rows if int(r['candidate'])<5]
    best=min(non,key=lambda r:float(r['prediction_ratio']));score=float(best['prediction_ratio']) if float(best['prediction_ratio'])<.95 else 1.
    hybrid=[]
    for r in rows:
        i=int(r['candidate'])
        if i<5:continue
        f=features[name,4096 if i==7 else 1024]
        if int(f['long_rows'])*10<int(f['rows']):hybrid.append(r)
    if hybrid:
        best=min(hybrid,key=lambda r:float(r['prediction_ratio']))
        if float(best['prediction_ratio'])<score*.9:score=float(best['prediction_ratio'])
    return score
results=[];status_counts=collections.Counter()
for item in index:
    name=item['dataset'];status_path=root/'status'/f'{name}.json'
    if not status_path.exists():status_counts['pending']+=1;continue
    status=json.loads(status_path.read_text());status_counts[status['status']]+=1
    if status['status']!='ok':continue
    samples=collections.defaultdict(list)
    for r in csv.DictReader((root/'raw'/f'{name}.csv').open()):samples[r['order'],r['backend']].append(float(r['us']))
    orders=('baseline_first','selection_first')
    required=BASELINES+('auto_model',)+tuple(f'csr_vec{w}' for w in WIDTHS)
    if any((o,b) not in samples or not all(math.isfinite(t) and t>0 for t in samples[o,b]) for o in orders for b in required):raise ValueError(f'{name}: missing or invalid measurements')
    times={key:statistics.median(v) for key,v in samples.items()}
    x=vector(features[name,1024],3)
    predictions=np.exp(model.predict(np.array([x+[math.log2(w)] for w in WIDTHS])))
    winner=int(np.argmin(predictions));chosen=bool(predictions[winner]<fallback(name))
    kind=f'csr_vec{WIDTHS[winner]}' if chosen else 'auto_model'
    speed_vs_model=geo(times[o,'auto_model']/times[o,kind] for o in orders)
    speed_vs_fixed=geo(min(times[o,b] for b in BASELINES)/times[o,kind] for o in orders)
    results.append({'dataset':name,'chosen':kind,'predicted_ratio_to_default':float(predictions[winner]),
                    'fallback_predicted_ratio_to_default':fallback(name),'speedup_vs_model':speed_vs_model,
                    'speedup_vs_fixed_oracle':speed_vs_fixed})
report={'method':'Preselected ExtraTrees model trained on frozen tuning-time labels; selects one vector width only if its predicted ratio beats the existing model prediction. Independently measured warm Graph timing in both Plan-construction orders. Limited fixed baseline; setup excluded; approximate family grouping.',
        'manifest_total':len(index),'status_counts':dict(status_counts),'training_families':len(train_groups),'sklearn_version':sklearn.__version__,
        'training_sha256':sha(training_path),'features_sha256':sha(frozen/'features-exact.csv'),'references_sha256':sha(frozen/'unified-reference.csv'),
        'selected_vector':sum(r['chosen']!='auto_model' for r in results),'regressions_vs_model':sum(r['chosen']!='auto_model' and r['speedup_vs_model']<1 for r in results),
        'test_family_disjoint':True,'complete_success':len(results)==len(index),'cases':len(results)}
if results:
    for key in ('speedup_vs_model','speedup_vs_fixed_oracle'):
        report[key]={'geomean':geo(r[key] for r in results),'median':statistics.median(r[key] for r in results),'wins':sum(r[key]>1 for r in results)}
    with (root/'frozen-model-results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
(root/'frozen-model-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
