#!/usr/bin/env python3
"""Replay an exploratory medium-row model against the existing unified selector.

Hybrid labels are tuning medians relative to an independently timed default
Plan. This is retrospective, and some timing sources differ by candidate.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from train_vector_accuracy import enriched

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', type=Path, required=True)
p.add_argument('--tuning', type=Path, required=True)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--vector-model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with a.dataset.open(newline='') as f:
    rows = list(csv.DictReader(f))
with a.features.open(newline='') as f:
    features = {(r['dataset'], int(r['threshold'])):r for r in csv.DictReader(f)}
references = defaultdict(dict)
with a.reference.open(newline='') as f:
    for r in csv.DictReader(f): references[r['dataset']][int(r['candidate'])] = float(r['prediction_ratio'])
vmodel = joblib.load(a.vector_model)
keys = [('atbsr',3,0,8,128,1,1024,0), ('atbsr',0,1,8,128,1,1024,0),
        ('csr_dsm',-1,-1,8,256,1,1024,0), ('csr_two_stage',-1,-1,8,256,1,1024,0),
        ('cusparse',-1,-1,8,128,1,1024,0), ('hybrid_dsm_atbsr',3,0,8,256,1,1024,0),
        ('hybrid_dsm_atbsr',5,0,8,256,1,1024,0), ('hybrid_dsm_atbsr',5,0,8,128,1,4096,0)]
widths = (1,2,4,8,16,32)
names = sorted({r['dataset'] for r in rows})
cases = {}
for name in names:
    defaults = defaultdict(list)
    with (a.tuning/'raw'/(name+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):
            if r['backend'] == 'atbsr': defaults[r['order']].append(float(r['us']))
    defaults = {k:statistics.median(v) for k,v in defaults.items()}
    active, plan, order, blocks = None, None, None, {}
    for line in (a.tuning/'raw'/(name+'.log')).read_text().splitlines():
        x = next(csv.reader([line]))
        if x[0] == 'BENCH_ORDER': order = x[1]
        elif x[0] == 'PLAN_PREP': plan = x[1]
        elif line.startswith('TUNING_BEGIN'): active = {} if plan == 'auto_tuned' else None
        elif x[0] == 'TUNING' and active is not None:
            key = (x[1], *(int(v) for v in x[2:9]))
            if key in active: raise ValueError(f'{name}: duplicate candidate')
            active[key] = float(x[9])
        elif line == 'TUNING_END' and active is not None:
            blocks[order] = active; active = None
    if set(blocks) != set(defaults) or len(blocks) != 2: raise ValueError(f'{name}: incomplete orders')
    # Match choosePlanModel's fixed-candidate, hybrid and vector gates.
    base_key, base_ratio, hybrid_ratio = None, 1., None
    refs = references[name]
    for i in range(5):
        if refs[i] < .95 and refs[i] < base_ratio: base_key, base_ratio = keys[i], refs[i]
    for i in range(5,8):
        f = features[name,4096 if i == 7 else 1024]
        if 0 < int(f['long_rows'])*10 < int(f['rows']) and i in refs and refs[i] < base_ratio*.9:
            if hybrid_ratio is None or refs[i] < hybrid_ratio:
                hybrid_ratio, base_key = refs[i], keys[i]
    if hybrid_ratio is not None: base_ratio = hybrid_ratio
    vec = np.exp(vmodel.predict(np.array([enriched(features,name,w) for w in widths])))
    vi = int(np.argmin(vec))
    if vec[vi] < base_ratio:
        base_ratio, base_key = float(vec[vi]), ('csr_vector',-1,-1,8,128,1,1024,widths[vi])
    if base_key is not None and any(base_key not in block for block in blocks.values()):
        raise ValueError(f'{name}: predicted baseline candidate missing in tuning logs')
    cases[name] = {'defaults':defaults,'blocks':blocks,'base_key':base_key,'base_prediction':base_ratio}
for row in rows:
    case = cases[row['dataset']]
    key = ('hybrid_dsm_atbsr',int(row['tile']),0,int(row['parts']),256,4,128,0)
    row['key'] = key
    row['ratio_default'] = math.exp(statistics.mean(math.log(block[key]/case['defaults'][o]) for o,block in case['blocks'].items()))
    row['x'] = [float(row[f'x{i}']) for i in range(30)]
training = [r for r in rows if r['split']=='train']
model = ExtraTreesRegressor(n_estimators=100,max_depth=14,min_samples_leaf=3,random_state=20261029,n_jobs=4)
model.fit(np.array([r['x'] for r in training]),np.log([r['ratio_default'] for r in training]))
a.output.mkdir(parents=True,exist_ok=True)
joblib.dump(model,a.output/'exploratory.joblib')
summary = {'scope':'Retrospective replay using the actual unified prediction gates; medium candidates use initialization medians, whereas default AtBSR uses separately timed warm Graph. Non-default baseline candidates also use tuning medians. Timing-source and allocation biases remain; no fresh Plan execution. Approximate family holdout and existing unified model may have data overlap.',
           'source_dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
           'vector_model_sha256':hashlib.sha256(a.vector_model.read_bytes()).hexdigest(),
           'model_sha256':hashlib.sha256((a.output/'exploratory.joblib').read_bytes()).hexdigest(), 'metrics':{}}
results = []
for split in ('train','dev','holdout'):
    subset = [r for r in rows if r['split']==split]
    predictions = np.exp(model.predict(np.array([r['x'] for r in subset])))
    grouped = defaultdict(list)
    for r,pred in zip(subset,predictions): grouped[r['dataset']].append((r,float(pred)))
    ratios, selected, wins, regressions, regrets0, regrets1 = [],0,0,0,[],[]
    for name, candidates in grouped.items():
        case = cases[name]; chosen, estimate = min(candidates,key=lambda x:x[1])
        use = estimate < case['base_prediction']*.9
        speed, regret0, regret1 = [],[],[]
        for order, block in case['blocks'].items():
            old = case['defaults'][order] if case['base_key'] is None else block[case['base_key']]
            new = block[chosen['key']] if use else old
            oracle = min([old]+[block[r['key']] for r,_ in candidates])
            speed.append(old/new); regret0.append(old/oracle); regret1.append(new/oracle)
        geo=lambda v:math.exp(statistics.mean(map(math.log,v)))
        value=geo(speed); ratios.append(value); regrets0.append(geo(regret0)); regrets1.append(geo(regret1))
        selected += int(use); wins += int(value>1.05); regressions += int(value<1/1.05)
        results.append({'dataset':name,'split':split,'selected_medium':use,'speedup_vs_existing':value,
                        'old_over_restricted_oracle':geo(regret0),'new_over_restricted_oracle':geo(regret1),
                        'predicted_medium_ratio':estimate,'predicted_existing_ratio':case['base_prediction'],
                        'existing_config':json.dumps(case['base_key']),
                        'medium_tile':chosen['tile'],'medium_parts':chosen['parts']})
    summary['metrics'][split] = {'matrices':len(grouped),'selected_medium':selected,'geomean_speedup_vs_existing':geo(ratios),
        'faster_5pct':wins,'slower_5pct':regressions,'old_over_restricted_oracle':geo(regrets0),'new_over_restricted_oracle':geo(regrets1),
        'candidate_mape':float(np.mean(np.abs(predictions/np.array([r['ratio_default'] for r in subset])-1)))}
with (a.output/'replay.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
(a.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
