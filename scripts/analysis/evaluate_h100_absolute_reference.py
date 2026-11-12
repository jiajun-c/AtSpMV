#!/usr/bin/env python3
"""Evaluate a frozen absolute reference on independently measured H100 Plans."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import joblib
import numpy as np
from train_vector_accuracy import enriched

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--training',type=Path,required=True)
p.add_argument('--splits',type=Path,required=True)
p.add_argument('--features',type=Path,required=True)
p.add_argument('--historical-status',type=Path,required=True)
p.add_argument('--additional-feature-hashes',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.run/'manifest.json').read_text())
meta=json.loads((a.run/'metadata.json').read_text())
if not meta.get('log_model_predictions') or not meta.get('vector_candidates'):
    raise ValueError('requires prediction logs and all vector widths')
with a.training.open(newline='') as f:
    training={r['dataset']:r for r in csv.DictReader(f)}
with a.splits.open(newline='') as f:
    splits={r['dataset']:r['split'] for r in csv.DictReader(f)}
with a.features.open(newline='') as f:
    features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader(f)}
model=joblib.load(a.model)
additional_hashes=json.loads(a.additional_feature_hashes.read_text()) if a.additional_feature_hashes else {}
counts=Counter(); rows=[]
for item in manifest:
    name=item['dataset'];status_path=a.run/'status'/(name+'.json')
    if not status_path.exists():counts['pending']+=1;continue
    state=json.loads(status_path.read_text());counts[state['status']]+=1
    if state['status']!='ok':continue
    historical=a.historical_status/(name+'.json')
    expected_hash=(json.loads(historical.read_text()).get('matrix_sha256') if historical.exists() else additional_hashes.get(name))
    if not expected_hash or expected_hash!=state.get('matrix_sha256'):
        counts['historical_matrix_mismatch_or_missing']+=1;continue
    if name in training and training[name]['matrix_sha256'] != state.get('matrix_sha256'):
        counts['reference_training_matrix_mismatch']+=1;continue
    if any((name,t) not in features for t in (128,256,1024,4096)):
        counts['missing_features']+=1;continue
    samples=defaultdict(list)
    with (a.run/'raw'/(name+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f):samples[r['order'],r['backend']].append(float(r['us']))
    orders=sorted({o for o,_ in samples});times={k:statistics.median(v) for k,v in samples.items()}
    if len(orders)!=2 or any(len(v)!=meta['trials'] for v in samples.values()):raise ValueError(f'{name}: incomplete samples')
    refs={};predictions={};order=None
    for line in (a.run/'raw'/(name+'.log')).read_text().splitlines():
        fields=next(csv.reader([line]))
        if not fields:continue
        if fields[0]=='BENCH_ORDER':order=fields[1]
        elif fields[0]=='MODEL_REFERENCE' and fields[1]=='auto_vector_model_v2':refs[order]=float(fields[2])
        elif fields[0]=='MODEL_PREDICTION' and fields[1]=='auto_vector_model_v2' and fields[2]=='csr_vector':
            predictions[order,int(fields[9])]=(float(fields[10]),float(fields[11]))
    if set(refs)!=set(orders) or set(predictions)!={(o,w) for o in orders for w in (1,2,4,8,16,32)}:
        raise ValueError(f'{name}: incomplete logged predictions')
    pred=float(np.exp(model.predict(np.array([enriched(features,name,1)[:38]]))[0]))
    if not math.isfinite(pred) or pred<=0:raise ValueError(f'{name}: invalid predicted reference')
    old_errors=[];new_errors=[];old_vector=[];new_vector=[]
    for o in orders:
        actual=times[o,'atbsr'];old_errors.append(abs(refs[o]/actual-1));new_errors.append(abs(pred/actual-1))
        for width in (1,2,4,8,16,32):
            ratio,old_us=predictions[o,width];measured=times[o,f'csr_vec{width}']
            old_vector.append(abs(old_us/measured-1));new_vector.append(abs(pred*ratio/measured-1))
    nnz=int(item['nnz'])
    longest=int(features[name,1024]['longest'])
    actual_ref=math.exp(statistics.mean(math.log(times[o,'atbsr']) for o in orders))
    rows.append({'dataset':name,'split':splits.get(name,'not_in_training_table'),
                 'nnz':nnz,'nnz_bucket':'<10k' if nnz<10000 else '10k-100k' if nnz<100000 else '100k-1M' if nnz<=1000000 else '>1M',
                 'longest_row':longest,'large_short_posthoc':nnz>=600000 and longest<32,
                 'used_for_reference_training':name in training and training[name]['split']=='train',
                 'predicted_reference_us':pred,'actual_reference_us':actual_ref,'new_reference_ratio':pred/actual_ref,
                 'old_reference_mape':statistics.mean(old_errors),'new_reference_mape':statistics.mean(new_errors),
                 'old_vector_absolute_mape':statistics.mean(old_vector),'new_vector_absolute_mape':statistics.mean(new_vector)})
result={'status_counts':dict(counts),'total':len(manifest),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'method':'Frozen model trained from H100 device 2; evaluation applies it offline to later H100 device 3 timings and checks matrix hashes. Approximate family split, shared host and previously inspected data limit independence. Event nodes and separate allocations affect timing. This replay does not exercise the opt-in Plan integration.'}
for split in ('all','train','dev','holdout','not_in_training_table'):
    group=rows if split=='all' else [r for r in rows if r['split']==split]
    if not group:continue
    errors=sorted(r['new_reference_mape'] for r in group)
    result[split]={'cases':len(group),**{field:statistics.mean(r[field] for r in group) for field in ('old_reference_mape','new_reference_mape','old_vector_absolute_mape','new_vector_absolute_mape')},
                   'new_reference_median_ape':statistics.median(errors),'new_reference_p90_ape':errors[int(.9*(len(errors)-1))],
                   'new_reference_max_ape':errors[-1],'new_reference_over_20pct':sum(x>.2 for x in errors),
                   'new_reference_worse_than_old':sum(r['new_reference_mape']>r['old_reference_mape'] for r in group),
                   'new_reference_geomean_ratio':math.exp(statistics.mean(math.log(r['new_reference_ratio']) for r in group))}
result['segments']={}
for bucket in sorted({r['nnz_bucket'] for r in rows}):
    group=[r for r in rows if r['nnz_bucket']==bucket]
    result['segments'][bucket]={'cases':len(group),'old_reference_mape':statistics.mean(r['old_reference_mape'] for r in group),
        'new_reference_mape':statistics.mean(r['new_reference_mape'] for r in group),
        'new_reference_over_20pct':sum(r['new_reference_mape']>.2 for r in group)}
large_short=[r for r in rows if r['large_short_posthoc']]
large_short_training=[r for r in training.values() if r['split']=='train' and math.expm1(float(r['x2']))>=600000 and math.expm1(float(r['x5']))<32]
result['large_short_posthoc']={'definition':'nnz >= 600000 and longest row < 32; thresholds chosen after inspecting errors',
    'training_cases':len(large_short_training),'evaluation_cases':len(large_short),
    'old_reference_mape':statistics.mean(r['old_reference_mape'] for r in large_short) if large_short else None,
    'new_reference_mape':statistics.mean(r['new_reference_mape'] for r in large_short) if large_short else None,
    'cases_by_split':dict(Counter(r['split'] for r in large_short))}
result['worst_reference_cases']=[{'dataset':r['dataset'],'split':r['split'],'nnz':r['nnz'],
    'old_reference_mape':r['old_reference_mape'],'new_reference_mape':r['new_reference_mape']} for r in sorted(rows,key=lambda x:-x['new_reference_mape'])[:20]]
a.output.mkdir(parents=True,exist_ok=True)
if rows:
    tmp=a.output/'reference-evaluation.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    os.replace(tmp,a.output/'reference-evaluation.csv')
tmp=a.output/'reference-evaluation.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,a.output/'reference-evaluation.json')
print(json.dumps(result,indent=2))
