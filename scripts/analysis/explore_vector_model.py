#!/usr/bin/env python3
"""Study a vector-vs-existing-model selector on a completed paired sample.

The sample is small and was drawn from previously examined matrices. No
model is exported to production; the report is diagnostic only.
"""
import collections
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from fit_hybrid_selector import OUT, vector

ROOT=OUT/'csr-vector-paired'
def split(name):
    group=re.sub(r'[\d_\-]+$','',name.lower()) or name.lower()
    n=int(hashlib.sha256(('vector-experiment-v1:'+group).encode()).hexdigest()[:8],16)%10
    return 'train' if n<6 else 'dev' if n<8 else 'holdout'

def geo(values):return math.exp(statistics.mean(map(math.log,values)))
def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    statuses={p.stem:json.loads(p.read_text()) for p in (ROOT/'status').glob('*.json')}
    if len(statuses)!=len(manifest) or any(s.get('status')!='ok' for s in statuses.values()):
        raise RuntimeError('requires every matrix in the paired manifest to complete successfully')
    features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features-exact.csv').open())}
    data={}
    for item in manifest:
        name=item['dataset'];times=collections.defaultdict(list);configs={}
        with (ROOT/'raw'/f'{name}.csv').open() as f:
            for r in csv.DictReader(f):
                key=(r['order'],r['backend']);times[key].append(float(r['us']));configs[key]=r['config']
        times={k:statistics.median(v) for k,v in times.items()}
        orders=('baseline_first','selection_first')
        choices=[configs[(o,'auto_model')].split('/')[0] for o in orders]
        if choices[0]!=choices[1]:raise ValueError(f'{name}: differing model selection across construction orders')
        f=features[(name,1024)];x=vector(f,3)
        # The existing model's decision is an input, not a measured time.
        x += [float(choices[0]==k) for k in ('atbsr','csr_dsm','csr_two_stage','cusparse','hybrid_dsm_atbsr')]
        baselines=('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse')
        items=[]
        for width in (1,2,4,8,16,32):
            ratio=geo(times[(o,f'csr_vec{width}')]/times[(o,'auto_model')] for o in orders)
            items.append({'width':width,'x':x+[math.log2(width)],'ratio':ratio})
        baseline=geo(min(times[(o,k)] for k in baselines)/times[(o,'auto_model')] for o in orders)
        data[name]={'split':split(name),'items':items,'baseline_speedup':baseline}
    combinations={};models={};predictions={}
    for family in ('boost','extra','forest'):
        train=[v for d in data.values() if d['split']=='train' for v in d['items']]
        if family=='boost':model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=12,l2_regularization=1,random_state=20261028)
        elif family=='extra':model=ExtraTreesRegressor(n_estimators=120,max_depth=8,min_samples_leaf=4,random_state=20261028,n_jobs=1)
        else:model=RandomForestRegressor(n_estimators=120,max_depth=8,min_samples_leaf=4,random_state=20261028,n_jobs=1)
        model.fit(np.array([v['x'] for v in train]),np.log([v['ratio'] for v in train]));models[family]=model
        for n,d in data.items():
            out=np.exp(model.predict(np.array([v['x'] for v in d['items']])))
            predictions[family,n]=out
        for gate in (.8,.9,.95,1.):
            ratios=[];against=[];changed=bad=0
            for name,d in data.items():
                if d['split']!='dev':continue
                i=int(np.argmin(predictions[family,name]));item=d['items'][i]
                choose=bool(predictions[family,name][i]<gate)
                ratio=item['ratio'] if choose else 1.
                ratios.append(1/ratio);against.append(d['baseline_speedup']/ratio)
                changed+=choose;bad+=choose and ratio>1
            combinations[f'{family}:{gate}']={'cases':len(ratios),'speedup_vs_model':geo(ratios),'speedup_vs_fixed_oracle':geo(against),'changed':changed,'regressions_vs_model':bad}
    best=max(combinations,key=lambda k:combinations[k]['speedup_vs_fixed_oracle'])
    family,gate=best.split(':');gate=float(gate)
    ratios=[];against=[];changed=bad=0
    for name,d in data.items():
        if d['split']!='holdout':continue
        i=int(np.argmin(predictions[family,name]));item=d['items'][i];choose=bool(predictions[family,name][i]<gate)
        ratio=item['ratio'] if choose else 1.
        ratios.append(1/ratio);against.append(d['baseline_speedup']/ratio)
        changed+=choose;bad+=choose and ratio>1
    report={'method':'Historical paired candidate timings on 80 previously examined matrices. Grouped train/dev/holdout split; best family and gate chosen on dev; candidate order counterbalanced. Training is diagnostic and no production weights are exported.',
            'split_counts':{s:sum(d['split']==s for d in data.values()) for s in ('train','dev','holdout')},'development':combinations,'selected_on_development':best,
            'holdout':{'cases':len(ratios),'speedup_vs_model':geo(ratios),'speedup_vs_fixed_oracle':geo(against),'changed':changed,'regressions_vs_model':bad}}
    (OUT/'vector-model-exploratory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
