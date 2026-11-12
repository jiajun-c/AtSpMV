#!/usr/bin/env python3
"""Explore a vector model using tuning logs and a disjoint paired sample.

This diagnostic never writes production weights. Tuning labels are noisy and
sampled while candidates compete for selection; paired data is only used for
development/holdout evaluation. Matrix-name grouping is approximate.
"""
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from fit_hybrid_selector import OUT, vector

WIDTHS=(1,2,4,8,16,32)
SAMPLE=OUT/'csr-vector-paired'
def family(name):return re.sub(r'[\d_\-]+$','',name.lower()) or name.lower()
def split(name):
    n=int(hashlib.sha256(('vector-log-v1:'+family(name)).encode()).hexdigest()[:8],16)%10
    return 'dev' if n<5 else 'holdout'
def geo(values):return math.exp(statistics.mean(math.log(v) for v in values))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('training',type=Path)
    ap.add_argument('--min-groups',type=int,default=100)
    ap.add_argument('--names-from',type=Path,help='Use training_names from a previously frozen result')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    sample_names=[r['dataset'] for r in json.loads((SAMPLE/'manifest.json').read_text())]
    excluded={family(n) for n in sample_names}
    features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features-exact.csv').open())}
    refs=collections.defaultdict(list)
    for r in csv.DictReader((OUT/'unified-reference.csv').open()):refs[r['dataset']].append(r)
    def fallback_prediction(name):
        rows=refs[name];non=[r for r in rows if int(r['candidate'])<5]
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
    allowed_names=set(json.loads(a.names_from.read_text())['training_names']) if a.names_from else None
    training=[];train_names=[];issues=[]
    for path in sorted((a.training/'status').glob('*.json')):
        status=json.loads(path.read_text());name=status['dataset']
        if status['status']!='ok' or family(name) in excluded or (allowed_names is not None and name not in allowed_names):continue
        if (name,1024) not in features:issues.append(name+': missing features');continue
        records=list(csv.DictReader((a.training/'raw'/f'{name}.csv').open()))
        orders=list(dict.fromkeys(r['order'] for r in records));baseline={o:statistics.median(float(r['us']) for r in records if r['order']==o and r['backend']=='atbsr') for o in orders}
        blocks=[];current=None
        for line in (a.training/'raw'/f'{name}.log').read_text().splitlines():
            if line.startswith('TUNING_BEGIN '):current={}
            elif line.startswith('TUNING,') and current is not None:
                f=next(csv.reader([line]))
                if f[1]=='csr_vector':current[int(f[8])]=float(f[9])
            elif line=='TUNING_END':blocks.append(current);current=None
        if len(orders)!=2 or len(blocks)!=2 or any(x is None or set(x)!=set(WIDTHS) for x in blocks):issues.append(name+': invalid tuning blocks');continue
        x=vector(features[name,1024],3)
        for width in WIDTHS:
            ratio=geo(blocks[i][width]/baseline[o] for i,o in enumerate(orders))
            training.append({'name':name,'width':width,'x':x+[math.log2(width)],'ratio':ratio})
        train_names.append(name)
    groups={family(n) for n in train_names}
    if allowed_names is not None and set(train_names)!=allowed_names:issues.append('frozen training names missing or invalid')
    if issues:raise ValueError('\n'.join(issues))
    if len(groups)<a.min_groups:raise SystemExit(f'need {a.min_groups} training groups; currently {len(groups)}')
    data={}
    for name in sample_names:
        samples=collections.defaultdict(list)
        for r in csv.DictReader((SAMPLE/'raw'/f'{name}.csv').open()):samples[r['order'],r['backend']].append(float(r['us']))
        times={k:statistics.median(v) for k,v in samples.items()};orders=('baseline_first','selection_first')
        baseline=geo(min(times[o,b] for b in ('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse'))/times[o,'auto_model'] for o in orders)
        x=vector(features[name,1024],3)
        candidates=[{'x':x+[math.log2(w)],'ratio':geo(times[o,f'csr_vec{w}']/times[o,'auto_model'] for o in orders)} for w in WIDTHS]
        data[name]={'split':split(name),'fallback_score':fallback_prediction(name),'baseline':baseline,'candidates':candidates}
    experiments={};predictions={}
    for family_name in ('boost','forest','extra'):
        if family_name=='boost':model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=1,random_state=20261028)
        elif family_name=='forest':model=RandomForestRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1)
        else:model=ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1)
        model.fit(np.array([r['x'] for r in training]),np.log([r['ratio'] for r in training]))
        for n,d in data.items():predictions[family_name,n]=np.exp(model.predict(np.array([r['x'] for r in d['candidates']])))
        for gate in (.8,.9,.95,1.):
            speed=[];base=[];changed=bad=0
            for n,d in data.items():
                if d['split']!='dev':continue
                i=int(np.argmin(predictions[family_name,n]));choose=bool(predictions[family_name,n][i]<d['fallback_score']*gate)
                ratio=d['candidates'][i]['ratio'] if choose else 1.;speed.append(1/ratio);base.append(d['baseline']/ratio);changed+=choose;bad+=choose and ratio>1
            experiments[f'{family_name}:{gate}']={'cases':len(speed),'speedup_vs_model':geo(speed),'speedup_vs_fixed_oracle':geo(base),'changed':changed,'regressions':bad}
    best=max(experiments,key=lambda k:experiments[k]['speedup_vs_fixed_oracle']);family_name,gate=best.split(':');gate=float(gate)
    speed=[];base=[];changed=bad=0
    for n,d in data.items():
        if d['split']!='holdout':continue
        i=int(np.argmin(predictions[family_name,n]));choose=bool(predictions[family_name,n][i]<d['fallback_score']*gate)
        ratio=d['candidates'][i]['ratio'] if choose else 1.;speed.append(1/ratio);base.append(d['baseline']/ratio);changed+=choose;bad+=choose and ratio>1
    training_path=a.output.with_suffix('.training.csv')
    with training_path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','width']+[f'x{i}' for i in range(len(training[0]['x']))]+['ratio'])
        for item in training:writer.writerow([item['name'],item['width'],*item['x'],item['ratio']])
    result={'training_matrices':len(train_names),'training_groups':len(groups),'training_names':train_names,
            'training_names_sha256':hashlib.sha256('\n'.join(train_names).encode()).hexdigest(),
            'training_data_sha256':hashlib.sha256(training_path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'evaluation_manifest_sha256':hashlib.sha256((SAMPLE/'manifest.json').read_bytes()).hexdigest(),
            'excluded_eval_families':len(excluded),'development':experiments,'selected_on_development':best,
            'holdout':{'cases':len(speed),'speedup_vs_model':geo(speed),'speedup_vs_fixed_oracle':geo(base),'changed':changed,'regressions':bad},
            'caveat':'Tuning-time labels, not independent times; evaluation sample already examined; suffix grouping is heuristic. No production model is emitted.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
