#!/usr/bin/env python3
"""Evaluate hybrid/default time-ratio models on a frozen matrix split.

This first study uses the pre-optimization fixed H100 candidate sweep. It
does not train on selected-plan results or evaluate end-to-end setup time.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sklearn

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/model/hybrid-time-model'
FULL=ROOT/'results/model/dsm-atbsr-hybrid/full-suite'
CONFIGS=[('3/0/8/256/1/1024',3,1024),('5/0/8/256/1/1024',5,1024),('5/0/8/128/1/4096',5,4096)]

def group(name):
    # Keep common numbered matrix variants together. This is only a heuristic,
    # not a guarantee of independence across SuiteSparse families.
    return re.sub(r'[\d_\-]+$', '', name.lower()) or name.lower()

def split(name):
    h=int(hashlib.sha256(('hybrid-v1:'+group(name)).encode()).hexdigest()[:8],16)%10
    return 'train' if h<6 else ('dev' if h<8 else 'holdout')

def vector(f,tile):
    rows,cols,nnz=(float(f[k]) for k in ('rows','cols','nnz'))
    long_rows,long_nnz,longest,longest_short,squared=(float(f[k]) for k in ('long_rows','long_nnz','longest','longest_short','long_length_squared'))
    tiles=float(f[f'short_tiles_{tile}']);max_tiles=float(f[f'short_max_tiles_{tile}'])
    return [math.log1p(v) for v in (rows,cols,nnz,long_rows,long_nnz,longest,longest_short,squared,tiles,max_tiles)]+[long_rows/max(rows,1),long_nnz/max(nnz,1),tiles*32/max(nnz,1),float(cols<32767)]

def main():
    features={}
    with (OUT/'features.csv').open() as f:
        for r in csv.DictReader(f):features[(r['dataset'],int(r['threshold']))]=r
    summary={r['dataset']:r for r in csv.DictReader((FULL/'analysis.csv').open())}
    names=sorted({k[0] for k in features})
    dataset=[]
    for name in names:
        if name not in summary:continue
        samples={}
        with (FULL/'raw'/f'{name}.csv').open() as f:
            for r in csv.DictReader(f):
                key=(r['backend'],r['config']);samples.setdefault(key,[]).append(float(r['us']))
        default=statistics.median(samples[('atbsr','-1/-1/8/128/1/1024')])
        for config,tile,threshold in CONFIGS:
            feat=features[(name,threshold)]
            if int(feat['long_rows'])==0:continue
            actual=statistics.median(samples[('hybrid_dsm_atbsr',config)])
            dataset.append({'name':name,'split':split(name),'config':config,'x':vector(feat,tile),'ratio':actual/default})
    experiments={}
    exported=None
    for algorithm in ('ridge','forest','boost'):
        models={}
        predictions={}
        for config,_,_ in CONFIGS:
            train=[r for r in dataset if r['config']==config and r['split']=='train']
            if algorithm=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=20))
            elif algorithm=='forest':model=RandomForestRegressor(n_estimators=100,max_depth=6,min_samples_leaf=8,random_state=20261028,n_jobs=1)
            else:model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=12,l2_regularization=1,random_state=20261028)
            model.fit(np.array([r['x'] for r in train]),np.log([r['ratio'] for r in train]));models[config]=model
            for r in (r for r in dataset if r['config']==config):
                predictions[(r['name'],config)]=float(math.exp(model.predict(np.asarray([r['x']]))[0]))
        if algorithm=='boost':exported=(models,predictions.copy())
        for gate in (0.7,0.8,0.9,0.95):
            metrics={}
            for subset in ('train','dev','holdout'):
                subset_names=[n for n in names if split(n)==subset]
                speeds=[];chosen=bad=0
                for name in subset_names:
                    options=[r for r in dataset if r['name']==name]
                    if options:
                        best=min(options,key=lambda r:predictions[(name,r['config'])])
                        if predictions[(name,best['config'])]<gate:
                            ratio=best['ratio'];chosen+=1;bad+=ratio>1; speeds.append(1/ratio);continue
                    speeds.append(1.)
                metrics[subset]={'matrices':len(speeds),'hybrid_selected':chosen,'hybrid_slower_than_default':bad,
                                 'geomean_speedup':math.exp(statistics.mean(math.log(x) for x in speeds))}
            experiments[f'{algorithm}:{gate}']=metrics
    report={'method':'Predict hybrid/default time ratio from row structure; decision measured against default AtBSR on the same frozen initial sweep. Family-like suffix grouping is heuristic; train/dev/holdout disjoint by group. Absolute timing and end-to-end costs not evaluated.',
            'datasets':len(names),'eligible_candidates':len(dataset),'split_counts':{s:sum(split(n)==s for n in names) for s in ('train','dev','holdout')},
            'experiments':experiments}
    (OUT/'exploratory.json').write_text(json.dumps(report,indent=2)+'\n')
    # The development split favors boost:0.9 (0.95 ties); use the more
    # conservative gate. Keep training restricted to the original train split.
    models,predictions=exported
    header=['// Generated by scripts/analysis/fit_hybrid_selector.py; do not edit.', '#pragma once', '#include <cstddef>', 'namespace atspmv::hybrid_model_data {',
            'struct Node { double value, threshold; int feature, left, right; bool leaf; };',
            'struct Tree { const Node* nodes; };', 'struct Model { double baseline; const Tree* trees; std::size_t count; };']
    for index,(config,_,_) in enumerate(CONFIGS):
        model=models[config]
        for tree,pred in enumerate(model._predictors):
            nodes=pred[0].nodes
            if any(n['is_categorical'] for n in nodes):raise ValueError('categorical tree unsupported')
            header.append(f'inline constexpr Node nodes_{index}_{tree}[] = {{')
            for n in nodes:
                header.append('  {'+f"{float(n['value']):.17g},{float(n['num_threshold']):.17g},{int(n['feature_idx'])},{int(n['left'])},{int(n['right'])},{str(bool(n['is_leaf'])).lower()}"+'},')
            header.append('};')
        header.append(f'inline constexpr Tree trees_{index}[] = {{'+','.join('{nodes_'+str(index)+'_'+str(i)+'}' for i in range(len(model._predictors)))+'};')
    header.append('inline constexpr Model models[] = {')
    for i,(config,_,_) in enumerate(CONFIGS):header.append(f'  {{{float(models[config]._baseline_prediction[0,0]):.17g},trees_{i},sizeof(trees_{i})/sizeof(trees_{i}[0])}},')
    header.extend(['};','} // namespace atspmv::hybrid_model_data',''])
    # Exploratory hybrid-only artifact must not overwrite the production
    # eight-candidate unified calibration.
    (OUT/'hybrid-only-calibration.h').write_text('\n'.join(header))
    with (OUT/'model-reference.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','config','split','prediction_ratio','actual_ratio'])
        for r in dataset:w.writerow([r['name'],r['config'],r['split'],predictions[(r['name'],r['config'])],r['ratio']])
    (OUT/'model-metadata.json').write_text(json.dumps({'algorithm':'HistGradientBoostingRegressor','sklearn_version':sklearn.__version__,'selection_gate':0.9,'configurations':[x[0] for x in CONFIGS],
        'features_sha256':hashlib.sha256((OUT/'features.csv').read_bytes()).hexdigest(), 'manifest_sha256':hashlib.sha256((OUT/'names.txt').read_bytes()).hexdigest(),
        'split':'sha256 of name with trailing digits, underscores and hyphens stripped; 60/20/20 buckets', 'training':'train split only',
        'dev_and_holdout':experiments['boost:0.9'], 'source':'full-suite fixed candidates on H100 PCIe; warm graph; <=1M nnz'},indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
