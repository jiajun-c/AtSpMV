#!/usr/bin/env python3
"""Compare model families on the training/development split only.

Do not use the holdout here. Metrics use historical fixed-order timings and
are exploratory because those timings have known construction-order bias.
"""
import collections
import csv
import json
import math
import statistics
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fit_hybrid_selector import OUT, FULL, split, vector
from fit_unified_selector import CONFIGS

def main():
    features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features-exact.csv').open())}
    names=sorted({n for n,_ in features if split(n) in ('train','dev')})
    data=collections.defaultdict(list)
    for name in names:
        samples=collections.defaultdict(list)
        for r in csv.DictReader((FULL/'raw'/f'{name}.csv').open()):samples[(r['backend'],r['config'])].append(float(r['us']))
        default=statistics.median(samples['atbsr','-1/-1/8/128/1/1024'])
        for i,(backend,config,tile,threshold) in enumerate(CONFIGS):
            f=features[(name,threshold)]
            if i>=5 and (not int(f['long_rows']) or int(f['long_rows'])*10>=int(f['rows'])):continue
            data[name].append({'id':i,'x':vector(f,tile),'ratio':statistics.median(samples[backend,config])/default})
    report={}
    for family in ('ridge','boost_small','boost_large','forest','extra'):
        pred={}
        for i in range(len(CONFIGS)):
            train=[(name,r) for name,rs in data.items() if split(name)=='train' for r in rs if r['id']==i]
            all_rows=[(name,r) for name,rs in data.items() for r in rs if r['id']==i]
            if family=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=20))
            elif family=='boost_small':model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=12,l2_regularization=1,random_state=20261028)
            elif family=='boost_large':model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=2,random_state=20261028)
            elif family=='forest':model=RandomForestRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1)
            else:model=ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1)
            model.fit(np.array([r['x'] for _,r in train]),np.log([r['ratio'] for _,r in train]))
            values=np.exp(model.predict(np.array([r['x'] for _,r in all_rows])))
            for (name,r),value in zip(all_rows,values):pred[name,r['id']]=float(value)
        for gate in (.95,.97,.99,1.):
            summaries={}
            for subset in ('train','dev'):
                vs_default=[];vs_oracle=[];bad=hybrid=0
                for name,rs in data.items():
                    if split(name)!=subset:continue
                    normal=[r for r in rs if r['id']<5]
                    best=min(normal,key=lambda r:pred[name,r['id']]);chosen=best if pred[name,best['id']]<gate else None
                    estimate=pred[name,chosen['id']] if chosen else 1.
                    mixed=[r for r in rs if r['id']>=5]
                    if mixed:
                        best=min(mixed,key=lambda r:pred[name,r['id']])
                        if pred[name,best['id']]<estimate*.9:chosen=best
                    ratio=chosen['ratio'] if chosen else 1.
                    oracle=min([1.]+[r['ratio'] for r in normal])
                    vs_default.append(1/ratio);vs_oracle.append(oracle/ratio)
                    bad+=ratio>1;hybrid+=bool(chosen and chosen['id']>=5)
                geo=lambda v:math.exp(statistics.mean(map(math.log,v)))
                summaries[subset]={'count':len(vs_default),'geomean_vs_default':geo(vs_default),'geomean_vs_fixed_nonhybrid_oracle':geo(vs_oracle),'slower_than_default':bad,'hybrid_selected':hybrid}
            report[f'{family}:{gate}']=summaries
    (OUT/'candidate-model-dev-exploration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
