#!/usr/bin/env python3
"""Audit inference using exact runtime features against rounded training CSV."""
import collections
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from fit_hybrid_selector import OUT, vector
from fit_unified_selector import CONFIGS

if len(sys.argv)!=2:
    raise SystemExit('usage: audit_feature_precision.py CPP_PREDICTOR')
features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features-exact.csv').open())}
records=list(csv.DictReader((OUT/'unified-reference.csv').open()))
lines=[]
for r in records:
    i=int(r['candidate']);_,_,tile,threshold=CONFIGS[i]
    x=vector(features[(r['dataset'],threshold)],tile)
    lines.append(str(i)+' '+' '.join(format(v,'.17g') for v in x))
p=subprocess.run([sys.argv[1]],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
values=[float(x) for x in p.stdout.splitlines()]
if len(values)!=len(records):raise ValueError('prediction count mismatch')
by_name=collections.defaultdict(list)
for r,value in zip(records,values):by_name[r['dataset']].append((r,value))
changed=[]
def choice(name,rows,exact):
    def pred(item):return item[1] if exact else float(item[0]['prediction_ratio'])
    normal=[x for x in rows if int(x[0]['candidate'])<5]
    best=min(normal,key=pred);selected=best if pred(best)<.95 else None;estimate=pred(selected) if selected else 1.
    hybrid=[]
    for x in rows:
        i=int(x[0]['candidate'])
        if i<5:continue
        f=features[name,4096 if i==7 else 1024]
        if int(f['long_rows'])*10<int(f['rows']):hybrid.append(x)
    if hybrid:
        best=min(hybrid,key=pred)
        if pred(best)<estimate*.9:selected=best
    return int(selected[0]['candidate']) if selected else -1
for name,rows in by_name.items():
    a,b=choice(name,rows,False),choice(name,rows,True)
    if a!=b:changed.append({'dataset':name,'rounded_choice':a,'exact_choice':b,'split':rows[0][0]['split']})
max_error=max(abs(math.log(v/float(r['prediction_ratio']))) for r,v in zip(records,values))
report={'predictions':len(values),'matrices':len(by_name),'max_abs_log_ratio_error':max_error,'changed_decisions':changed,
        'note':'Compare runtime-equivalent exact row-length sums against the original six-significant-digit feature CSV, using the frozen trained model and post-hoc guard.'}
(OUT/'feature-precision-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
