#!/usr/bin/env python3
"""Heuristic family-cluster bootstrap for a completed plan benchmark."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--seed',type=int,default=20261028)
p.add_argument('--iterations',type=int,default=5000)
p.add_argument('--allow-partial',action='store_true',help='Explicitly allow pending manifest entries; report the missing coverage')
a=p.parse_args()
if a.iterations<1:raise SystemExit('iterations must be positive')
manifest=json.loads((a.directory/'manifest.json').read_text())
names=[r['dataset'] for r in manifest]
if len(names)!=len(set(names)):raise SystemExit('duplicate manifest dataset')
status_counts=collections.Counter();successful=set();failures=[]
for name in names:
    path=a.directory/'status'/f'{name}.json'
    if not path.exists():status_counts['pending']+=1;continue
    state=json.loads(path.read_text()).get('status');status_counts[state]+=1
    if state=='ok':successful.add(name)
    else:failures.append({'dataset':name,'status':state})
if status_counts['pending'] and not a.allow_partial:
    raise SystemExit(f"{status_counts['pending']} manifest entries are pending; use --allow-partial explicitly")
rows=list(csv.DictReader((a.directory/'analysis.csv').open()))
if not rows or len({r['dataset'] for r in rows})!=len(rows) or {r['dataset'] for r in rows}!=successful:
    raise SystemExit('analysis rows do not match successful manifest entries')
def family(name):return re.sub(r'[\d_\-]+$', '',name.lower()) or name.lower()
groups=collections.defaultdict(list)
for r in rows:groups[family(r['dataset'])].append(r)
keys=sorted(groups);rng=random.Random(a.seed)
metric_names=[x for x in rows[0] if x.startswith('speedup_')]
if not metric_names or any(not math.isfinite(float(r[k])) or float(r[k])<=0 for r in rows for k in metric_names):
    raise SystemExit('invalid speedup values')
samples={k:[] for k in metric_names}
for _ in range(a.iterations):
    draw=[rng.choice(keys) for _ in keys]
    for metric in metric_names:
        values=[math.log(float(r[metric])) for name in draw for r in groups[name]]
        samples[metric].append(math.exp(statistics.mean(values)))
report={'matrices':len(rows),'manifest_total':len(names),'status_counts':dict(status_counts),'failures':failures,
        'complete_success':len(successful)==len(names),'groups':len(keys),'iterations':a.iterations,'seed':a.seed,
        'caveat':'Approximate bootstrap using name-based groups. Matrix families may be misidentified, sampling need not represent SuiteSparse, and shared-GPU/timing bias is not captured.', 'metrics':{}}
for metric,v in samples.items():
    v.sort();report['metrics'][metric]={'point':math.exp(statistics.mean(math.log(float(r[metric])) for r in rows)),
        'ci95':[v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]}
(a.directory/'uncertainty.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
