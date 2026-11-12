#!/usr/bin/env python3
"""Summarize independently measured medium-candidate holdout Plans."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--replay',type=Path,required=True)
a=p.parse_args();root=a.directory
with a.replay.open(newline='') as f:
    cases={r['dataset']:r for r in csv.DictReader(f) if r['split']=='holdout'}
counts=Counter(); rows=[]
amortization = {n:[] for n in (10000,100000,1000000)}
for name,case in cases.items():
    status=root/'status'/(name+'.json')
    if not status.exists(): counts['pending']+=1;continue
    state=json.loads(status.read_text())['status'];counts[state]+=1
    if state!='ok': continue
    groups=defaultdict(list)
    with (root/'raw'/(name+'.csv')).open(newline='') as f:
        for r in csv.DictReader(f): groups[r['order'],r['backend']].append(float(r['us']))
    orders=sorted({o for o,_ in groups}); times={k:statistics.median(v) for k,v in groups.items()}
    if len(orders)!=2: raise ValueError(f'{name}: missing order')
    prep={};order=None
    for line in (root/'raw'/(name+'.log')).read_text().splitlines():
        fields=next(csv.reader([line]))
        if fields[0]=='BENCH_ORDER': order=fields[1]
        elif fields[0]=='PLAN_PREP':
            key=(order,fields[1])
            if order is None or key in prep: raise ValueError(f'{name}: invalid preparation log')
            prep[key]=float(fields[2])
    if any((o,b) not in prep or not math.isfinite(prep[o,b]) or prep[o,b]<=0
           for o in orders for b in ('auto_vector_model_v2','hybrid_dsm_atbsr')):
        raise ValueError(f'{name}: incomplete preparation log')
    selected=case['selected_medium']=='True'
    speed,regret=[],[]
    costs={n:[] for n in amortization}
    for order in orders:
        base=times[order,'auto_vector_model_v2'];hybrid=times[order,'hybrid_dsm_atbsr']
        actual=hybrid if selected else base
        speed.append(base/actual);regret.append(actual/min(base,hybrid))
        base_prep=prep[order,'auto_vector_model_v2']
        selected_prep=prep[order,'hybrid_dsm_atbsr'] if selected else base_prep
        for n in costs:
            costs[n].append((base_prep+n*base)/(selected_prep+n*actual))
    geo=lambda v:math.exp(statistics.mean(map(math.log,v)))
    for n in costs: amortization[n].append(geo(costs[n]))
    rows.append({'dataset':name,'selected_medium':selected,'speedup_vs_existing':geo(speed),
                 'over_two_plan_oracle':geo(regret),'medium_tile':case['medium_tile'],'medium_parts':case['medium_parts']})
summary={'total':len(cases),'status_counts':dict(counts),
         'method':'Measured actual Plans, two construction orders and warm Graph. The oracle only compares existing vector-v2 Plan and the single candidate predicted by the exploratory model; it is not a global optimum. Setup excluded, sample previously viewed.'}
if rows:
    speed=[r['speedup_vs_existing'] for r in rows];regret=[r['over_two_plan_oracle'] for r in rows]
    summary.update(cases=len(rows),selected_medium=sum(r['selected_medium'] for r in rows),
                   speedup_geomean=math.exp(statistics.mean(map(math.log,speed))),
                   faster_5pct=sum(x>1.05 for x in speed),slower_5pct=sum(x<1/1.05 for x in speed),
                   within_5pct_oracle=sum(x<=1.05 for x in regret))
    summary['amortization']={str(n):{'geomean_speedup':math.exp(statistics.mean(map(math.log,values))),
                                    'wins':sum(v>1 for v in values)} for n,values in amortization.items()}
    summary['amortization_method']='Per-order estimated selected Plan setup plus N warm Graph executions. Medium-model prediction, input reading, allocation by caller and Graph creation are excluded. Measurements contain construction-order and allocation effects.'
    with (root/'analysis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
