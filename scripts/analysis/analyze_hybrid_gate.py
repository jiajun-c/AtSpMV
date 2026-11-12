#!/usr/bin/env python3
"""Retrospective effect of the post-hoc long-row fraction guard."""
import collections
import csv
import json
import math
import statistics
from fit_hybrid_selector import OUT

features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features.csv').open())}
rows=list(csv.DictReader((OUT/'unified-reference.csv').open()))
by_name=collections.defaultdict(list)
for r in rows:by_name[r['dataset']].append(r)
report={'method':'Retrospective replay after inspecting a holdout error. Neither dev nor holdout is an independent validation of this post-hoc guard.', 'splits':{}}
for split in ('train','dev','holdout'):
    results={}
    for gate in (False,True):
        speeds=[];selected=slower=hybrid=hybrid_slower=0
        dominated=dominated_wins=0
        for name,rs in by_name.items():
            if rs[0]['split']!=split:continue
            nonhybrid=[r for r in rs if int(r['candidate'])<5]
            best=min(nonhybrid,key=lambda r:float(r['prediction_ratio']))
            choice=best if float(best['prediction_ratio'])<.95 else None
            estimate=float(choice['prediction_ratio']) if choice else 1.
            candidates=[]
            for r in rs:
                i=int(r['candidate'])
                if i<5:continue
                f=features[name,4096 if i==7 else 1024]
                is_dominated=int(f['long_rows'])*10>=int(f['rows'])
                if is_dominated:
                    dominated+=1;dominated_wins+=float(r['actual_ratio'])<1
                if not gate or not is_dominated:candidates.append(r)
            if candidates:
                best=min(candidates,key=lambda r:float(r['prediction_ratio']))
                if float(best['prediction_ratio'])<estimate*.9:choice=best
            ratio=float(choice['actual_ratio']) if choice else 1.
            selected+=bool(choice);slower+=ratio>1
            hybrid+=bool(choice and int(choice['candidate'])>=5)
            hybrid_slower+=bool(choice and int(choice['candidate'])>=5 and ratio>1)
            speeds.append(1/ratio)
        results['guarded' if gate else 'original']={'matrices':len(speeds),'geomean':math.exp(statistics.mean(map(math.log,speeds))),
            'selected':selected,'slower':slower,'hybrid_selected':hybrid,'hybrid_slower':hybrid_slower,
            'long_row_dominated_candidates':dominated,'long_row_dominated_wins':dominated_wins}
    report['splits'][split]=results
(OUT/'hybrid-gate-retrospective.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
