#!/usr/bin/env python3
"""Summarize counterbalanced mixed-tile benchmark without dropping failures."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
manifest=json.loads((root/'manifest.json').read_text());meta=json.loads((root/'metadata.json').read_text())
fixed=[f'm{m}_b{b}' for m in (1,2,4,8,16,32) for b in (0,1)];items=[];counts=collections.Counter();errors=[]
for item in manifest:
    name=item['dataset'];path=root/'status'/f'{name}.json'
    if not path.exists():counts['pending']+=1;continue
    s=json.loads(path.read_text());counts[s['status']]+=1
    if s['status']!='ok':continue
    raw=list(csv.DictReader((root/'raw'/f'{name}.csv').open())); groups=collections.defaultdict(list);meta_rows={}
    for r in raw:
        key=(r['order'],r['candidate']);groups[key].append(float(r['us']))
        desc=(int(r['slots']),int(r['stored_bytes']),int(r['nnz']),tuple(int(r[f'groups_m{m}']) for m in (1,2,4,8,16,32)))
        if key in meta_rows and meta_rows[key]!=desc:raise ValueError(f'{name}: inconsistent candidate metadata')
        meta_rows[key]=desc
    expected={(o,c) for o in ('0','1') for c in ['mixed','mixed_specialized','mixed_bal4','mixed_bal16','default']+fixed}
    if set(groups)!=expected or any(len(v)!=meta['trials'] or any(not math.isfinite(x) or x<=0 for x in v) for v in groups.values()):raise ValueError(f'{name}: incomplete timings')
    times={k:statistics.median(v) for k,v in groups.items()}
    if meta_rows[('0','mixed')]!=meta_rows[('1','mixed')]:raise ValueError(f'{name}: inconsistent mixed layout')
    if any(meta_rows[(o,'mixed')]!=meta_rows[(o,'mixed_specialized')] for o in ('0','1')):raise ValueError(f'{name}: mixed representations differ')
    if any(meta_rows[(o,c)][0]!=meta_rows[(o,'mixed')][0] for o in ('0','1') for c in ('mixed_bal4','mixed_bal16')):raise ValueError(f'{name}: balanced slots differ')
    speed=[];specialized=[];balanced4=[];balanced16=[];best_mixed=[];specialized_vs_dynamic=[];default=[];bytes_speed=[];slots=[];best=[]
    for order in ('0','1'):
        fastest=min(fixed,key=lambda c:times[(order,c)]);best.append(fastest)
        m=meta_rows[(order,'mixed')];f=meta_rows[(order,fastest)]
        min_slots=min(meta_rows[(order,c)][0] for c in fixed)
        if m[0]>min_slots:raise ValueError(f'{name}: mixed slots exceed a fixed layout')
        speed.append(times[(order,fastest)]/times[(order,'mixed')]);default.append(times[(order,'default')]/times[(order,'mixed')])
        specialized.append(times[(order,fastest)]/times[(order,'mixed_specialized')]);specialized_vs_dynamic.append(times[(order,'mixed')]/times[(order,'mixed_specialized')])
        balanced4.append(times[(order,fastest)]/times[(order,'mixed_bal4')]);balanced16.append(times[(order,fastest)]/times[(order,'mixed_bal16')])
        best_mixed.append(times[(order,fastest)]/min(times[(order,c)] for c in ('mixed','mixed_specialized','mixed_bal4','mixed_bal16')))
        bytes_speed.append(m[1]/f[1]);slots.append(m[0]/min_slots if min_slots else 1.)
    geo=lambda x:math.exp(statistics.mean(math.log(v) for v in x))
    m=meta_rows[('0','mixed')];items.append({'dataset':name,'nnz':m[2],'speedup_vs_fastest_fixed':geo(speed),'specialized_vs_fastest_fixed':geo(specialized),'balanced4_vs_fastest_fixed':geo(balanced4),'balanced16_vs_fastest_fixed':geo(balanced16),'best_mixed_vs_fastest_fixed':geo(best_mixed),'specialized_vs_dynamic':geo(specialized_vs_dynamic),'speedup_vs_default':geo(default),
        'mixed_bytes_vs_fastest_fixed':geo(bytes_speed),'mixed_slots_vs_fewest_fixed':geo(slots),'mixed_slots':m[0],'mixed_bytes':m[1],
        'group_counts':'/'.join(map(str,m[3])),'fastest_fixed':'/'.join(best)})
result={'manifest_total':len(manifest),'status_counts':dict(counts),'complete':counts['ok']==len(manifest),
        'method':'Per-matrix geometric mean across opposite construction orders. Best fixed is the minimum measured median of 12 single-shape/mode candidates per order, an optimistic measured baseline. Memory counts final packed arrays and metadata, not input CSR, temporary allocations, setup or Graph. Samples with failures remain in status counts. Warm GPU Graph excludes setup.'}
if items:
    for key in ('speedup_vs_fastest_fixed','specialized_vs_fastest_fixed','balanced4_vs_fastest_fixed','balanced16_vs_fastest_fixed','best_mixed_vs_fastest_fixed','specialized_vs_dynamic','speedup_vs_default','mixed_bytes_vs_fastest_fixed','mixed_slots_vs_fewest_fixed'):
        values=[r[key] for r in items];result[key]={'geomean':math.exp(statistics.mean(math.log(x) for x in values)),
            'median':statistics.median(values),'min':min(values),'max':max(values),'cases':len(values),'above_one':sum(x>1 for x in values)}
    result['storage_slot_strictly_lower_cases']=sum(r['mixed_slots_vs_fewest_fixed']<1 for r in items)
    result['mixed_faster_cases']=sum(r['speedup_vs_fastest_fixed']>1 for r in items)
    result['specialized_faster_cases']=sum(r['specialized_vs_fastest_fixed']>1 for r in items)
    with (root/'analysis.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
