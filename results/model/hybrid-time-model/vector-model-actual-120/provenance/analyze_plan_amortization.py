#!/usr/bin/env python3
"""Estimate call-count break-even using recorded Plan preparation times."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args();root=a.directory
meta=json.loads((root/'metadata.json').read_text());selection=meta['selection'];manifest=json.loads((root/'manifest.json').read_text())
output=[];problems=[];scenario_ratios=collections.defaultdict(list);full_scenario_ratios=collections.defaultdict(list);model_scenario_ratios=collections.defaultdict(list)
for path in sorted((root/'status').glob('*.json')):
    status=json.loads(path.read_text())
    if status['status']!='ok':continue
    name=status['dataset'];prep=collections.defaultdict(dict);order=None
    for line in (root/'raw'/f'{name}.log').read_text().splitlines():
        if line.startswith('BENCH_ORDER,'):order=line.split(',',1)[1]
        elif line.startswith('PLAN_PREP,'):
            _,backend,value=line.split(',')
            if order is None or backend in prep[order]:problems.append(f'{name}: bad preparation markers');continue
            prep[order][backend]=float(value)
    samples=collections.defaultdict(list)
    for row in csv.DictReader((root/'raw'/f'{name}.csv').open()):
        samples[row['order'],row['backend']].append(float(row['us']))
    orders={k[0] for k in samples}
    if not orders or any('atbsr' not in prep[o] or selection not in prep[o] for o in orders):
        problems.append(f'{name}: missing preparation timings');continue
    if any(not math.isfinite(prep[o][b]) or prep[o][b]<0 for o in orders for b in ('atbsr',selection)):
        problems.append(f'{name}: invalid preparation timings');continue
    compare_full=meta.get('compare_full_tune',False)
    compare_model=meta.get('compare_auto_model',False)
    if compare_full and any('auto_tuned' not in prep[o] or (o,'auto_tuned') not in samples or not math.isfinite(prep[o]['auto_tuned']) or prep[o]['auto_tuned']<0 for o in orders):
        problems.append(f'{name}: missing or invalid full tuning timings');continue
    if compare_model and any('auto_model' not in prep[o] or (o,'auto_model') not in samples or not math.isfinite(prep[o]['auto_model']) or prep[o]['auto_model']<0 for o in orders):
        problems.append(f'{name}: missing or invalid existing model timings');continue
    preparation_delta=statistics.mean(prep[o][selection]-prep[o]['atbsr'] for o in orders)
    saving=statistics.mean(statistics.median(samples[o,'atbsr'])-statistics.median(samples[o,selection]) for o in orders)
    # Arithmetic averages reflect total time over alternating construction
    # orders. Negative/zero saving has no finite positive break-even.
    calls=max(0,math.ceil(preparation_delta/saving)) if saving>0 else None
    entry={'dataset':name,'extra_plan_setup_us':preparation_delta,'saved_us_per_call':saving,'break_even_calls':calls}
    if compare_full:
        entry['full_minus_fast_setup_us']=statistics.mean(prep[o]['auto_tuned']-prep[o][selection] for o in orders)
        entry['full_minus_fast_per_call_us']=statistics.mean(statistics.median(samples[o,'auto_tuned'])-statistics.median(samples[o,selection]) for o in orders)
    if compare_model:
        entry['model_minus_selected_setup_us']=statistics.mean(prep[o]['auto_model']-prep[o][selection] for o in orders)
        entry['model_minus_selected_per_call_us']=statistics.mean(statistics.median(samples[o,'auto_model'])-statistics.median(samples[o,selection]) for o in orders)
    output.append(entry)
    for count in (1,10,100,1000,10000,100000,1000000):
        ratios=[]
        for o in orders:
            base=prep[o]['atbsr']+count*statistics.median(samples[o,'atbsr'])
            selected=prep[o][selection]+count*statistics.median(samples[o,selection])
            if base<=0 or selected<=0:problems.append(f'{name}: invalid scenario total');break
            ratios.append(base/selected)
        if len(ratios)==len(orders):scenario_ratios[count].append(math.exp(statistics.mean(map(math.log,ratios))))
        if compare_full:
            full_ratios=[]
            for o in orders:
                full=prep[o]['auto_tuned']+count*statistics.median(samples[o,'auto_tuned'])
                fast=prep[o][selection]+count*statistics.median(samples[o,selection])
                if full<=0 or fast<=0:problems.append(f'{name}: invalid full tuning scenario total');break
                full_ratios.append(full/fast)
            if len(full_ratios)==len(orders):full_scenario_ratios[count].append(math.exp(statistics.mean(map(math.log,full_ratios))))
        if compare_model:
            model_ratios=[]
            for o in orders:
                base=prep[o]['auto_model']+count*statistics.median(samples[o,'auto_model'])
                selected=prep[o][selection]+count*statistics.median(samples[o,selection])
                if base<=0 or selected<=0:problems.append(f'{name}: invalid model scenario total');break
                model_ratios.append(base/selected)
            if len(model_ratios)==len(orders):model_scenario_ratios[count].append(math.exp(statistics.mean(map(math.log,model_ratios))))
if problems:raise SystemExit('\n'.join(problems))
if output:
    with (root/'amortization.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
    positive=[x['break_even_calls'] for x in output if x['break_even_calls'] is not None]
    result={'cases':len(output),'manifest_total':len(manifest),'all_cases_completed_successfully':len(output)==len(manifest),'positive_per_call_saving':len(positive),'median_break_even_calls':statistics.median(positive) if positive else None,
            'method':'Additional plan setup divided by independently measured warm Graph per-call savings; arithmetic average over construction orders. Excludes input loading, caller allocations, graph creation and CPU launch overhead. Not applicable to one-off cold execution.'}
    result['reuse_scenarios']={str(n):{'cases':len(v),'geomean_speedup':math.exp(statistics.mean(map(math.log,v))),'wins':sum(x>1 for x in v)} for n,v in scenario_ratios.items()}
    if meta.get('compare_full_tune',False):
        result['versus_full_tune']={'median_full_minus_fast_setup_us':statistics.median(x['full_minus_fast_setup_us'] for x in output),
            'reuse_scenarios':{str(n):{'cases':len(v),'geomean_speedup':math.exp(statistics.mean(map(math.log,v))),'wins':sum(x>1 for x in v)} for n,v in full_scenario_ratios.items()}}
    if meta.get('compare_auto_model',False):
        result['versus_auto_model']={'median_model_minus_selected_setup_us':statistics.median(x['model_minus_selected_setup_us'] for x in output),
            'reuse_scenarios':{str(n):{'cases':len(v),'geomean_speedup':math.exp(statistics.mean(map(math.log,v))),'wins':sum(x>1 for x in v)} for n,v in model_scenario_ratios.items()}}
    (root/'amortization.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
