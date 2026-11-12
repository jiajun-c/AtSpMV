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
output=[];problems=[]
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
    preparation_delta=statistics.mean(prep[o][selection]-prep[o]['atbsr'] for o in orders)
    saving=statistics.mean(statistics.median(samples[o,'atbsr'])-statistics.median(samples[o,selection]) for o in orders)
    # Arithmetic averages reflect total time over alternating construction
    # orders. Negative/zero saving has no finite positive break-even.
    calls=max(0,math.ceil(preparation_delta/saving)) if saving>0 else None
    output.append({'dataset':name,'extra_plan_setup_us':preparation_delta,'saved_us_per_call':saving,'break_even_calls':calls})
if problems:raise SystemExit('\n'.join(problems))
if output:
    with (root/'amortization.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
    positive=[x['break_even_calls'] for x in output if x['break_even_calls'] is not None]
    result={'cases':len(output),'manifest_total':len(manifest),'all_cases_completed_successfully':len(output)==len(manifest),'positive_per_call_saving':len(positive),'median_break_even_calls':statistics.median(positive) if positive else None,
            'method':'Additional plan setup divided by independently measured warm Graph per-call savings; arithmetic average over construction orders. Excludes input loading, caller allocations, graph creation and CPU launch overhead. Not applicable to one-off cold execution.'}
    (root/'amortization.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
