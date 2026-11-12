#!/usr/bin/env python3
"""Break down Plan initialization and warm-call reuse scenarios by matrix size."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.directory
metadata = json.loads((root/'metadata.json').read_text())
selection = metadata['selection']
with (root/'analysis.csv').open() as f:
    cases = list(csv.DictReader(f))
groups = collections.defaultdict(list)
reuse = (1, 100, 1000, 10000, 100000, 1000000)
for case in cases:
    name=case['dataset'];nnz=int(case['nnz'])
    label='<10k' if nnz<10_000 else '10k-100k' if nnz<100_000 else '100k-1M' if nnz<=1_000_000 else '>1M'
    prep=collections.defaultdict(dict);order=None
    for line in (root/'raw'/f'{name}.log').read_text().splitlines():
        if line.startswith('BENCH_ORDER,'):
            order=line.split(',',1)[1]
        elif line.startswith('PLAN_PREP,'):
            _,kind,value=line.split(',')
            if order is None or kind in prep[order]:raise ValueError(f'{name}: invalid preparation markers')
            prep[order][kind]=float(value)
    samples=collections.defaultdict(list)
    with (root/'raw'/f'{name}.csv').open() as f:
        for row in csv.DictReader(f):
            samples[row['order'],row['backend']].append(float(row['us']))
    orders=sorted({key[0] for key in samples})
    if not orders or any((o,b) not in samples or b not in prep[o] for o in orders for b in ('atbsr',selection)):
        raise ValueError(f'{name}: incomplete preparation or independent timing')
    if any(not math.isfinite(prep[o][b]) or prep[o][b]<0 or not all(math.isfinite(v) and v>0 for v in samples[o,b]) for o in orders for b in ('atbsr',selection)):
        raise ValueError(f'{name}: invalid timing')
    results={}
    for calls in reuse:
        per_order=[]
        for o in orders:
            base=prep[o]['atbsr']+calls*statistics.median(samples[o,'atbsr'])
            chosen=prep[o][selection]+calls*statistics.median(samples[o,selection])
            if base<=0 or chosen<=0:raise ValueError(f'{name}: invalid total time')
            per_order.append(base/chosen)
        results[calls]=math.exp(statistics.mean(map(math.log,per_order)))
    groups[label].append({'setup':statistics.mean(prep[o][selection] for o in orders),
                          'baseline_setup':statistics.mean(prep[o]['atbsr'] for o in orders), 'reuse':results})
report={'cases':len(cases),'selection':selection,
        'method':'Per-order Plan setup plus N independent warm Graph GPU calls, compared to default AtBSR; per-case paired geometric mean before group aggregation. Setup excludes input loading, caller allocations, Graph creation and CPU launch overhead. Grouped by nnz; incomplete sample is not a full-suite result.', 'groups':{}}
for label,values in sorted(groups.items()):
    report['groups'][label]={'cases':len(values),
        'median_selection_setup_ms':statistics.median(v['setup'] for v in values)/1000,
        'median_default_setup_ms':statistics.median(v['baseline_setup'] for v in values)/1000,
        'reuse_scenarios':{str(n):{'geomean_speedup':math.exp(statistics.mean(math.log(v['reuse'][n]) for v in values)),
                                    'wins':sum(v['reuse'][n]>1 for v in values)} for n in reuse}}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
