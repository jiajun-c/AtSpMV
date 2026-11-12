#!/usr/bin/env python3
"""Quantify timing differences for separately allocated identical Plans."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--first',required=True)
parser.add_argument('--second',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.directory
manifest=json.loads((root/'manifest.json').read_text());groups=collections.defaultdict(list);cases=[]
for item in manifest:
    name=item['dataset'];status=root/'status'/f'{name}.json'
    if not status.exists() or json.loads(status.read_text())['status']!='ok':continue
    samples=collections.defaultdict(list);configs={}
    for row in csv.DictReader((root/'raw'/f'{name}.csv').open()):
        if row['backend'] not in (args.first,args.second):continue
        key=(row['order'],row['backend']);samples[key].append(float(row['us']))
        old=configs.setdefault(key,row['config'])
        if old!=row['config']:raise ValueError(f'{name}: inconsistent selected plan')
    orders=('baseline_first','selection_first')
    if any((o,b) not in samples for o in orders for b in (args.first,args.second)):raise ValueError(f'{name}: missing samples')
    same=all(configs[o,args.first]==configs[o,args.second] for o in orders)
    ratios={o:statistics.median(samples[o,args.first])/statistics.median(samples[o,args.second]) for o in orders}
    if any(not math.isfinite(v) or v<=0 for v in ratios.values()):raise ValueError(f'{name}: invalid timings')
    ratio=math.exp(statistics.mean(map(math.log,ratios.values())))
    groups['same_config' if same else 'different_config'].append(ratio)
    if same:
        for order,value in ratios.items():groups['same_config/'+order].append(value)
        first_backend=configs[orders[0],args.first].split('/')[0]
        second_backend=configs[orders[1],args.first].split('/')[0]
        if first_backend==second_backend:groups['same_config/backend/'+first_backend].append(ratio)
    cases.append({'dataset':name,'same_config':same,'ratio':ratio})
report={'cases':len(cases),'manifest_total':len(manifest),'first':args.first,'second':args.second,
        'method':'Ratio of independently allocated Plan timings, per order, with paired geometric mean. Equal backend/config does not imply identical buffer placement or execution state; timing differences are measurement sensitivity, not a causal explanation.', 'groups':{}}
for key,values in sorted(groups.items()):
    report['groups'][key]={'cases':len(values),'geomean_ratio':math.exp(statistics.mean(map(math.log,values))),'median_ratio':statistics.median(values),'min_ratio':min(values),'max_ratio':max(values)}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
