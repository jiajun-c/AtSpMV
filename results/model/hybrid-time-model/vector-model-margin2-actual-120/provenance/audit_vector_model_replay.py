#!/usr/bin/env python3
"""Check actual Plan decisions match the frozen Python model replay."""
import argparse
import collections
import csv
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('actual',type=Path)
parser.add_argument('replay',type=Path)
parser.add_argument('--margin',type=float,default=0.)
args=parser.parse_args();actual=args.actual
if not 0<=args.margin<1:raise ValueError('invalid margin')
expected={r['dataset']:r for r in csv.DictReader((args.replay/'frozen-model-results.csv').open())}
selection=json.loads((actual/'metadata.json').read_text())['selection']
issues=[];counts=collections.Counter();checked=0
manifest=json.loads((actual/'manifest.json').read_text())
for item in manifest:
    name=item['dataset'];path=actual/'status'/f'{name}.json'
    if not path.exists():counts['pending']+=1;continue
    status=json.loads(path.read_text());counts[status['status']]+=1
    if status['status']!='ok':continue
    if name not in expected:issues.append(f'{name}: missing replay decision');continue
    configs=collections.defaultdict(dict)
    for row in csv.DictReader((actual/'raw'/f'{name}.csv').open()):
        if row['backend'] not in (selection,'auto_model'):continue
        previous=configs[row['order']].setdefault(row['backend'],row['config'])
        if previous!=row['config']:issues.append(f'{name}: inconsistent plan config')
    if set(configs)!={'baseline_first','selection_first'}:
        issues.append(f'{name}: missing construction order');continue
    for order,values in configs.items():
        actual_config=values.get(selection,'');baseline=values.get('auto_model','')
        if not actual_config or not baseline:issues.append(f'{name}/{order}: missing plan');continue
        replay=expected[name];choice=replay['chosen']
        if choice!='auto_model' and float(replay['predicted_ratio_to_default']) >= (1.-args.margin)*float(replay['fallback_predicted_ratio_to_default']):choice='auto_model'
        if choice=='auto_model':
            if actual_config!=baseline:issues.append(f'{name}/{order}: expected fallback, got {actual_config} versus {baseline}')
        else:
            width=int(choice.removeprefix('csr_vec'))
            if not actual_config.startswith('csr_vector/') or not actual_config.endswith(f'/vw={width}'):
                issues.append(f'{name}/{order}: expected {choice}, got {actual_config}')
        checked+=1
report={'cases':sum(counts.values())-counts['pending'],'status_counts':dict(counts),'decisions_checked':checked,
        'issues':issues,'scope':'Checks actual selected backends and widths against previously frozen Python replay; does not validate GPU timing independently.'}
(actual/'replay-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(bool(issues))
