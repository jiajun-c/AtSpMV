#!/usr/bin/env python3
"""Audit logged Auto+tune decisions against measured medians and gate."""
import collections
import csv
import json
from pathlib import Path
import re
import sys

if len(sys.argv)!=2:raise SystemExit('usage: audit_tuned_choice.py DIRECTORY')
root=Path(sys.argv[1]);meta=json.loads((root/'metadata.json').read_text())
selection=meta.get('selection')
if selection not in ('auto_tuned','auto_fast_tuned'):raise SystemExit('requires Auto+tune data')
problems=[];cases=0;blocks_checked=0
for status_path in sorted((root/'status').glob('*.json')):
    status=json.loads(status_path.read_text())
    if status.get('status')!='ok':continue
    name=status['dataset'];records=list(csv.DictReader((root/'raw'/f'{name}.csv').open()))
    labels=[];selected={}
    for r in records:
        label=r['order']
        if label not in labels:labels.append(label)
        if r['backend']==selection:
            old=selected.setdefault(label,r['config'])
            if old!=r['config']:problems.append(f'{name}: inconsistent selected config')
    groups=[];current=None;description=None;last_plan=None;block_plan=None
    for line in (root/'raw'/f'{name}.log').read_text().splitlines():
        if line.startswith('PLAN_PREP,'):
            last_plan=line.split(',')[1]
        elif line.startswith('TUNING_BEGIN '):
            m=re.search(r'selected=([^ ]+) vector_width=(\d+)',line)
            if current is not None or not m:problems.append(f'{name}: invalid begin');continue
            current=[];description=(m.group(1),int(m.group(2)));block_plan=last_plan or selection
        elif line.startswith('TUNING,') and current is not None:
            f=next(csv.reader([line]));current.append({'backend':f[1],'width':int(f[8]),'us':float(f[9])})
        elif line=='TUNING_END':
            if current is None:problems.append(f'{name}: invalid end');continue
            if block_plan==selection:groups.append((description,current))
            current=None;last_plan=None;block_plan=None
    if current is not None or len(groups)!=len(labels):problems.append(f'{name}: tuning block count');continue
    for order,(description,candidates) in zip(labels,groups):
        # Auto's built-in mixed shortlist contains only nonempty long-row
        # partitions. Explicit custom lists are not used by this benchmark.
        mixed={'hybrid_dsm_atbsr','csr_dsm_vector'}
        non=[x for x in candidates if x['backend'] not in mixed];hybrid=[x for x in candidates if x['backend'] in mixed]
        if not non:problems.append(f'{name}: no nonmixed candidate');continue
        winner=min(non,key=lambda x:x['us'])
        if hybrid:
            best=min(hybrid,key=lambda x:x['us'])
            if best['us']<=winner['us']*.95:winner=best
        if description!=(winner['backend'],winner['width']):problems.append(f'{name}/{order}: logged winner differs from timing gate')
        config=selected.get(order,'').split('/')
        if not config or config[0]!=description[0] or config[-1]!=f'vw={description[1]}':
            problems.append(f'{name}/{order}: benchmark selection differs from tuning log')
        blocks_checked+=1
    cases+=1
report={'cases':cases,'blocks_checked':blocks_checked,'problems':problems,
        'scope':'Checks logged medians and 5% selection gate; does not independently validate GPU timing or numerical accuracy.'}
(root/'tuning-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(bool(problems))
