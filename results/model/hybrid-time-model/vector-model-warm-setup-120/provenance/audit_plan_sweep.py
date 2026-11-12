#!/usr/bin/env python3
"""Audit manifest, paired timing records and imported-record provenance."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args();root=a.directory
meta=json.loads((root/'metadata.json').read_text());manifest=json.loads((root/'manifest.json').read_text());names=[r['dataset'] for r in manifest]
errors=[];counts=collections.Counter();failures=[]
if len(names)!=len(set(names)) or any(Path(n).name!=n for n in names):errors.append('duplicate or invalid manifest names')
if meta.get('warm_setup_anchor'):
    forward=(root/'configs.csv').read_text().splitlines();reverse=(root/'configs-reversed.csv').read_text().splitlines()
    if not forward or not forward[0].startswith('atbsr,') or reverse!=[forward[0]]+list(reversed(forward[1:])):
        errors.append('warm setup anchor or reversed plan order is invalid')
candidate={'atbsr','cusparse',meta['selection']}
if meta.get('extended_baselines'):candidate.update(('atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed'))
if meta.get('vector_candidates'):candidate.update(f'csr_vec{x}' for x in (1,2,4,8,16,32))
if meta.get('compare_full_tune'):candidate.add('auto_tuned')
if meta.get('compare_auto_model'):candidate.add('auto_model')
orders=('baseline_first','selection_first') if meta.get('counterbalance') else ('baseline_first',)
for name in names:
    status_path=root/'status'/f'{name}.json'
    if not status_path.exists():counts['pending']+=1;continue
    status=json.loads(status_path.read_text());state=status.get('status');counts[state]+=1
    if state!='ok':failures.append({'dataset':name,'status':state,'returncode':status.get('returncode')});continue
    try:
        records=list(csv.DictReader((root/'raw'/f'{name}.csv').open()));group=collections.defaultdict(set);config={}
        for r in records:
            value=float(r['us']);trial=int(r['trial']);order=r.get('order','baseline_first');key=(order,r['backend'])
            if not math.isfinite(value) or value<=0:raise ValueError('invalid timing')
            if trial in group[key]:raise ValueError('duplicate trial')
            group[key].add(trial)
            previous=config.setdefault(key,r['config'])
            if previous!=r['config']:raise ValueError('inconsistent configuration')
        if set(group)!={(o,c) for o in orders for c in candidate} or any(v!=set(range(meta['trials'])) for v in group.values()):raise ValueError('missing or unexpected candidates/trials')
    except (OSError,KeyError,ValueError) as e:errors.append(f'{name}: {e}')
imports=root/'imported-v3.json'
imported=0
if imports.exists():
    data=json.loads(imports.read_text())
    for item in data['cases']:
        name=item['dataset'];imported+=1
        for label,path in (('status_sha256',root/'status'/f'{name}.json'),('raw_sha256',root/'raw'/f'{name}.csv'),('log_sha256',root/'raw'/f'{name}.log')):
            try:digest=hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:errors.append(f'{name}: missing imported {label}');continue
            if digest!=item[label]:errors.append(f'{name}: imported {label} mismatch')
report={'manifest_total':len(names),'status_counts':dict(counts),'imported_records':imported,'failures':failures,'problems':errors,
        'scope':'Checks status and recorded trials, not numerical results or unsampled GPU interference; benchmark independently performs numerical reference checks.'}
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(bool(errors))
