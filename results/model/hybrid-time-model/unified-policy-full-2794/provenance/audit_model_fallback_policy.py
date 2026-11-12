#!/usr/bin/env python3
"""Audit model-versus-tuning policy and its logged tuning decisions."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--assume-h100-default-scope',action='store_true',help='Also check matrix-size routing under default H100 PCIe model settings')
args=parser.parse_args();root=args.directory
meta=json.loads((root/'metadata.json').read_text())
if meta['selection']!='auto_model_fallback_tuned':raise SystemExit('requires model fallback tuning selection')
manifest=json.loads((root/'manifest.json').read_text());issues=[];counts=collections.Counter();paths=collections.Counter();retries=collections.Counter();checked=0
for item in manifest:
    name=item['dataset'];status_path=root/'status'/f'{name}.json'
    if not status_path.exists():counts['pending']+=1;continue
    status=json.loads(status_path.read_text());counts[status['status']]+=1
    if status['status']!='ok':continue
    selected={}
    for r in csv.DictReader((root/'raw'/f'{name}.csv').open()):
        if r['backend']!=meta['selection']:continue
        old=selected.setdefault(r['order'],r['config'])
        if old!=r['config']:issues.append(f'{name}: inconsistent selected config')
    order=None;mode={};retry={};blocks={};current=None;description=None
    for line in (root/'raw'/f'{name}.log').read_text().splitlines():
        if line.startswith('BENCH_ORDER,'):order=line.split(',',1)[1]
        elif line.startswith('PLAN_OOM_RETRY,auto_model_fallback_tuned,'):
            if order is None or order in retry:issues.append(f'{name}: invalid retry marker');continue
            retry[order]=line.rsplit(',',1)[1]
        elif line.startswith('MODEL_FALLBACK_POLICY,'):
            if order is None or order in mode:issues.append(f'{name}: invalid policy marker');continue
            mode[order]=line.split(',')[1]
        elif line.startswith('TUNING_BEGIN '):
            match=re.search(r'selected=([^ ]+) vector_width=(\d+)',line)
            if current is not None or not match:issues.append(f'{name}: invalid tuning begin');continue
            current=[];description=(match.group(1),int(match.group(2)))
        elif line.startswith('TUNING,') and current is not None:
            f=next(csv.reader([line]));current.append({'backend':f[1],'width':int(f[8]),'us':float(f[9])})
        elif line=='TUNING_END':
            if current is None or order in blocks:issues.append(f'{name}: invalid tuning end');continue
            blocks[order]=(description,current);current=None
    orders=('baseline_first','selection_first') if meta.get('counterbalance') else ('baseline_first',)
    if current is not None or set(mode)!=set(orders) or set(selected)!=set(orders):issues.append(f'{name}: incomplete policy records');continue
    if meta.get('retry_shared_storage_on_oom') and set(retry)!=set(orders):issues.append(f'{name}: incomplete retry records');continue
    for o in orders:
        kind=mode[o];paths[kind]+=1
        if kind not in ('model','tuned'):issues.append(f'{name}/{o}: unknown policy');continue
        if meta.get('retry_shared_storage_on_oom'):
            if retry[o] not in ('0','1') or (retry[o]=='1' and kind!='tuned'):issues.append(f'{name}/{o}: invalid retry outcome')
            else:retries[retry[o]]+=1
        if args.assume_h100_default_scope:
            in_scope=int(item['nnz'])<=1000000 and int(item['rows'])<=349651 and int(item['columns'])<=1785345
            if kind!=('model' if in_scope else 'tuned'):issues.append(f'{name}/{o}: unexpected scope routing')
        if kind=='model':
            if o in blocks:issues.append(f'{name}/{o}: model path unexpectedly tuned')
        else:
            if o not in blocks:issues.append(f'{name}/{o}: missing tuning block');continue
            desc,candidates=blocks[o];mixed={'hybrid_dsm_atbsr','csr_dsm_vector'}
            non=[x for x in candidates if x['backend'] not in mixed];hybrid=[x for x in candidates if x['backend'] in mixed]
            if not non:issues.append(f'{name}/{o}: missing nonmixed candidates');continue
            winner=min(non,key=lambda x:x['us'])
            if hybrid:
                best=min(hybrid,key=lambda x:x['us'])
                if best['us']<=winner['us']*.95:winner=best
            if desc!=(winner['backend'],winner['width']):issues.append(f'{name}/{o}: tuning gate mismatch')
            config=selected[o].split('/')
            if config[0]!=desc[0] or config[-1]!=f'vw={desc[1]}':issues.append(f'{name}/{o}: selected config mismatch')
        checked+=1
report={'manifest_total':len(manifest),'status_counts':dict(counts),'policy_paths':dict(paths),'oom_retry_orders':dict(retries),'orders_checked':checked,'issues':issues,
        'scope':'Checks recorded routing, logs and default 5% gate, not independent timing accuracy. H100 scope check assumes no model/kernel environment overrides.'}
(root/'policy-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(bool(issues))
