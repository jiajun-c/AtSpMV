#!/usr/bin/env python3
"""Combine disjoint small/large sweeps while retaining pending and failures."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('small',type=Path);p.add_argument('large',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();scripts=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dirs=(a.small,a.large);indices=[];metas=[];rows=[];status=collections.Counter();details=[]
for root in dirs:
    index=list(csv.DictReader((root/'index.csv').open()));names={x['dataset'] for x in index}
    if len(names)!=len(index):raise ValueError(f'{root}: duplicate index entries')
    indices.append(names)
    executable=root/'provenance/spmv_hybrid_validation'
    if not executable.is_file():raise ValueError(f'{root}: missing frozen executable')
    metadata_path=root/'metadata.json'
    if not metadata_path.exists():
        status['pending']+=len(index);metas.append(None)
        details.append({'directory':str(root),'status':'not_started','index_cases':len(index),'exe_sha256':digest(executable)})
        continue
    meta=json.loads(metadata_path.read_text());metas.append(meta)
    if meta['exe_sha256']!=digest(executable):raise ValueError(f'{root}: executable hash mismatch')
    if meta['index_sha256']!=digest(root/'index.csv'):raise ValueError(f'{root}: index hash mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    if {x['dataset'] for x in manifest}!=names or len(manifest)!=len(names):raise ValueError(f'{root}: manifest mismatch')
    for attempt in range(5):
        subprocess.run([sys.executable,str(scripts/'audit_plan_sweep.py'),str(root)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,str(scripts/'analyze_plan_selection.py'),str(root)],check=True,stdout=subprocess.DEVNULL)
        audit=json.loads((root/'audit.json').read_text())
        observed=list(csv.DictReader((root/'analysis.csv').open())) if (root/'analysis.csv').exists() else []
        if len(observed)==audit['status_counts'].get('ok',0):break
    else:raise RuntimeError(f'{root}: results changed during repeated snapshots')
    status.update(audit['status_counts'])
    for row in observed:row['suite']=root.name;rows.append(row)
    details.append({'directory':str(root),'status':'started','index_cases':len(index),'status_counts':audit['status_counts'],
                    'failures':audit['failures'],'exe_sha256':meta['exe_sha256'],
                    'defer_memory_infeasible':meta.get('defer_memory_infeasible',False)})
if indices[0]&indices[1]:raise ValueError('small and large indices overlap')
if digest(dirs[0]/'provenance/spmv_hybrid_validation')!=digest(dirs[1]/'provenance/spmv_hybrid_validation'):
    raise ValueError('sweeps use different executables')
if all(metas):
    fields=('exe_sha256','gpu_uuid','seed','trials','timeout_s','allowed_gpu_pids','monitor_resident','selection','counterbalance','extended_baselines','vector_candidates','config_sha256','reverse_config_sha256','idle_policy')
    mismatch=[k for k in fields if metas[0].get(k)!=metas[1].get(k)]
    if mismatch:raise ValueError('incompatible benchmark metadata: '+', '.join(mismatch))
a.output.mkdir(parents=True,exist_ok=True)
report={'index_cases':sum(len(x) for x in indices),'status_counts':dict(status),'suite_complete':status.get('pending',0)==0,
        'all_successful':status.get('ok',0)==sum(len(x) for x in indices),'suites':details,
        'method':'Two construction orders; each ratio uses medians and a geometric average across orders. Aggregate includes successful cases only. Shared GPU and finite baseline shortlist; setup excluded. Memory-based deferral biases incomplete large-suite snapshots.'}
if rows:
    names=list(rows[0]);
    with (a.output/'analysis.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    for key in (k for k in names if k.startswith('speedup_')):
        values=[float(x[key]) for x in rows]
        report[key]={'cases':len(values),'geomean':math.exp(statistics.mean(math.log(x) for x in values)),
                     'median':statistics.median(values),'wins':sum(x>1 for x in values)}
tmp=a.output/'analysis.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output/'analysis.json')
print(json.dumps(report,indent=2))
