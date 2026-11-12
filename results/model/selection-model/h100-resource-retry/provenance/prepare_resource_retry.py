#!/usr/bin/env python3
"""Prepare reduced-selection retries for recorded resource failures only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[];seen=set();counts=Counter();evidence=[]
for run in a.run:
    manifest=json.loads((run/'manifest.json').read_text())
    for item in manifest:
        name=item['dataset']
        if name in seen: raise ValueError(f'duplicate matrix across runs: {name}')
        seen.add(name)
        status_path=run/'status'/(name+'.json')
        if not status_path.exists(): raise ValueError(f'{name}: source run incomplete')
        status=json.loads(status_path.read_text());state=status['status']
        reason=None
        if state=='timeout': reason='timeout'
        elif state=='error':
            log_path=run/'raw'/(name+'.log')
            lines=log_path.read_text(errors='replace').lower().splitlines()
            last=next((line.strip() for line in reversed(lines) if line.strip()),'')
            if 'out of memory' in last or 'cuda_error_out_of_memory' in last or last == 'cusparse status 2':
                reason='gpu_memory_error'
            elif 'no tile layout fits the current 32-bit kernel offsets' in last:
                reason='tile_offset_overflow'
        counts[reason or state]+=1
        if reason:
            rows.append(item)
            evidence.append({'dataset':name,'reason':reason,'source_run':str(run.resolve()),
                             'status_sha256':hashlib.sha256(status_path.read_bytes()).hexdigest(),
                             'log_sha256':hashlib.sha256((run/'raw'/(name+'.log')).read_bytes()).hexdigest()})
a.output.mkdir(parents=True,exist_ok=True)
fields=['dataset','rows','columns','nnz','longest_row','longest_row_nnz','ratio','status']
(a.output/'retry-evidence.json').write_text(json.dumps({'source_counts':dict(counts),'retries':evidence},indent=2)+'\n')
temporary=a.output/'index.csv.tmp'
with temporary.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
os.replace(temporary,a.output/'index.csv')
print(f'{len(seen)} source cases checked; {len(rows)} eligible for reduced retry')
