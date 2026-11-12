#!/usr/bin/env python3
"""Summarize both candidate orders; keep failed and missing cases explicit."""
import argparse
import csv
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('out',type=Path)
a=p.parse_args()
rows=[]
for blocks in (2,8):
    for density in (5,25,100):
        for n in (8,32,128):
            states={}
            for order in ('forward','reverse'):
                path=a.out/'cases'/f'b{blocks}-d{density}-n{n}-{order}.json'
                if not path.exists(): states[order]={};continue
                data=json.loads(path.read_text())
                states[order]={r['method']:r for r in data['rows']} if data['returncode']==0 else {}
            for method in ('csr_alg2','csr_alg3'):
                times=[]
                for order in ('forward','reverse'):
                    left=states[order].get(method,{});right=states[order].get('blocked_ell',{})
                    if left.get('status')!='ok' or right.get('status')!='ok': break
                    if any(left[k]!=right[k] for k in ('m','k','n','nnz','block','blocks_per_row','requested_density')):
                        raise ValueError('inconsistent input')
                    times.append((float(left['median_us']),float(right['median_us'])))
                row={'blocks_per_row':blocks,'density_percent':density,'n':n,'method':method,
                     'status':'ok' if len(times)==2 else 'missing_or_failed'}
                if len(times)==2:
                    row.update(csr_us=math.sqrt(times[0][0]*times[1][0]),
                               ell_us=math.sqrt(times[0][1]*times[1][1]),
                               ell_over_csr=math.sqrt((times[0][1]/times[0][0])*(times[1][1]/times[1][0])))
                rows.append(row)
names=['blocks_per_row','density_percent','n','method','status','csr_us','ell_us','ell_over_csr']
with (a.out/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
valid=[r for r in rows if r['status']=='ok']
summary={'comparisons':len(rows),'valid':len(valid),
         'csr_more_than_5pct_faster':sum(r['ell_over_csr']>1.05 for r in valid),
         'ell_more_than_5pct_faster':sum(r['ell_over_csr']<1/1.05 for r in valid),
         'scope':'Synthetic FP16 inputs/FP32 accumulation and output; two plan orders, warm Graph; format conversion and setup excluded. CUDA/Tensor instructions require independent profiling.'}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
