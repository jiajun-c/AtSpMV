#!/usr/bin/env python3
"""Count exact dependency components of each real lower triangular factor."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components

p=argparse.ArgumentParser()
for key in ('manifest','factors','output'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();rows=[]
for name in json.loads(a.manifest.read_text())['names']:
    row={'name':name,'status':'factor_missing'};path=a.factors/(name+'-L.mtx')
    if path.exists():
        try:
            m=mmread(path).tocsr()
            if m.shape[0]!=m.shape[1] or m.shape[0]==0:raise ValueError('non-square or empty')
            m.eliminate_zeros();count,labels=connected_components(m,directed=False)
            sizes=np.bincount(labels,minlength=count);multi=sizes[sizes>1]
            row.update(status='ok',rows=m.shape[0],nnz=m.nnz,components=count,
                       nontrivial_components=len(multi),largest_component=int(sizes.max()),
                       nontrivial_rows=int(multi.sum()),largest_fraction=float(sizes.max()/m.shape[0]))
        except Exception as e:row.update(status='error',error=repr(e))
    rows.append(row)
fields=['name','status','rows','nnz','components','nontrivial_components','largest_component','nontrivial_rows','largest_fraction','error']
with a.output.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
print(json.dumps({'records':len(rows),'counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}},indent=2))
