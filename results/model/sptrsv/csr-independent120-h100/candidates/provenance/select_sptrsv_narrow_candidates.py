#!/usr/bin/env python3
"""Freeze an exploratory deep, narrow-level CSR candidate set."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.io import mmread

p=argparse.ArgumentParser()
for key in ('manifest','factors','output'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--max-bytes',type=int,default=225000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for name in json.loads(a.manifest.read_text())['names']:
    row={'name':name,'status':'factor_missing','rows':'','levels':'','mean_rows_per_level':'','csr_bytes':'','selected':False};path=a.factors/(name+'-L.mtx')
    if path.exists():
        try:
            mat=mmread(path).tocsr();mat.sum_duplicates();mat.eliminate_zeros();mat.sort_indices();n=mat.shape[0]
            if n==0 or mat.shape[1]!=n or np.iscomplexobj(mat.data) or not np.all(np.isfinite(mat.data)):raise ValueError('invalid input')
            level=np.zeros(n,dtype=np.int32)
            for i in range(n):
                col=mat.indices[mat.indptr[i]:mat.indptr[i+1]]
                if len(col)==0 or col[-1]!=i or np.any(col>i) or mat.data[mat.indptr[i+1]-1]==0:raise ValueError('expected lower triangular with explicit diagonal')
                if len(col)>1:level[i]=int(level[col[:-1]].max())+1
            levels=int(level.max())+1;mean=n/levels;nnz=int(mat.nnz)
            shared=8*(2*n+nnz)+4*((n+1)+nnz+n+(levels+1))
            row.update(status='ok',rows=n,levels=levels,mean_rows_per_level=mean,csr_bytes=shared,
                       selected=n<=4000 and levels>=32 and mean<=4 and shared<=a.max_bytes)
        except Exception as error:row.update(status='error',error=repr(error))
    rows.append(row)
with (a.output/'features.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['name','status','rows','levels','mean_rows_per_level','csr_bytes','selected','error']);w.writeheader();w.writerows(rows)
selected=[r['name'] for r in rows if r['selected']]
(a.output/'manifest.json').write_text(json.dumps({'selection':'levels >=32, n/levels <=4, n<=4000, CSR cache bytes <= max_bytes','max_bytes':a.max_bytes,'names':selected},indent=2)+'\n')
print(json.dumps({'records':len(rows),'valid':sum(r['status']=='ok' for r in rows),'selected':len(selected)},indent=2))
