#!/usr/bin/env python3
"""Pad lower factors with independent identity rows for TC-SpTRSV tests."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import mmread,mmwrite
from scipy.sparse import block_diag,eye

p=argparse.ArgumentParser()
for name in ('manifest','factors','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=[]
for name in json.loads(a.manifest.read_text())['names']:
    path=a.factors/(name+'-L.mtx');record={'name':name}
    if not path.exists():record['status']='factor_missing';records.append(record);continue
    try:
        matrix=mmread(path).tocsr()
        if matrix.shape[0]==0 or matrix.shape[0]!=matrix.shape[1] or np.iscomplexobj(matrix.data):raise ValueError('expected nonempty real square factor')
        n=matrix.shape[0];pad=(-n)%16
        if pad:matrix=block_diag((matrix,eye(pad,format='csr')),format='csr')
        matrix.sort_indices();mmwrite(a.output/path.name,matrix,field='real',symmetry='general')
        record.update(status='ok',rows=n,padded_rows=n+pad)
    except Exception as error:record.update(status='padding_failed',error=repr(error))
    records.append(record)
(a.output/'padding.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'count':len(records),'status_counts':{s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}},indent=2))
