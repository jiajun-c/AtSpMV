#!/usr/bin/env python3
"""Summarize input-column locality on a frozen list of matrices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--list',type=Path,required=True)
p.add_argument('--matrix-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
names=[s.strip() for s in a.list.read_text().splitlines() if s.strip() and not s.startswith('#')]
if len(set(names))!=len(names) or any(Path(n).name!=n for n in names):raise ValueError('invalid matrix list')
cases=[]
for name in names:
    path=a.matrix_root/name/(name+'.mtx');h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    csr=mmread(path).tocsr();csr.sort_indices()
    lengths=np.diff(csr.indptr);nonempty=np.flatnonzero(lengths)
    first=csr.indptr[nonempty];last=csr.indptr[nonempty+1]-1
    spans=(csr.indices[last]-csr.indices[first]+1) if len(nonempty) else np.array([],dtype=np.int64)
    adjacent=np.diff(csr.indices.astype(np.int64))
    if len(adjacent):
        boundary=csr.indptr[1:-1]-1;boundary=boundary[(boundary>=0)&(boundary<len(adjacent))]
        mask=np.ones(len(adjacent),dtype=bool);mask[boundary]=False;adjacent=adjacent[mask]
    cases.append({'dataset':name,'matrix_sha256':h.hexdigest(),'rows':csr.shape[0],'cols':csr.shape[1],
                  'nnz_after_scipy_duplicate_coalescing':int(csr.nnz),
                  'distinct_columns':int(np.unique(csr.indices).size),
                  'distinct_64byte_x_lines':int(np.unique(csr.indices//8).size),
                  'mean_nonempty_row_span':float(np.mean(spans)) if len(spans) else 0.,
                  'median_nonempty_row_span':float(np.median(spans)) if len(spans) else 0.,
                  'mean_within_row_sorted_column_gap':float(np.mean(adjacent)) if len(adjacent) else 0.})
result={'method':'CPU structural descriptors after SciPy CSR duplicate coalescing and column sorting. A 64-byte x line assumes aligned FP64 x. Descriptors do not model actual AtBSR row order, hardware cache state, or performance causality.',
        'cases':cases}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
