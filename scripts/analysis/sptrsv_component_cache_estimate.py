#!/usr/bin/env python3
"""Estimate tile cache bytes for exact independent graph components.

Uses the same row grouping and K thresholds as sptrsvPacking.h. Counts the
existing single-CTA kernel's shared data only; an implementation with local
row mappings or per-CTA metadata can require additional memory.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components


def estimate(matrix):
    a=matrix.tocsr();a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
    n=a.shape[0]
    if n==0 or a.shape[1]!=n:raise ValueError('nonempty square matrix required')
    level=np.zeros(n,dtype=np.int64);length=np.zeros(n,dtype=np.int64)
    for i in range(n):
        cols=a.indices[a.indptr[i]:a.indptr[i+1]]
        if len(cols)==0 or cols[-1]!=i or np.any(cols>i):raise ValueError('expected explicit lower triangular diagonal')
        deps=cols[:-1];length[i]=len(deps)
        if len(deps):level[i]=level[deps].max()+1
    count,labels=connected_components(a,directed=False);order=np.argsort(labels,kind='stable');counts=np.bincount(labels,minlength=count)
    offset=0
    for component,size in enumerate(counts):
        rows=order[offset:offset+size];offset+=size
        grouped=sorted(rows,key=lambda i:(level[i],i));local_levels=int(level[rows].max()+1)
        groups=slots=row_slots=0;start=0
        for lev in range(local_levels):
            end=start
            while end<len(grouped) and level[grouped[end]]==lev:end+=1
            if start==end:raise ValueError('dependency level missing within component')
            section=grouped[start:end];mean=sum(int(length[i]) for i in section)/len(section)
            k=1 if mean<=2 else 4 if mean<=6 else 8 if mean<=16 else 16 if mean<=64 else 32
            height=32//k
            for first in range(0,len(section),height):
                maxlen=max(int(length[i]) for i in section[first:first+height]);groups+=1;row_slots+=height;slots+=((maxlen+k-1)//k)*32
            start=end
        cache_bytes=8*(3*int(size)+slots)+4*(2*groups+slots+row_slots+2*local_levels+2)
        yield {'component':component,'rows':int(size),'levels':local_levels,'groups':groups,'slots':slots,'estimated_bytes':cache_bytes}


def main():
    p=argparse.ArgumentParser();p.add_argument('factor',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=list(estimate(mmread(a.factor)));fields=['component','rows','levels','groups','slots','estimated_bytes']
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps({'factor':str(a.factor),'components':len(rows),'max_estimated_bytes':max(r['estimated_bytes'] for r in rows),
                      'max_component_rows':max(r['rows'] for r in rows)},indent=2))


if __name__=='__main__':main()
