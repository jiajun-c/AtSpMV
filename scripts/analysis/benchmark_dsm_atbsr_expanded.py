#!/usr/bin/env python3
"""Preselect and screen a stratified, previously unmeasured SuiteSparse set."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,required=True)
    ap.add_argument('--matrix-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--count',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20261027)
    ap.add_argument('--timeout',type=int,default=300)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'screen').mkdir(exist_ok=True);(a.output/'configs').mkdir(exist_ok=True)
    prior={p.stem for p in (ROOT/'results/model/dsm-atbsr-hybrid/screen').glob('*.csv')}
    buckets={k:[] for k in ('high','medium','low','very_low','no_long')}
    index=ROOT/'results/analysis/longest_row_ratio.csv'
    for row in csv.DictReader(index.open()):
        try:
            name=row['dataset']; m=int(row['rows']);n=int(row['columns']);nnz=int(row['nnz']);mx=int(row['longest_row_nnz']);ratio=float(row['ratio'])
            path=a.matrix_root/name/(name+'.mtx')
            if row['status']!='ok' or name in prior or not(100<=m<=200000 and 1<=n<=1000000 and 100000<=nnz<=2000000) or not path.is_file():continue
            group='no_long' if mx<1024 else 'high' if ratio>=.04 else 'medium' if ratio>=.01 else 'low' if ratio>=.001 else 'very_low'
            buckets[group].append({'name':name,'rows':m,'cols':n,'nnz':nnz,'max_row':mx,'max_row_fraction':ratio,'group':group,'file':str(path.resolve()),'sha256':sha(path)})
        except (ValueError,KeyError):continue
    rng=random.Random(a.seed);selected=[]
    for group,values in buckets.items():
        values.sort(key=lambda r:r['name']);rng.shuffle(values);selected.extend(values[:a.count])
    meta={'selection':{'seed':a.seed,'per_stratum':a.count,'population_counts':{k:len(v) for k,v in buckets.items()},'criteria':'Exclude prior hybrid screen; local primary mtx, 100<=rows<=200000, cols<=1000000, 100000<=nnz<=2000000. Stratify by max row fraction >=0.04, >=0.01, >=0.001, below 0.001 (all with max row >=1024), plus max row <1024. Random sample without replacement in each stratum; do not infer population-wide averages.'},'manifest':selected,'matrix_stats_sha256':sha(index),'screen_exe_sha256':sha(a.exe),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_before':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    status=[]
    for r in selected:
        name=r['name']; t=time.monotonic();out=a.output/'screen'/f'{name}.csv';err=a.output/'screen'/f'{name}.log'
        try:
            with out.open('w') as stdout,err.open('w') as stderr:
                p=subprocess.run([str(a.exe.resolve()),r['file'],'--tune'],stdout=stdout,stderr=stderr,timeout=a.timeout)
            if p.returncode:raise RuntimeError(f'exit {p.returncode}')
            with out.open() as f:
                headline=f.readline().strip();rows=list(csv.DictReader(f))
            if 'correct=1' not in headline:raise RuntimeError('missing output validation')
            chosen=[]
            for kind in ['atbsr','csr_dsm','csr_two_stage','hybrid_dsm_atbsr']:
                candidates=sorted((v for v in rows if v['backend']==kind),key=lambda v:float(v['median_us']))
                if not candidates:raise RuntimeError('missing '+kind)
                seen=set()
                for x in candidates:
                    cfg=tuple(x[k] for k in ('tile_index','balanced','parts','threads','rows_per_group','threshold'))
                    if cfg in seen:continue
                    seen.add(cfg);chosen.append(','.join((kind,*cfg)))
                    if len(seen)>=3:break
            (a.output/'configs'/f'{name}-configs.csv').write_text('\n'.join(chosen)+'\n')
            status.append({'name':name,'status':'ok','headline':headline,'seconds':round(time.monotonic()-t,2)})
        except Exception as e:status.append({'name':name,'status':'error','error':str(e),'seconds':round(time.monotonic()-t,2)})
        (a.output/'screen-status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(name,status[-1]['status'],status[-1]['seconds'],flush=True)
    (a.output/'names.txt').write_text('\n'.join(x['name'] for x in status if x['status']=='ok')+'\n')
if __name__=='__main__':main()
