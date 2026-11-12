#!/usr/bin/env python3
"""Resumable per-matrix warm-cache DSM/AtBSR/cuSPARSE comparison.

Run only on an otherwise unused H100. Full manifest includes unavailable or
failed inputs; completed results are never silently treated as successful.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def apps(uuid):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    found=[]
    for line in result.stdout.splitlines():
        parts=[x.strip() for x in line.split(',',2)]
        if len(parts)==3 and parts[0]==uuid:found.append({'pid':int(parts[1]),'name':parts[2]})
    return found

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--index',type=Path,default=ROOT/'results/analysis/longest_row_ratio.csv')
    ap.add_argument('--matrix-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu-index',type=int,required=True,help='physical nvidia-smi index; used as CUDA_VISIBLE_DEVICES')
    ap.add_argument('--timeout',type=int,default=300);ap.add_argument('--trials',type=int,default=9);ap.add_argument('--seed',type=int,default=202610271)
    ap.add_argument('--limit',type=int,default=0,help='stop after this many new cases (0: all)')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for d in ('raw','status'): (a.output/d).mkdir(exist_ok=True)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=list(csv.DictReader(a.index.open()));seen=set()
    for row in rows:
        name=row['dataset']
        if name in seen or Path(name).name!=name:raise ValueError('duplicate or invalid dataset name')
        seen.add(name)
    # Small first gets broad coverage quickly; no matrix is excluded on size.
    rows.sort(key=lambda r:(int(r['nnz']) if r['status']=='ok' and r['nnz'] else 2**63,r['dataset']))
    metadata={'index_path':str(a.index.resolve()),'index_sha256':sha(a.index),'exe':str(a.exe.resolve()),'exe_sha256':sha(a.exe),
              'config_sha256':sha(a.config),'gpu_uuid':uuid,'physical_gpu_index':a.gpu_index,'seed':a.seed,'trials':a.trials,'timeout_s':a.timeout,
              'candidates':'Fixed predeclared configs, plus cuSPARSE CSR_ALG1. Each candidate verified against long-double reference for two inputs and once by graph. 3 warmup batches, 9 interleaved trials of 8 graph calls, warm before every timed graph.',
              'order':'Ascending indexed nnz; includes missing inputs and no size cutoff; failures are retained separately. Best-of-fixed shortlist, not best-of-all configurations.',
              'sources':{p:sha(ROOT/p) for p in ['bench/spmv_hybrid_validation.cu','src/spmvPlan.cu','src/spmvPlanDsm.cu','include/spmvPlan.h','include/dsmLongRow.cuh','src/atspmvProduct.cu','scripts/analysis/benchmark_dsm_atbsr_full.py']}}
    meta=a.output/'metadata.json'
    if meta.exists() and json.loads(meta.read_text())!=metadata:raise RuntimeError('metadata mismatch; use a fresh output directory for changed inputs/config/binary')
    if not meta.exists():write_json(meta,metadata);write_json(a.output/'manifest.json',rows)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
    completed=0
    # Wait for a sustained idle period after startup or observed contention;
    # brief gaps between another user's kernels are not an idle GPU.
    require_idle_grace=True
    for i,row in enumerate(rows):
        name=row['dataset'];status_path=a.output/'status'/f'{name}.json'
        if status_path.exists():continue
        path=a.matrix_root/name/(name+'.mtx')
        if row['status']!='ok' or not path.is_file():
            write_json(status_path,{'dataset':name,'status':'missing','index_status':row['status'],'path':str(path)});continue
        idle_since=None
        while True:
            other=apps(uuid)
            if other:
                idle_since=None;require_idle_grace=True
                print('WAIT GPU busy',other,flush=True);time.sleep(15);continue
            if not require_idle_grace:break
            if idle_since is None:idle_since=time.monotonic()
            if time.monotonic()-idle_since>=30:require_idle_grace=False;break
            time.sleep(5)
        result={'dataset':name,'path':str(path.resolve()),'matrix_sha256':sha(path),'index_nnz':int(row['nnz']),'index_rows':int(row['rows'])}
        tmp=a.output/'raw'/f'{name}.csv.tmp';err=a.output/'raw'/f'{name}.log.tmp';start=time.monotonic();contended=[]
        with tmp.open('w') as stdout,err.open('w') as stderr:
            proc=subprocess.Popen([str(a.exe.resolve()),str(path),str(a.config.resolve()),str(a.seed),str(a.trials)],stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            while proc.poll() is None:
                time.sleep(.5)
                try:foreign=[x for x in apps(uuid) if x['pid']!=proc.pid]
                except Exception as e:
                    foreign=[{'pid':-1,'name':'GPU monitor error: '+str(e)}]
                if foreign:
                    contended=foreign;os.killpg(proc.pid,signal.SIGTERM);break
                if proc.poll() is None and time.monotonic()-start>a.timeout:
                    os.killpg(proc.pid,signal.SIGTERM);result['status']='timeout';break
            try:rc=proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
        if contended:
            print('RETRY due to external GPU use',name,contended,flush=True);tmp.unlink(missing_ok=True);err.unlink(missing_ok=True)
            # Not an authoritative result: retry once the GPU is idle.
            while apps(uuid):time.sleep(15)
            # Do not move on: restart this matrix via an explicit process restart.
            return 75
        result['elapsed_s']=round(time.monotonic()-start,3);result['returncode']=rc
        os.replace(tmp,a.output/'raw'/f'{name}.csv');os.replace(err,a.output/'raw'/f'{name}.log')
        if 'status' not in result:
            result['status']='ok' if rc==0 else 'error'
            if rc==0:
                with (a.output/'raw'/f'{name}.csv').open() as f:records=list(csv.DictReader(f))
                if not records or len(records)!=9*a.trials or len({r['backend'] for r in records})!=5:
                    result['status']='invalid_output';result['records']=len(records)
        write_json(status_path,result);completed+=1
        if completed%20==0 or result['status']!='ok':print(f'{i+1}/{len(rows)} {name}: {result["status"]}, new={completed}',flush=True)
        if a.limit and completed>=a.limit:break
    print('DONE new cases',completed,flush=True)
if __name__=='__main__':raise SystemExit(main())
