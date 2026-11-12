#!/usr/bin/env python3
"""Retest one oversized matrix with an explicitly reduced candidate set."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from finalize_dsm_atbsr_full import alive
from benchmark_dsm_atbsr_optimization import gpu_apps

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner-pid',type=int,required=True);p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    while alive(a.runner_pid):time.sleep(30)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
    cfg=a.output/'configs.csv';out=a.output/'result.csv';err=a.output/'result.log'
    while True:
        idle=None
        while True:
            others=gpu_apps(uuid)
            if others:idle=None;print('WAIT GPU',others,flush=True);time.sleep(10);continue
            if idle is None:idle=time.monotonic()
            if time.monotonic()-idle>=30:break
            time.sleep(3)
        tmp=out.with_name(out.name+'.tmp');tmp_err=err.with_name(err.name+'.tmp');foreign=[];timed_out=False
        with tmp.open('w') as f,tmp_err.open('w') as e:
            proc=subprocess.Popen([str(a.exe.resolve()),str(a.matrix.resolve()),str(cfg.resolve()),'202610271','9'],stdout=f,stderr=e,env=env,start_new_session=True);start=time.monotonic()
            try:
                while proc.poll() is None:
                    time.sleep(.5);foreign=[x for x in gpu_apps(uuid) if x[0]!=proc.pid]
                    if foreign:os.killpg(proc.pid,signal.SIGTERM);break
                    if proc.poll() is None and time.monotonic()-start>900:os.killpg(proc.pid,signal.SIGTERM);timed_out=True;break
            finally:
                try:rc=proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
        if foreign:
            print('RETRY external GPU use',foreign,flush=True);tmp.unlink(missing_ok=True);tmp_err.unlink(missing_ok=True);continue
        os.replace(tmp,out);os.replace(tmp_err,err)
        result={'dataset':a.matrix.stem,'status':'timeout' if timed_out else ('ok' if rc==0 else 'error'),'returncode':rc,'elapsed_s':time.monotonic()-start,'gpu_uuid':uuid,'matrix_sha256':digest(a.matrix),'config_sha256':digest(cfg),'exe_sha256':digest(a.exe),'excluded_candidate':'atbsr,0,1,8,128,1,1024','note':'Reduced candidate set, must not be merged into the uniform full-sweep aggregate.'}
        if rc==0:
            with out.open() as f:records=list(csv.DictReader(f))
            if len(records)!=8*9:result['status']='invalid_output';result['records']=len(records)
        (a.output/'status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return
if __name__=='__main__':main()
