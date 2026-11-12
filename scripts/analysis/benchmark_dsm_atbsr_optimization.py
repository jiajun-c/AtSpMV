#!/usr/bin/env python3
"""Isolated correctness and crossover optimization comparison, coordinated with full sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def gpu_apps(uuid):
    r=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
    out=[]
    for line in r.splitlines():
        q=[x.strip() for x in line.split(',',2)]
        if len(q)==3 and q[0]==uuid:out.append((int(q[1]),q[2]))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-index',type=int,required=True);ap.add_argument('--full',type=Path,required=True)
    ap.add_argument('--base-build',type=Path,required=True);ap.add_argument('--opt-build',type=Path,required=True)
    ap.add_argument('--opt-source',type=Path,required=True);ap.add_argument('--matrix-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--auto-names',nargs='+',default=['t60k','c-53','a0nsdsil'])
    ap.add_argument('--skip-crossover',action='store_true')
    ap.add_argument('--case-config-dir',type=Path)
    ap.add_argument('--case-names',nargs='+')
    ap.add_argument('--skip-auto',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
    pause=a.full/'PAUSE';pause.write_text('Optimization confirmation has exclusive use; removed when complete.\n')
    def interrupted(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    metadata={'gpu_uuid':uuid,'device_index':a.gpu_index,'sources':{'optimized_spmvPlan.cu':sha(a.opt_source/'src/spmvPlan.cu'),'optimized_plan_test.cu':sha(a.opt_source/'tests/spmv_plan_test.cu')},
              'executables':{str(p):sha(p) for p in [a.base_build/'spmv_plan_bench',a.opt_build/'spmv_plan_bench',a.opt_build/'spmv_hybrid_validation',a.opt_build/'spmv_plan_test']},
              'test_method':'Exclusive GPU, 30 seconds continuous idle before each externally interrupted run. Each configuration validates two nonuniform inputs and a graph against CPU long double; three randomized seeds, each 21 interleaved timed batches of 8 warm graph calls. No prep cost in CUDA events.',
              'auto_method':'Alternating baseline/optimized automatic tuning runs; compare setup time and actual chosen backend and timing, not end-to-end inference.',
              'auto_names':a.auto_names,'skip_crossover':a.skip_crossover,'skip_auto':a.skip_auto}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    try:
        def wait_idle():
            t=None;last=None
            while True:
                other=gpu_apps(uuid)
                if other:
                    t=None
                    if other!=last:print('WAIT external GPU use',other,flush=True);last=other
                    time.sleep(5);continue
                if t is None:t=time.monotonic()
                if time.monotonic()-t>=30:return
                time.sleep(3)
        wait_idle()
        def run(label,command):
            while True:
                other=gpu_apps(uuid)
                if other:wait_idle()
                out=a.output/(label+'.out');err=a.output/(label+'.log');temp=a.output/(label+'.tmp');temp_err=a.output/(label+'.tmp.log');foreign=[]
                print('RUN',label,flush=True)
                with temp.open('w') as stdout,temp_err.open('w') as stderr:
                    proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
                    started=time.monotonic()
                    try:
                        while proc.poll() is None:
                            time.sleep(.5);other=[p for p in gpu_apps(uuid) if p[0]!=proc.pid]
                            if other:
                                foreign=other;os.killpg(proc.pid,signal.SIGTERM);break
                            if proc.poll() is None and time.monotonic()-started>300:
                                os.killpg(proc.pid,signal.SIGTERM);raise RuntimeError('timeout '+label)
                    finally:
                        try:rc=proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
                if foreign:
                    print('RETRY',label,foreign,flush=True);temp.unlink(missing_ok=True);temp_err.unlink(missing_ok=True);wait_idle();continue
                os.replace(temp,out);os.replace(temp_err,err)
                if rc:raise RuntimeError(f'{label}: exit {rc}; see {err}')
                return
        run('plan-test',[str(a.opt_build/'spmv_plan_test'),'--quick'])
        configurations={
          'c-53':['atbsr,3,1,8,128,1,1024','atbsr,2,1,8,128,1,1024','hybrid_dsm_atbsr,0,0,4,256,1,1024','hybrid_dsm_atbsr,3,0,4,256,4,128','hybrid_dsm_atbsr,2,0,4,256,4,128','hybrid_dsm_atbsr,2,0,4,256,4,256'],
          'rajat20':['atbsr,3,1,8,128,1,1024','atbsr,2,1,8,128,1,1024','hybrid_dsm_atbsr,1,0,4,256,1,1024','hybrid_dsm_atbsr,2,0,4,256,4,128','hybrid_dsm_atbsr,2,0,4,256,4,256'],
          'a0nsdsil':['atbsr,4,1,8,128,1,1024','atbsr,3,1,8,128,1,1024','hybrid_dsm_atbsr,5,0,8,128,1,1024','hybrid_dsm_atbsr,3,0,4,256,4,128']}
        if a.case_config_dir:
            if not a.case_names:raise ValueError('--case-names is required with --case-config-dir')
            configurations={name:(a.case_config_dir/f'{name}-configs.csv').read_text().splitlines() for name in a.case_names}
        for name,lines in ([] if a.skip_crossover else configurations.items()):
            matrix=a.matrix_root/name/(name+'.mtx');cfg=a.output/(name+'-configs.csv');cfg.write_text('\n'.join(lines)+'\n')
            metadata.setdefault('inputs',{})[name]={'path':str(matrix),'sha256':sha(matrix)}
            for seed in (202610281,202610282,202610283):
                run(f'{name}-{seed}',[str(a.opt_build/'spmv_hybrid_validation'),str(matrix),str(cfg),str(seed),'21'])
        cases=[] if a.skip_auto else [(name,mode,rep) for name in a.auto_names for mode in ('baseline','optimized') for rep in range(3)]
        random.Random(20261027).shuffle(cases)
        for name,mode,rep in cases:
            exe=(a.base_build if mode=='baseline' else a.opt_build)/'spmv_plan_bench';matrix=a.matrix_root/name/(name+'.mtx')
            if name not in metadata.setdefault('inputs',{}):metadata['inputs'][name]={'path':str(matrix),'sha256':sha(matrix)}
            run(f'{name}-auto-{mode}-{rep}',[str(exe),str(matrix),'--tune'])
        metadata['complete']=True;(a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print('COMPLETE',flush=True)
    finally:pause.unlink(missing_ok=True)
if __name__=='__main__':main()
