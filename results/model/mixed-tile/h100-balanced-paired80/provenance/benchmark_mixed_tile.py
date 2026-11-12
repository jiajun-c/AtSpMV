#!/usr/bin/env python3
"""Paired mixed-vs-fixed tile experiment; retain errors and monitor GPU use."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from benchmark_dsm_atbsr_full import apps, sha, write_json
from benchmark_plan_selection import gpu_state

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,required=True)
p.add_argument('--index',type=Path,required=True)
p.add_argument('--matrix-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gpu-index',type=int,required=True)
p.add_argument('--trials',type=int,default=9)
p.add_argument('--seed',type=int,default=20261028)
p.add_argument('--timeout',type=int,default=600)
a=p.parse_args()
if a.trials<1 or a.timeout<1:raise ValueError('invalid trials or timeout')
out=a.output
for sub in ('raw','status'): (out/sub).mkdir(parents=True,exist_ok=True)
uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
with a.index.open(newline='') as f: manifest=list(csv.DictReader(f))
if any(Path(r['dataset']).name!=r['dataset'] for r in manifest) or len({r['dataset'] for r in manifest})!=len(manifest):raise ValueError('invalid dataset names')
random.Random(a.seed).shuffle(manifest)
meta={'gpu_uuid':uuid,'exe_sha256':sha(a.exe),'index_sha256':sha(a.index),'runner_sha256':sha(Path(__file__)),'trials':a.trials,'seed':a.seed,'timeout':a.timeout,
      'method':'Two separate processes with opposite construction order; 8 SpMVs per warm Graph and internal event nodes; random interleaved candidates. Includes shared-layout dynamic, specialized, and two balanced mixed kernels, default and twelve fixed AtBSR candidates. Check two input vectors directly and Graph against CPU long-double reference. Setup excluded. Foreign GPU detection is sampled.'}
if (out/'metadata.json').exists():
    if json.loads((out/'metadata.json').read_text())!=meta:raise RuntimeError('metadata mismatch')
    if json.loads((out/'manifest.json').read_text())!=manifest:raise RuntimeError('manifest mismatch')
else:
    write_json(out/'metadata.json',meta);write_json(out/'manifest.json',manifest)
env={k:v for k,v in os.environ.items() if not k.startswith('ATSPMV_')};env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
idle_since=None;grace=True
for i,row in enumerate(manifest):
    name=row['dataset'];status=out/'status'/f'{name}.json'
    if status.exists():continue
    path=a.matrix_root/name/(name+'.mtx')
    if row['status']!='ok' or not path.is_file():write_json(status,{'status':'missing','dataset':name});continue
    while True:
        foreign=apps(uuid);state=gpu_state(a.gpu_index)
        if foreign or state['free_memory_mib']<1024 or (grace and state['utilization_percent']):
            idle_since=None;grace=True;time.sleep(5);continue
        if not grace:break
        if idle_since is None:idle_since=time.monotonic()
        if time.monotonic()-idle_since>=30:grace=False;break
        time.sleep(5)
    result={'dataset':name,'matrix_sha256':sha(path),'gpu_before':gpu_state(a.gpu_index)}
    orders=[0,1]
    if int(hashlib.sha256(f'{a.seed}:{name}'.encode()).hexdigest()[:2],16)%2:orders.reverse()
    records=[];contested=False;timed_out=False;rc=0;started=time.monotonic()
    log=out/'raw'/f'{name}.log.tmp'
    with log.open('w') as err:
        for ordinal,order in enumerate(orders):
            tmp=out/'raw'/f'{name}.csv.tmp'
            err.write(f'ORDER,{order}\n');err.flush()
            with tmp.open('w') as output:
                proc=subprocess.Popen([str(a.exe.resolve()),str(path.resolve()),str(a.seed+ordinal),str(a.trials),str(order)],stdout=output,stderr=err,env=env,start_new_session=True)
                while proc.poll() is None:
                    time.sleep(.25)
                    foreign=[x for x in apps(uuid) if x['pid']!=proc.pid]
                    if foreign or time.monotonic()-started>a.timeout:
                        contested=bool(foreign);timed_out=not contested
                        os.killpg(proc.pid,signal.SIGTERM);break
                try:rc=proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
            if contested or rc:break
            with tmp.open(newline='') as f:part=list(csv.DictReader(f))
            for r in part:r['order']=str(order)
            records.extend(part)
    if contested:
        print('RETRY due to external GPU use',name,foreign,flush=True);tmp.unlink(missing_ok=True);log.unlink(missing_ok=True);grace=True
        raise SystemExit(75)
    os.replace(log,out/'raw'/f'{name}.log')
    if rc==0:
        expected={(c,t,o) for c in ['mixed','mixed_specialized','mixed_bal4','mixed_bal16','default']+[f'm{m}_b{b}' for m in (1,2,4,8,16,32) for b in (0,1)] for t in range(a.trials) for o in ('0','1')}
        actual={(r['candidate'],int(r['trial']),r['order']) for r in records}
        if len(records)!=len(expected) or actual!=expected or any(not 0<float(r['us'])<float('inf') for r in records):result['status']='invalid_output'
        else:result['status']='ok'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        os.replace(tmp,out/'raw'/f'{name}.csv')
    else:
        result['status']='timeout' if timed_out else 'error';tmp.unlink(missing_ok=True)
    result.update(returncode=rc,elapsed_s=round(time.monotonic()-started,3),gpu_after=gpu_state(a.gpu_index));write_json(status,result)
    print(f'{i+1}/{len(manifest)} {name}: {result["status"]}',flush=True)
print('DONE',flush=True)
