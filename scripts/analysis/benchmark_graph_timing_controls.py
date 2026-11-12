#!/usr/bin/env python3
"""Run a small 2x2 Graph/clock control on an otherwise idle H100."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

import pynvml

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--gpu-index',type=int,required=True)
args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
names=('685_bus','fe_sphere','c-66b','bloweybq')
matrix_root=Path('/staff/chengjiajun/workspace/database/suitsparse')
forward=['atbsr,-1,-1,8,128,1,1024','auto_model,-1,-1,8,128,1,1024','auto_vector_model,-1,-1,8,128,1,1024','auto_vector_model_2pct,-1,-1,8,128,1,1024']
(out/'forward.csv').write_text('\n'.join(forward)+'\n')
(out/'reverse.csv').write_text('\n'.join([forward[0]]+list(reversed(forward[1:])))+'\n')
pynvml.nvmlInit();gpu=pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
def active():return {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={'method':'Four selected cases known to have equal vector configurations; random order over two Graph-sharing modes and two event placements, each with two construction orders, 21 samples. Same Plan, optionally same Graph. Poll GPU compute PIDs at 250 ms; polling does not exclude brief activity, clock drift or non-compute interference.',
      'gpu_uuid':pynvml.nvmlDeviceGetUUID(gpu),'exe_sha256':sha(args.exe),'seed':20261028,
      'matrices':{n:sha(matrix_root/n/(n+'.mtx')) for n in names}}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
work=[(n,shared,internal,order) for n in names for shared in (False,True) for internal in (False,True) for order in ('forward','reverse')]
random.Random(20261028).shuffle(work);envbase=os.environ.copy();envbase['CUDA_VISIBLE_DEVICES']=str(args.gpu_index);envbase['ATSPMV_BENCH_ALIAS_IDENTICAL_MODELS']='1'
for name,shared,internal,order in work:
    label=f'{name}-graph{int(shared)}-event{int(internal)}-{order}'
    if active():raise RuntimeError(f'{label}: GPU already has another compute process')
    env=envbase.copy()
    for enabled,var in ((shared,'ATSPMV_BENCH_ALIAS_IDENTICAL_GRAPHS'),(internal,'ATSPMV_BENCH_INTERNAL_GRAPH_EVENTS')):
        if enabled:env[var]='1'
        else:env.pop(var,None)
    matrix=matrix_root/name/(name+'.mtx')
    with (out/(label+'.csv.tmp')).open('w') as f,(out/(label+'.log.tmp')).open('w') as e:
        proc=subprocess.Popen([str(args.exe.resolve()),str(matrix),str(out/(order+'.csv')),'20261028','21'],env=env,stdout=f,stderr=e,start_new_session=True)
        while proc.poll() is None:
            time.sleep(.25);other=active()-{proc.pid}
            if other:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise RuntimeError(f'{label}: interference {other}; partial files preserved')
        if proc.returncode:raise RuntimeError(f'{label}: benchmark exited {proc.returncode}')
    for ext in ('csv','log'):(out/(label+'.'+ext+'.tmp')).replace(out/(label+'.'+ext))
    rows=list(csv.DictReader((out/(label+'.csv')).open()))
    values={}
    for backend in ('auto_vector_model','auto_vector_model_2pct'):
        samples=[float(r['us']) for r in rows if r['backend']==backend]
        cfg={r['config'] for r in rows if r['backend']==backend}
        if len(samples)!=21 or len(cfg)!=1:raise ValueError(f'{label}: invalid records')
        values[backend]=(next(iter(cfg)),statistics.median(samples))
    if values['auto_vector_model'][0]!=values['auto_vector_model_2pct'][0]:raise ValueError(f'{label}: model configurations differ')
    log=(out/(label+'.log')).read_text()
    if 'PLAN_ALIAS,auto_vector_model_2pct,shared' not in log or ('TIMING_MODE,internal_graph_events' in log)!=internal or ('GRAPH_ALIAS,auto_vector_model_2pct,' in log)!=shared:raise ValueError(f'{label}: mode markers mismatch')
    print(label,'ratio',values['auto_vector_model'][1]/values['auto_vector_model_2pct'][1],flush=True)
