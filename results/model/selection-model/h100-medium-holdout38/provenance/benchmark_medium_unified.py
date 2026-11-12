#!/usr/bin/env python3
"""Measure a frozen medium-row replay on H100 with two construction orders."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--replay', type=Path, required=True)
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--matrix-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--idle-helper', type=Path, required=True)
p.add_argument('--expected-hashes', type=Path, required=True)
p.add_argument('--gpu-index', type=int, required=True)
p.add_argument('--case')
p.add_argument('--trials', type=int, default=9)
a = p.parse_args()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def save(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,path)

with a.replay.open(newline='') as f:
    cases = {r['dataset']:r for r in csv.DictReader(f) if r['split']=='holdout'}
if len(cases) != 38: raise ValueError('unexpected holdout size')
expected_hashes = json.loads(a.expected_hashes.read_text())
if set(expected_hashes) != set(cases): raise ValueError('matrix hash manifest mismatch')
out = a.output
for sub in ('status','raw','configs'): (out/sub).mkdir(parents=True,exist_ok=True)
if a.case:
    row = cases[a.case]
    name = row['dataset']
    matrix = a.matrix_root/name/(name+'.mtx')
    matrix_hash = sha(matrix)
    if matrix_hash != expected_hashes[name]: raise ValueError(f'{name}: matrix changed since historical tuning')
    lines = ['atbsr,-1,-1,8,128,1,1024','auto_vector_model_v2,-1,-1,8,128,1,1024',
             f'hybrid_dsm_atbsr,{int(row["medium_tile"])},0,{int(row["medium_parts"])},256,4,128']
    collected = []
    log = out/'raw'/(name+'.log.tmp')
    env = os.environ.copy(); env['ATSPMV_BENCH_INTERNAL_GRAPH_EVENTS'] = '1'
    env.pop('ATSPMV_BENCH_ALIAS_IDENTICAL_MODELS',None);env.pop('ATSPMV_BENCH_ALIAS_IDENTICAL_GRAPHS',None)
    orders = [('baseline_first',lines),('candidate_first',[lines[0],*reversed(lines[1:])])]
    if int(hashlib.sha256(name.encode()).hexdigest()[:2],16)%2: orders.reverse()
    with log.open('w') as err:
        for label, configs in orders:
            config = out/'configs'/(name+'-'+label+'.csv')
            config.write_text('\n'.join(configs)+'\n')
            err.write('BENCH_ORDER,'+label+'\n');err.flush()
            proc = subprocess.run([str(a.exe.resolve()),str(matrix.resolve()),str(config.resolve()),'20261029',str(a.trials)],
                                  capture_output=True,text=True,env=env,timeout=900)
            err.write(proc.stderr);err.flush()
            if proc.returncode: raise RuntimeError(f'{name}/{label}: exit {proc.returncode}')
            samples = list(csv.DictReader(proc.stdout.splitlines()))
            expected = {(backend,i) for backend in ('atbsr','auto_vector_model_v2','hybrid_dsm_atbsr','cusparse') for i in range(a.trials)}
            if {(r['backend'],int(r['trial'])) for r in samples} != expected or len(samples) != len(expected):
                raise ValueError(f'{name}/{label}: incomplete measurements')
            baseline_configs = {r['config'] for r in samples if r['backend']=='auto_vector_model_v2'}
            expected_config = json.loads(row['existing_config'])
            if len(baseline_configs) != 1:
                raise ValueError(f'{name}/{label}: inconsistent baseline configurations')
            actual_config = next(iter(baseline_configs)).split('/')
            if expected_config is None:
                if actual_config[0] != 'atbsr': raise ValueError(f'{name}/{label}: baseline replay mismatch')
            else:
                expected_text = [str(x) for x in expected_config[:-1]] + ['vw='+str(expected_config[-1])]
                if actual_config != expected_text: raise ValueError(f'{name}/{label}: baseline replay mismatch')
            for r in samples:
                t=float(r['us'])
                if not math.isfinite(t) or t<=0: raise ValueError('invalid timing')
                r['order'] = label
            collected.extend(samples)
    if sha(matrix) != matrix_hash: raise ValueError(f'{name}: matrix changed during measurement')
    raw = out/'raw'/(name+'.csv.tmp')
    with raw.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['backend','config','trial','us','order']);w.writeheader();w.writerows(collected)
    os.replace(raw,out/'raw'/(name+'.csv'));os.replace(log,out/'raw'/(name+'.log'))
    save(out/'status'/(name+'.json'),{'dataset':name,'status':'ok','matrix_sha256':matrix_hash})
    sys.exit(0)
meta={'replay_sha256':sha(a.replay),'exe_sha256':sha(a.exe),'runner_sha256':sha(Path(__file__)),
      'idle_helper_sha256':sha(a.idle_helper),'matrix_manifest_sha256':sha(a.expected_hashes),
      'gpu_index':a.gpu_index,'trials':a.trials}
if (out/'metadata.json').exists() and json.loads((out/'metadata.json').read_text()) != meta:
    raise ValueError('metadata mismatch')
save(out/'metadata.json',meta)
for i,name in enumerate(sorted(cases),1):
    status=out/'status'/(name+'.json')
    if status.exists(): continue
    command=[sys.executable,str(a.idle_helper.resolve()),'--gpu-index',str(a.gpu_index),sys.executable,str(Path(__file__).resolve()),
             '--replay',str(a.replay.resolve()),'--exe',str(a.exe.resolve()),'--matrix-root',str(a.matrix_root.resolve()),
             '--output',str(out.resolve()),'--idle-helper',str(a.idle_helper.resolve()),'--gpu-index',str(a.gpu_index),
             '--expected-hashes',str(a.expected_hashes.resolve()),
             '--trials',str(a.trials),'--case',name]
    while True:
        rc=subprocess.run(command).returncode
        if rc != 75: break
    if rc != 0: save(status,{'dataset':name,'status':'error','returncode':rc})
    print(f'{i}/{len(cases)} {name}: '+json.loads(status.read_text())['status'],flush=True)
print('benchmark completed',flush=True)
