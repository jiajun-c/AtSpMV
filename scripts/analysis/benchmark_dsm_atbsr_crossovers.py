#!/usr/bin/env python3
"""After 2000 valid full-suite results, retest frozen crossover candidates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from benchmark_dsm_atbsr_optimization import gpu_apps

ROOT=Path(__file__).resolve().parents[2]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--matrix-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu-index',type=int,required=True);ap.add_argument('--minimum',type=int,default=2000)
    ap.add_argument('--selection',choices=('crossover','medium-tail','large-wins'),default='crossover')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
    def interrupt(s,f):raise KeyboardInterrupt('signal '+str(s))
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    while True:
        ok=0
        for p in (a.full/'status').glob('*.json'):
            try:ok+=json.loads(p.read_text()).get('status')=='ok'
            except (OSError,ValueError):continue
        if ok>=a.minimum:break
        print(f'WAIT full-suite successful={ok}, required={a.minimum}',flush=True);time.sleep(60)
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/analyze_dsm_atbsr_full.py'),str(a.full)],stdout=subprocess.DEVNULL,check=True)
    rows=list(csv.DictReader((a.full/'analysis.csv').open()))
    long=[x for x in rows if int(x['max_row_nnz'])>=1024];short=[x for x in rows if int(x['max_row_nnz'])<1024]
    candidates=[];seen=set()
    def take(items,count,label):
        for row in items:
            if row['dataset'] in seen:continue
            seen.add(row['dataset']);candidates.append({'dataset':row['dataset'],'reason':label,'initial_speedup':float(row['speedup_vs_best_nonhybrid']),'max_row_nnz':int(row['max_row_nnz'])})
            count-=1
            if count==0:break
    if a.selection=='crossover':
        take(sorted((x for x in long if float(x['speedup_vs_best_nonhybrid'])>1),key=lambda x:float(x['speedup_vs_best_nonhybrid']),reverse=True),8,'long-row wins')
        take(sorted(long,key=lambda x:abs(float(x['speedup_vs_best_nonhybrid'])-1)),8,'long-row near crossover')
        take(sorted((x for x in short if float(x['speedup_vs_best_nonhybrid'])>1),key=lambda x:float(x['speedup_vs_best_nonhybrid']),reverse=True),8,'no-long-row wins')
        take(sorted(short,key=lambda x:abs(float(x['speedup_vs_best_nonhybrid'])-1)),8,'no-long-row near crossover')
        take([x for x in short if x['dataset']=='Pd'],1,'previous no-long-row win')
    elif a.selection=='medium-tail':
        medium=[x for x in short if int(x['max_row_nnz'])>=128]
        take(sorted(medium,key=lambda x:float(x['speedup_vs_best_nonhybrid'])),8,'medium-row worst first pass')
        medium.sort(key=lambda x:x['dataset']);random.Random(20261028).shuffle(medium)
        take(medium,8,'medium-row random from remainder')
        take(sorted(long,key=lambda x:float(x['speedup_vs_best_nonhybrid'])),8,'long-row worst first pass')
    else:
        large=[x for x in rows if int(x['nnz'])>=1000000 and float(x['speedup_vs_best_nonhybrid'])>1]
        take(sorted(large,key=lambda x:float(x['speedup_vs_best_nonhybrid']),reverse=True),8,'large-matrix wins')
    saved=a.output/'manifest.json'
    if saved.exists():
        # Preserve the original frozen cases on restart, even if the main
        # sweep has advanced or completed since that selection.
        candidates=json.loads(saved.read_text())['cases']
    if not candidates:raise RuntimeError('no eligible cases')
    pause=a.full/'PAUSE';pause.write_text('Isolated crossover remeasurement is using GPU; removed on completion.\n')
    results=[];failures=[]
    try:
        def wait_idle():
            since=None
            while True:
                other=gpu_apps(uuid)
                if other:since=None;print('WAIT GPU',other,flush=True);time.sleep(5);continue
                if since is None:since=time.monotonic()
                if time.monotonic()-since>=30:return
                time.sleep(3)
        wait_idle()
        lines=(a.full/'configs.csv').read_text().splitlines()
        # The first pass is intentionally bounded; make the crossover check
        # stronger by evaluating all standard AtBSR tile/mode pairs and a
        # wider set of non-hybrid split-row choices.
        for tile in range(6):
            for balanced in (0,1):lines.append(f'atbsr,{tile},{balanced},8,128,1,1024')
        for kind in ('csr_dsm','csr_two_stage'):
            for parts,threads,group,threshold in [(2,256,1,1024),(4,256,1,1024),(8,128,1,1024),(8,256,4,1024),(4,256,4,128),(8,128,1,4096)]:
                lines.append(f'{kind},-1,-1,{parts},{threads},{group},{threshold}')
        lines += ['csr_two_stage,-1,-1,16,128,1,1024','csr_two_stage,-1,-1,32,128,1,1024',
                  'hybrid_dsm_atbsr,2,0,4,256,4,128','hybrid_dsm_atbsr,3,0,4,256,4,128',
                  'hybrid_dsm_atbsr,2,0,4,256,4,256','hybrid_dsm_atbsr,2,0,8,256,4,128',
                  'hybrid_dsm_atbsr,3,0,8,256,4,128']
        lines=list(dict.fromkeys(lines))
        config=a.output/'configs.csv';config.write_text('\n'.join(lines)+'\n')
        descriptions={'crossover':'Choose up to 8 highest long-row wins, 8 other long-row cases nearest crossover, 8 highest no-long-row wins, 8 other no-long-row cases nearest crossover, and Pd if available.',
                      'medium-tail':'Choose 8 slowest medium-row cases, 8 random remaining medium-row cases (seed 20261028), and 8 slowest long-row cases.',
                      'large-wins':'Choose up to 8 highest first-pass wins among matrices with at least one million nonzeros.'}
        meta={'selection':descriptions[a.selection]+' Selected from the first pass, not a random holdout.','selection_mode':a.selection,'cases':candidates,
              'gpu_uuid':uuid,'exe':str(a.exe.resolve()),'exe_sha256':digest(a.exe),'config_sha256':digest(config),'seeds':[202610281,202610282,202610283],
              'method':'3 randomized orders, each 21 timed warm Graph batches of 8 calls. Every config has direct and Graph CPU reference check. Non-hybrid comparison includes all 12 AtBSR tile/mode pairs, an extended CSR shortlist and cuSPARSE CSR_ALG1. New candidates lower thresholds and use different tile/cluster/thread group; interpret as a combined config.'}
        (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
        for case in candidates:
            name=case['dataset'];path=a.matrix_root/name/(name+'.mtx');case['sha256']=digest(path)
            failed=False
            for seed in meta['seeds']:
                target=a.output/f'{name}-{seed}.csv';log=a.output/f'{name}-{seed}.log'
                if target.exists():
                    with target.open() as existing:
                        records=list(csv.DictReader(existing))
                    counts={}
                    for record in records:
                        key=(record.get('backend'),record.get('config'))
                        counts[key]=counts.get(key,0)+1
                    if counts and all(n==21 for n in counts.values()) and len(counts)==len(lines)+1:
                        results.append({'dataset':name,'seed':seed,'file_sha256':digest(target),'resumed':True})
                        continue
                    target.unlink()
                while True:
                    if gpu_apps(uuid):wait_idle()
                    tmp=target.with_suffix('.csv.tmp');tmp_log=log.with_suffix('.log.tmp');foreign=[];timed_out=False
                    print('RUN',name,seed,flush=True)
                    with tmp.open('w') as f,tmp_log.open('w') as e:
                        proc=subprocess.Popen([str(a.exe.resolve()),str(path),str(config.resolve()),str(seed),'21'],stdout=f,stderr=e,env=env,start_new_session=True);start=time.monotonic()
                        try:
                            while proc.poll() is None:
                                time.sleep(.5);foreign=[x for x in gpu_apps(uuid) if x[0]!=proc.pid]
                                if foreign:os.killpg(proc.pid,signal.SIGTERM);break
                                if proc.poll() is None and time.monotonic()-start>300:
                                    os.killpg(proc.pid,signal.SIGTERM);timed_out=True;break
                        finally:
                            try:rc=proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
                    if foreign:
                        tmp.unlink(missing_ok=True);tmp_log.unlink(missing_ok=True);print('RETRY',name,foreign,flush=True);wait_idle();continue
                    os.replace(tmp,target);os.replace(tmp_log,log)
                    if rc or timed_out:
                        failures.append({'dataset':name,'seed':seed,'status':'timeout' if timed_out else 'error','returncode':rc,'log':str(log)})
                        failed=True;print('FAILED',name,seed,rc,flush=True);break
                    results.append({'dataset':name,'seed':seed,'file_sha256':digest(target),'seconds':time.monotonic()-start});break
                if failed:break
        meta['results']=results;meta['failures']=failures;meta['complete']=True;(a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print('COMPLETE',flush=True)
    finally:pause.unlink(missing_ok=True)
if __name__=='__main__':main()
