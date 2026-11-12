#!/usr/bin/env python3
"""Run isolated plan correctness and sanitizers; coordinate with the full suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from benchmark_dsm_atbsr_optimization import gpu_apps

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def descendant(pid,root):
    for _ in range(32):
        if pid==root:return True
        try:
            fields=Path(f'/proc/{pid}/status').read_text().splitlines()
            ppid=int(next(x for x in fields if x.startswith('PPid:')).split()[1])
        except (OSError,StopIteration,ValueError):return False
        if ppid<=1 or ppid==pid:return False
        pid=ppid
    return False

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-index',type=int,required=True);ap.add_argument('--full',type=Path,required=True)
    ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu_index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    pause=a.full/'PAUSE';pause.write_text('Isolated optimized Plan correctness/sanitizer validation.\n')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu_index)
    def interrupt(s,f):raise KeyboardInterrupt('signal '+str(s))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    results={};san='/usr/local/cuda-13.0/bin/compute-sanitizer'
    def wait_idle():
        since=None
        while True:
            found=gpu_apps(uuid)
            if found:print('WAIT',found,flush=True);since=None;time.sleep(5);continue
            if since is None:since=time.monotonic()
            if time.monotonic()-since>=30:return
            time.sleep(3)
    try:
        wait_idle()
        for name,cmd in [('direct',[str(a.exe),'--quick']),('memcheck',[san,'--tool','memcheck','--leak-check','full','--error-exitcode','1',str(a.exe),'--quick']),('synccheck',[san,'--tool','synccheck','--error-exitcode','1',str(a.exe),'--quick']),('racecheck',[san,'--tool','racecheck','--error-exitcode','1',str(a.exe),'--quick'])]:
            while True:
                if gpu_apps(uuid):wait_idle()
                print('RUN',name,flush=True);tmp=a.output/(name+'.tmp');log=a.output/(name+'.log');foreign=[]
                with tmp.open('w') as f:
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);start=time.monotonic()
                    try:
                        while proc.poll() is None:
                            time.sleep(.5)
                            foreign=[x for x in gpu_apps(uuid) if not descendant(x[0],proc.pid)]
                            if foreign:os.killpg(proc.pid,signal.SIGTERM);break
                            if proc.poll() is None and time.monotonic()-start>300:os.killpg(proc.pid,signal.SIGTERM);raise RuntimeError('timeout '+name)
                    finally:
                        try:rc=proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
                if foreign:print('RETRY',name,foreign,flush=True);tmp.unlink(missing_ok=True);wait_idle();continue
                os.replace(tmp,log)
                if rc:raise RuntimeError(f'{name} failed: {rc}')
                results[name]={'returncode':rc,'log':str(log),'sha256':digest(log)};break
        source=Path('src/spmvPlan.cu');tests=Path('tests/spmv_plan_test.cu')
        obj={'gpu_uuid':uuid,'executable':str(a.exe),'executable_sha256':digest(a.exe),'source_sha256':digest(source),'test_source_sha256':digest(tests),'results':results}
        (a.output/'validation.json').write_text(json.dumps(obj,indent=2)+'\n');print('COMPLETE',flush=True)
    finally:pause.unlink(missing_ok=True)
if __name__=='__main__':main()
