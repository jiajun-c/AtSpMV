#!/usr/bin/env python3
"""Temporarily yield our sweep runner to the frozen vector holdout, then resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sweep-runner-pid',type=int,required=True)
parser.add_argument('--deadline-seconds',type=int,default=900)
parser.add_argument('--directory',type=Path,required=True)
args=parser.parse_args();root=args.directory
runner=args.sweep_runner_pid
expected=b'plan-tune-2247-v4/provenance/benchmark_plan_selection.py'
cmdline=Path(f'/proc/{runner}/cmdline').read_bytes()
if expected not in cmdline:raise SystemExit('runner PID does not match expected active sweep')
deadline=time.monotonic()+args.deadline_seconds
stopped=False;process=None
def interrupt(signum,frame):raise KeyboardInterrupt(f'interrupted by signal {signum}')
signal.signal(signal.SIGTERM,interrupt)
def alive_children():
    child_file=Path(f'/proc/{runner}/task/{runner}/children')
    try:pids=[int(v) for v in child_file.read_text().split()]
    except FileNotFoundError:return []
    result=[]
    for pid in pids:
        try:
            state=next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
            if not state.split()[1].startswith('Z'):result.append(pid)
        except (FileNotFoundError,StopIteration):pass
    return result
try:
    os.kill(runner,signal.SIGSTOP);stopped=True
    print('PAUSED sweep runner; waiting for its current child to finish',flush=True)
    while alive_children():
        if time.monotonic()>=deadline:raise TimeoutError('current sweep child did not finish before deadline')
        time.sleep(.5)
    print('RUNNING independent vector holdout',flush=True)
    cmd=[sys.executable,str(root/'provenance/benchmark_plan_selection.py'),'--exe',str(root/'provenance/spmv_hybrid_validation'),
         '--index',str(root/'index.csv'),'--matrix-root','/staff/chengjiajun/workspace/database/suitsparse','--output',str(root),
         '--gpu-index','3','--monitor-resident','--selection','auto_model','--counterbalance','--extended-baselines','--vector-candidates']
    while True:
        with (root/'runner.log').open('a') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=process.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('holdout deadline reached; completed cases preserved')
        if rc!=75:break
        if time.monotonic()>=deadline:raise TimeoutError('holdout repeatedly interrupted by GPU activity')
    if rc:raise RuntimeError(f'holdout runner exited {rc}')
finally:
    if process is not None and process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        except ProcessLookupError:pass
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    if stopped:
        try:
            if expected in Path(f'/proc/{runner}/cmdline').read_bytes():
                os.kill(runner,signal.SIGCONT);print('RESUMED sweep runner',flush=True)
            else:print('sweep runner PID no longer matches; did not signal',flush=True)
        except (ProcessLookupError,FileNotFoundError):print('sweep runner exited while paused',flush=True)
for name in ('audit_plan_sweep.py','analyze_plan_selection.py','evaluate_frozen_vector_model.py'):
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,str(root/'provenance'/name),str(root)],check=True,env=env)
report=json.loads((root/'frozen-model-evaluation.json').read_text())
if report['complete_success']:
    # The pre-existing queue is ours and would otherwise repeat this completed
    # evaluation after all other experiments. Keep it if there were failures.
    queued_pid=int((root/'queue.pid').read_text())
    try:
        queued=Path(f'/proc/{queued_pid}/cmdline').read_bytes()
        if b'vector-model-independent-120/run-after-stats.sh' in queued:
            os.killpg(queued_pid,signal.SIGTERM)
    except (ProcessLookupError,FileNotFoundError):pass
print('HOLDOUT complete_success=',report['complete_success'],flush=True)
