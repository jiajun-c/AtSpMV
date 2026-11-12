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
parser.add_argument('--actual-vector-model',action='store_true',help='Compare the integrated vector model against the existing model')
parser.add_argument('--model-fallback-tuned',action='store_true',help='Test model selection with tuning outside the model scope')
parser.add_argument('--warm-setup-anchor',action='store_true',help='Warm CUDA with an AtBSR plan before either model')
parser.add_argument('--vector-margin-two-percent',action='store_true',help='Compare a 2%% margin to both existing and ungated models')
parser.add_argument('--alias-identical-models',action='store_true',help='Execute one Plan when model configs match')
parser.add_argument('--alias-identical-graphs',action='store_true',help='Execute one Graph when model configs match')
parser.add_argument('--internal-graph-events',action='store_true',help='Measure with events inside each captured Graph')
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
def stop_holdout(proc):
    # The benchmark starts each GPU child in its own process group, so killing
    # the Python runner's group alone would leave that child running.
    try:children=[int(v) for v in Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()]
    except FileNotFoundError:children=[]
    for child in children:
        try:
            command=Path(f'/proc/{child}/cmdline').read_bytes()
            if str((root/'provenance/spmv_hybrid_validation').resolve()).encode() in command:
                os.killpg(child,signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError):pass
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
try:
    os.kill(runner,signal.SIGSTOP);stopped=True
    print('PAUSED sweep runner; waiting for its current child to finish',flush=True)
    while alive_children():
        if time.monotonic()>=deadline:raise TimeoutError('current sweep child did not finish before deadline')
        time.sleep(.5)
    print('RUNNING independent vector holdout',flush=True)
    if args.actual_vector_model and args.model_fallback_tuned:raise ValueError('conflicting selection modes')
    selection='auto_model_fallback_tuned' if args.model_fallback_tuned else ('auto_vector_model_2pct' if args.vector_margin_two_percent else 'auto_vector_model') if args.actual_vector_model else 'auto_model'
    cmd=[sys.executable,str(root/'provenance/benchmark_plan_selection.py'),'--exe',str(root/'provenance/spmv_hybrid_validation'),
         '--index',str(root/'index.csv'),'--matrix-root','/staff/chengjiajun/workspace/database/suitsparse','--output',str(root),
         '--gpu-index','3','--monitor-resident','--selection',selection,'--counterbalance','--extended-baselines']
    if args.actual_vector_model:cmd.append('--compare-auto-model')
    elif not args.model_fallback_tuned:cmd.append('--vector-candidates')
    if args.warm_setup_anchor:cmd.append('--warm-setup-anchor')
    if args.vector_margin_two_percent:
        if not args.actual_vector_model:raise ValueError('margin comparison requires actual vector model')
        cmd.append('--compare-ungated-vector')
    if args.alias_identical_models:cmd.append('--alias-identical-models')
    if args.alias_identical_graphs:cmd.append('--alias-identical-graphs')
    if args.internal_graph_events:cmd.append('--internal-graph-events')
    while True:
        with (root/'runner.log').open('a') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=process.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                stop_holdout(process)
                raise TimeoutError('holdout deadline reached; completed cases preserved')
        if rc!=75:break
        if time.monotonic()>=deadline:raise TimeoutError('holdout repeatedly interrupted by GPU activity')
    if rc:raise RuntimeError(f'holdout runner exited {rc}')
finally:
    if process is not None and process.poll() is None:
        stop_holdout(process)
    if stopped:
        try:
            if expected in Path(f'/proc/{runner}/cmdline').read_bytes():
                os.kill(runner,signal.SIGCONT);print('RESUMED sweep runner',flush=True)
            else:print('sweep runner PID no longer matches; did not signal',flush=True)
        except (ProcessLookupError,FileNotFoundError):print('sweep runner exited while paused',flush=True)
scripts=('audit_plan_sweep.py','analyze_plan_selection.py') if args.actual_vector_model or args.model_fallback_tuned else ('audit_plan_sweep.py','analyze_plan_selection.py','evaluate_frozen_vector_model.py')
for name in scripts:
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,str(root/'provenance'/name),str(root)],check=True,env=env)
if args.actual_vector_model or args.model_fallback_tuned:
    audit=json.loads((root/'audit.json').read_text())
    complete_success=not audit['failures'] and not audit['problems'] and audit['status_counts']=={'ok':audit['manifest_total']}
else:complete_success=json.loads((root/'frozen-model-evaluation.json').read_text())['complete_success']
if complete_success and (root/'queue.pid').exists():
    # The pre-existing queue is ours and would otherwise repeat this completed
    # evaluation after all other experiments. Keep it if there were failures.
    queued_pid=int((root/'queue.pid').read_text())
    try:
        queued=Path(f'/proc/{queued_pid}/cmdline').read_bytes()
        if b'vector-model-independent-120/run-after-stats.sh' in queued:
            os.killpg(queued_pid,signal.SIGTERM)
    except (ProcessLookupError,FileNotFoundError):pass
print('HOLDOUT complete_success=',complete_success,flush=True)
