#!/usr/bin/env python3
"""Run a command only on an idle GPU; abort and signal retry on contention."""
import argparse
import os
import signal
import subprocess
import time
from benchmark_dsm_atbsr_full import apps

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu-index', type=int, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
if not a.command:
    p.error('missing command')
uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu_index), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
idle = None
while True:
    other = apps(uuid)
    if other:
        idle = None
        print('WAIT GPU busy', other, flush=True)
        time.sleep(15)
    else:
        if idle is None:
            idle = time.monotonic()
        if time.monotonic()-idle >= 30:
            break
        time.sleep(5)
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = str(a.gpu_index)
proc = subprocess.Popen(a.command, env=env, start_new_session=True)
while proc.poll() is None:
    time.sleep(.5)
    try:
        other = [x for x in apps(uuid) if x['pid'] != proc.pid]
    except Exception:
        other = [{'pid': -1, 'name': 'GPU monitor error'}]
    if other:
        print('RETRY due to external GPU use', other, flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise SystemExit(75)
raise SystemExit(proc.returncode)
