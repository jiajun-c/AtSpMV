#!/usr/bin/env python3
"""Summarize setup amortization once the queued benchmark exits."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

def alive(pid, command):
    try:
        state=Path(f'/proc/{pid}/status').read_text()
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return 'State:\tZ' not in state and command.encode() in cmd
    except OSError:
        return False

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--command',default='atspmv-counterbalanced-master.sh',help='expected process command substring, to avoid PID reuse')
p.add_argument('directories',nargs='+',type=Path)
a=p.parse_args()
while alive(a.pid,a.command):time.sleep(30)
script=Path(__file__).with_name('analyze_plan_amortization.py')
for directory in a.directories:
    if (directory/'metadata.json').exists() and (directory/'manifest.json').exists():
        subprocess.run([sys.executable,str(script),str(directory)],check=True)
    else:print(f'skipped {directory}: benchmark did not start',flush=True)
