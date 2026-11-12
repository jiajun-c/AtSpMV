#!/usr/bin/env python3
"""Temporarily suspend this project's older GPU-3 queue, then resume it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=('pause','resume'))
p.add_argument('--record',type=Path,required=True)
a=p.parse_args()
patterns=(
    'shared-row-sort-h100/run-correctness-after.sh',
    'shared-row-sort-paired80/run-after-correctness.sh',
    'short-row-cache-h100/run-after-paired.sh',
    'shared-tile-stats-paired80/run-after-short-cache.sh',
    'vector-model-full-2247/run-after-80.sh',
    'vector-model-full-2247/finalize-uncertainty.sh',
    'unified-policy-full-2794/run-after-vector-full.sh',
    'shared-atbsr-paired80/run-after-policy.sh',
)
def details(pid):
    root=Path('/proc')/str(pid)
    try:
        if root.stat().st_uid!=os.getuid(): return None
        cmd=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        stat=(root/'stat').read_text().split()
        return {'pid':pid,'starttime':stat[21],'cmd':cmd}
    except (OSError,IndexError): return None
def children(pid):
    r=subprocess.run(['ps','-o','pid=','--ppid',str(pid)],capture_output=True,text=True)
    return [int(s) for s in r.stdout.split()]
def resume(records):
    for r in reversed(records):
        cur=details(r['pid'])
        if cur and cur['starttime']==r['starttime'] and cur['cmd']==r['cmd']:
            os.kill(r['pid'],signal.SIGCONT)
if a.action=='pause':
    records=[]
    try:
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            r=details(int(path.name))
            if not r or not any(pattern in r['cmd'] for pattern in patterns): continue
            if not r['cmd'].startswith('bash '): continue
            os.kill(r['pid'],signal.SIGSTOP);records.append(r)
            for child in children(r['pid']):
                c=details(child)
                if c and 'shared-row-sort-h100/provenance/run_on_idle_gpu.py' in c['cmd']:
                    os.kill(child,signal.SIGSTOP);records.append(c)
                    if children(child): raise RuntimeError('legacy GPU job already started; retry after it exits')
        a.record.parent.mkdir(parents=True,exist_ok=True)
        a.record.write_text(json.dumps({'paused':records},indent=2)+'\n')
        cmd=[sys.executable,str(Path(__file__).resolve()),'resume','--record',str(a.record.resolve())]
        try:
            subprocess.run(['tmux','new-session','-d','-s','atspmv_resume_legacy',*cmd],check=True)
        except Exception:
            resume(records);raise
        print(f'suspended {len(records)} owned queue processes; automatic resume scheduled')
    except Exception:
        resume(records);raise
else:
    records=json.loads(a.record.read_text())['paused']
    while any(subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0
              for name in ('atspmv_v2_queued','atspmv_medium_queued')):
        time.sleep(30)
    resume(records)
    a.record.with_name('legacy-resumed.json').write_text(json.dumps({'resumed':records},indent=2)+'\n')
