#!/usr/bin/env python3
"""After the suite runner exits, atomically refresh its report and graph."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
def alive(pid):
    try:
        state=Path(f'/proc/{pid}/status').read_text()
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return 'State:\tZ' not in state and b'atspmv-full-suite-wrapper.sh' in cmd
    except OSError:return False

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('--runner-pid',type=int,required=True);a=ap.parse_args()
    while alive(a.runner_pid):time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/analyze_dsm_atbsr_full.py'),str(a.root)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/analysis/plot_dsm_atbsr_full.py'),str(a.root)],check=True)
    data=json.loads((a.root/'analysis.json').read_text());data['suite_complete']=data['status_counts'].get('pending',0)==0
    tmp=a.root/'completion.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,a.root/'completion.json')
    print('FINAL complete=',data['suite_complete'],'counts=',data['status_counts'],flush=True)
if __name__=='__main__':main()
