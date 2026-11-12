#!/usr/bin/env python3
"""Audit the suite after its resumable runner stops."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from finalize_dsm_atbsr_full import alive

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--runner-pid',type=int,required=True);a=p.parse_args()
    while alive(a.runner_pid):time.sleep(30)
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_dsm_atbsr_full.py')),str(a.root)],check=True)
if __name__=='__main__':main()
