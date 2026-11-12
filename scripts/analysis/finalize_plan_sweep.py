#!/usr/bin/env python3
"""Audit and finalize a plan sweep when its recorded cases are complete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--runner-pid',type=int,required=True)
p.add_argument('--command',default='atspmv-counterbalanced-master.sh')
a=p.parse_args();root=a.directory
def alive():
    try:
        status=Path(f'/proc/{a.runner_pid}/status').read_text()
        cmd=Path(f'/proc/{a.runner_pid}/cmdline').read_bytes()
        return 'State:\tZ' not in status and a.command.encode() in cmd
    except OSError:return False
while True:
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if all((root/'status'/f"{item['dataset']}.json").exists() for item in manifest):
            # Allow the runner's own final analyzer to finish first.
            time.sleep(15)
            break
    if not alive():raise SystemExit('runner exited before every manifest entry acquired a status')
    time.sleep(30)
scripts=Path(__file__).resolve().parent
def run(name,*args):
    subprocess.run([sys.executable,str(scripts/name),*[str(v) for v in args]],check=True)
run('audit_plan_sweep.py',root)
metadata=json.loads((root/'metadata.json').read_text())
if metadata['selection'] in ('auto_tuned','auto_fast_tuned'):
    run('audit_tuned_choice.py',root)
    run('collect_plan_tuning.py',root,'--output',root/'tuning.csv')
run('analyze_plan_selection.py',root)
run('analyze_plan_amortization.py',root)
run('plan_speedup_uncertainty.py',root)
audit=json.loads((root/'audit.json').read_text());summary=json.loads((root/'analysis.json').read_text())
result={'suite_complete':not audit['status_counts'].get('pending',0),'status_counts':audit['status_counts'],
        'failures':audit['failures'],'audit_problems':audit['problems'],'selection':summary['selection'],
        'metrics':{k:v for k,v in summary.items() if k.startswith('speedup_')},
        'note':'Completed status includes recorded failures; inspect failures and sample scope. Shared-GPU polling cannot exclude every interference event.'}
tmp=root/'completion.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,root/'completion.json')
print('FINALIZED',result['status_counts'],flush=True)
