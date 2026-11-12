#!/usr/bin/env python3
"""Compare frozen fixed-512 and adaptive-thread Plan binaries in both orders."""
import argparse
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
for name in ('manifest','factors','output','fixed','adaptive'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--resume',action='store_true');p.add_argument('--timeout',type=int,default=90)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    factor=a.factors/(name+'-L.mtx')
    if not factor.exists():
        (a.output/(name+'-skip.json')).write_text('{}\n');continue
    for round_index in range(2):
        for which in (('fixed','adaptive') if round_index==0 else ('adaptive','fixed')):
            dest=a.output/f'{name}-{which}-{round_index}.json'
            if a.resume and dest.exists():
                try:
                    if json.loads(dest.read_text()).get('returncode')==0:continue
                except (OSError,ValueError):pass
            exe=a.fixed if which=='fixed' else a.adaptive
            try:
                r=subprocess.run([str(exe),str(factor)]+(['--reverse'] if round_index else []),capture_output=True,text=True,timeout=a.timeout)
                result={'name':name,'round':round_index,'which':which,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
            except subprocess.TimeoutExpired as e:
                result={'name':name,'round':round_index,'which':which,'timeout':a.timeout,
                        'stdout':(e.stdout or b'').decode(errors='replace'),'stderr':(e.stderr or b'').decode(errors='replace')}
            dest.write_text(json.dumps(result,indent=2)+'\n')
            print(name,round_index,which,result.get('returncode','timeout'),flush=True)
