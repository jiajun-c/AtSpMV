#!/usr/bin/env python3
"""Compare default and model SpTRSV Plans in both construction orders."""
import argparse
import json
import os
import subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--factors',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--exe',type=Path,required=True)
p.add_argument('--timeout',type=int,default=45)
p.add_argument('--resume',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    factor=a.factors/(name+'-L.mtx')
    if not factor.exists():
        (a.output/(name+'-skip.json')).write_text(json.dumps({'name':name,'status':'factor_missing'})+'\n')
        continue
    for order in ('forward','reverse'):
        target = a.output/(name+'-'+order+'.json')
        if a.resume and target.exists():
            try:
                previous = json.loads(target.read_text())
                if previous.get('returncode') == 0 and 'default,' in previous.get('stdout','') and 'model,' in previous.get('stdout',''):
                    print(name,order,'already complete',flush=True)
                    continue
            except (OSError, ValueError):
                pass
        cmd=[str(a.exe),str(factor)]+(['--reverse'] if order=='reverse' else [])
        try:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=a.timeout)
            result={'name':name,'order':order,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except subprocess.TimeoutExpired as e:
            result={'name':name,'order':order,'timeout_seconds':a.timeout,'stdout':(e.stdout or b'').decode(errors='replace'),'stderr':(e.stderr or b'').decode(errors='replace')}
        (a.output/(name+'-'+order+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,order,result.get('returncode','timeout'),flush=True)
