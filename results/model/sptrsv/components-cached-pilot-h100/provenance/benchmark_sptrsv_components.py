#!/usr/bin/env python3
"""Explore existing multi-CTA candidates on selected factored matrices."""
import argparse
import json
import os
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
for key in ('manifest','factors','output','exe'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--timeout',type=int,default=90);p.add_argument('--resume',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    dest=a.output/(name+'.json');factor=a.factors/(name+'-L.mtx')
    if not factor.exists():dest.write_text(json.dumps({'name':name,'status':'factor_missing'})+'\n');continue
    if a.resume and dest.exists():
        try:
            if json.loads(dest.read_text()).get('returncode')==0:continue
        except (OSError,ValueError):pass
    env=dict(os.environ);env.pop('ATSPTRSV_CACHED_ONLY',None);env.pop('ATSPTRSV_TILE_K2',None)
    try:
        r=subprocess.run([str(a.exe),str(factor),'--factor'],env=env,capture_output=True,text=True,timeout=a.timeout)
        result={'name':name,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired as e:
        result={'name':name,'timeout':a.timeout,'stdout':(e.stdout or b'').decode(errors='replace'),'stderr':(e.stderr or b'').decode(errors='replace')}
    dest.write_text(json.dumps(result,indent=2)+'\n');print(name,result.get('returncode','timeout'),flush=True)
