#!/usr/bin/env python3
"""Counterbalance baseline and narrow-level K=32 tile packing."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('manifest','factors','output','exe'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--timeout',type=int,default=90);p.add_argument('--resume',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    factor=a.factors/(name+'-L.mtx')
    if not factor.exists():(a.output/(name+'-skip.json')).write_text('{}\n');continue
    for rnd in range(2):
        for mode in (('usual','narrow') if rnd==0 else ('narrow','usual')):
            dest=a.output/f'{name}-{mode}-{rnd}.json'
            if a.resume and dest.exists():
                try:
                    if json.loads(dest.read_text()).get('returncode')==0:continue
                except (OSError,ValueError):pass
            env=dict(os.environ,ATSPTRSV_CACHED_ONLY='1')
            for k in ('ATSPTRSV_TILE_K2','ATSPTRSV_THREADS_512_1024','ATSPTRSV_THREADS_NARROW','ATSPTRSV_REVERSE_THREADS','ATSPTRSV_NARROW_K32'):env.pop(k,None)
            if mode=='narrow':env['ATSPTRSV_NARROW_K32']='1'
            if rnd:env['ATSPTRSV_REVERSE_THREADS']='1'
            try:
                r=subprocess.run([str(a.exe),str(factor),'--factor'],env=env,capture_output=True,text=True,timeout=a.timeout)
                result={'name':name,'round':rnd,'mode':mode,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
            except subprocess.TimeoutExpired as e:
                result={'name':name,'round':rnd,'mode':mode,'timeout':a.timeout,'stdout':(e.stdout or b'').decode(errors='replace'),'stderr':(e.stderr or b'').decode(errors='replace')}
            dest.write_text(json.dumps(result,indent=2)+'\n');print(name,rnd,mode,result.get('returncode','timeout'),flush=True)
