#!/usr/bin/env python3
"""Paired Plan vs locally adapted TC-SpTRSV; preserve every failure."""
import argparse,json,math,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--factors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--tc',type=Path,required=True);p.add_argument('--tc-workdir',type=Path,required=True)
p.add_argument('--timeout',type=int,default=35);p.add_argument('--resume',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
for name in json.loads(a.manifest.read_text())['names']:
    factor=(a.factors/(name+'-L.mtx')).resolve()
    if not factor.exists():
        (a.output/(name+'-skip.json')).write_text(json.dumps({'name':name,'status':'factor_missing'})+'\n');continue
    commands={'plan':[str(a.plan.resolve()),str(factor)],'tc':[str(a.tc.resolve()),str(factor)]}
    for rnd,order in enumerate((('plan','tc'),('tc','plan'))):
        for mode in order:
            target=a.output/f'{name}-{mode}-{rnd}.json'
            if a.resume and target.exists():
                try:
                    if json.loads(target.read_text()).get('complete'):continue
                except (OSError,ValueError):pass
            try:
                result=subprocess.run(commands[mode],cwd=a.tc_workdir if mode=='tc' else None,text=True,capture_output=True,timeout=a.timeout)
                d={'name':name,'mode':mode,'round':rnd,'complete':True,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
            except subprocess.TimeoutExpired as e:
                d={'name':name,'mode':mode,'round':rnd,'complete':True,'timeout_seconds':a.timeout,'stdout':(e.stdout or b'').decode(errors='replace'),'stderr':(e.stderr or b'').decode(errors='replace')}
            target.write_text(json.dumps(d,indent=2)+'\n');print(name,rnd,mode,d.get('returncode','timeout'),flush=True)
