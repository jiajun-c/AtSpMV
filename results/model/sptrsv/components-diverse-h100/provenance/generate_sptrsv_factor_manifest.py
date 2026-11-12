#!/usr/bin/env python3
"""Generate ILUT factors with per-input timeout and resumable records."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('manifest','matrix-root','output','generator'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--timeout',type=int,default=30);p.add_argument('--resume',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name in json.loads(a.manifest.read_text())['names']:
    record=a.output/(name+'-factor.json')
    if a.resume and record.exists():continue
    try:
        r=subprocess.run([sys.executable,str(a.generator),'--matrix-root',str(a.matrix_root),'--output',str(a.output),name],
                         env=env,capture_output=True,text=True,timeout=a.timeout)
        if r.returncode!=0 and not record.exists():
            (a.output/(name+'-L.mtx')).unlink(missing_ok=True)
            record.write_text(json.dumps({'matrix':name,'error':'generator failed','returncode':r.returncode,'stderr':r.stderr[-4000:]})+'\n')
        print(name,r.returncode,flush=True)
    except subprocess.TimeoutExpired:
        (a.output/(name+'-L.mtx')).unlink(missing_ok=True)
        record.write_text(json.dumps({'matrix':name,'error':'factorization timeout','timeout_seconds':a.timeout})+'\n')
        print(name,'timeout',flush=True)
