#!/usr/bin/env python3
"""Compare the GPU application against upstream MFEM ex16 running on CPU."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
for key in ('oracle','exe','mesh','output'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
results=[]
for refine in (2,4):
    folder=(a.output/f'r{refine}').resolve();folder.mkdir(exist_ok=True)
    args=['-m',str(a.mesh.resolve()),'-r',str(refine),'-o','2','-s','21','-dt','0.01','-tf','0.03','-a','0.01','-no-vis']
    ref=subprocess.run([str(a.oracle.resolve()),*args],cwd=folder,capture_output=True,text=True,timeout=120)
    (folder/'cpu.log').write_text(ref.stdout+'\n'+ref.stderr)
    if ref.returncode:raise RuntimeError('CPU oracle failed')
    text=(folder/'ex16-final.gf').read_text().split('Ordering: 0',1)[1]
    expected=np.fromstring(text,sep=' ')
    for backend in ('cusparse','mfem','atbsr','mixed'):
        dest=folder/(backend+'.bin')
        run=subprocess.run([str(a.exe.resolve()),*args,'-fa','-b',backend,'-sol',str(dest)],cwd=folder,capture_output=True,text=True,timeout=120)
        (folder/(backend+'.log')).write_text(run.stdout+'\n'+run.stderr)
        if run.returncode:raise RuntimeError('GPU application failed: '+backend)
        got=np.fromfile(dest,dtype=np.float64)
        if got.shape!=expected.shape or not np.isfinite(got).all():raise RuntimeError('invalid field')
        error=float(np.max(np.abs(got-expected)))/max(1.,float(np.max(np.abs(expected))))
        if error>1e-7:raise RuntimeError(f'CPU oracle mismatch: r={refine} {backend} {error}')
        results.append({'refine':refine,'backend':backend,'dofs':len(got),'relative_max_error':error})
        print(results[-1],flush=True)
(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
