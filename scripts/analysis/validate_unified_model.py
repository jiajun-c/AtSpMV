#!/usr/bin/env python3
"""Compare generated C++ inference with all frozen sklearn predictions."""
import csv
import math
import subprocess
import sys
from fit_unified_selector import CONFIGS
from fit_hybrid_selector import OUT, vector

if len(sys.argv)!=2:
    raise SystemExit('usage: validate_unified_model.py CPP_PREDICTOR')
features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader((OUT/'features.csv').open())}
reference=list(csv.DictReader((OUT/'unified-reference.csv').open()))
lines=[]
for r in reference:
    i=int(r['candidate']);_,_,tile,threshold=CONFIGS[i]
    x=vector(features[(r['dataset'],threshold)],tile)
    lines.append(str(i)+' '+' '.join(format(v,'.17g') for v in x))
p=subprocess.run([sys.argv[1]],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
out=[float(x) for x in p.stdout.splitlines()]
if len(out)!=len(reference):raise RuntimeError('prediction count mismatch')
error=max(abs(math.log(actual/float(r['prediction_ratio']))) for actual,r in zip(out,reference))
if error>1e-11:raise RuntimeError(f'inference mismatch {error}')
print(f'PASS: {len(out)} predictions; max absolute log-ratio error {error:.3g}')
