#!/usr/bin/env python3
"""Compare exported C++ forest predictions with the frozen sklearn model."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts/analysis'))
from fit_hybrid_selector import vector
data=root/'results/model/hybrid-time-model/vector-model-independent-120'
rows=list(csv.DictReader((data/'provenance/training.csv').open()))
x=np.array([[float(r[f'x{i}']) for i in range(15)] for r in rows]);y=np.log([float(r['ratio']) for r in rows])
model=ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=5,random_state=20261028,n_jobs=1).fit(x,y)
names={r['dataset'] for r in csv.DictReader((data/'index.csv').open())}
features={r['dataset']:r for r in csv.DictReader((data/'provenance/features-exact.csv').open()) if r['threshold']=='1024' and r['dataset'] in names}
assert set(features)==names
inputs=[vector(features[name],3)+[math.log2(width)] for name in sorted(names) for width in (1,2,4,8,16,32)]
inputs.extend(x[i].tolist() for i in range(0,len(x),max(1,len(x)//150)))
expected=np.exp(model.predict(np.asarray(inputs)))
with tempfile.TemporaryDirectory() as d:
    binary=Path(d)/'predict'
    subprocess.run(['g++','-O2','-std=c++17','-I',str(root/'include'),str(root/'tools/vector_model_predict.cpp'),str(root/'src/selection/vectorModelCalibration.cpp'),'-o',str(binary)],check=True)
    text=''.join(' '.join(format(v,'.17g') for v in row)+'\n' for row in inputs)
    result=subprocess.run([str(binary)],input=text,text=True,capture_output=True,check=True)
    actual=np.asarray([float(line) for line in result.stdout.splitlines()])
    assert len(actual)==len(expected)
    error=np.max(np.abs(actual-expected)/expected)
    assert error<1e-12,error
print(f'exported vector model matches sklearn for {len(inputs)} predictions')
