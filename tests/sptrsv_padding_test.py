#!/usr/bin/env python3
import json,subprocess,sys,tempfile
from pathlib import Path
try:
    import numpy as np
    from scipy.io import mmread,mmwrite
    from scipy.sparse import csr_matrix
except ImportError:
    print('NumPy and SciPy are required for this test')
    sys.exit(77)

script=Path(__file__).resolve().parents[1]/'scripts/analysis/pad_sptrsv_factors.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);factors=root/'factors';factors.mkdir();out=root/'out'
    matrix=csr_matrix(([1.,-.25,2.,3.],([0,1,1,2],[0,0,1,2])),shape=(3,3))
    mmwrite(factors/'real-L.mtx',matrix,symmetry='general')
    mmwrite(factors/'complex-L.mtx',csr_matrix(np.array([[1+1j]])),symmetry='general')
    (root/'manifest.json').write_text(json.dumps({'names':['real','complex','missing']}))
    subprocess.run([sys.executable,str(script),'--manifest',str(root/'manifest.json'),'--factors',str(factors),'--output',str(out)],check=True,capture_output=True,text=True)
    records={r['name']:r for r in json.loads((out/'padding.json').read_text())}
    assert records['real']['status']=='ok' and records['complex']['status']=='padding_failed' and records['missing']['status']=='factor_missing'
    b=mmread(out/'real-L.mtx').tocsr();assert b.shape==(16,16)
    assert (b[:3,:3]!=matrix).nnz==0 and b[3:,:3].nnz==0 and b[:3,3:].nnz==0
    assert np.all(b[3:,3:].diagonal()==1) and b[3:,3:].nnz==13
