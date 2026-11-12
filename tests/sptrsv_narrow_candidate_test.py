#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
try:
    from scipy.io import mmwrite
    from scipy.sparse import diags
except ImportError:
    print('SciPy is required for this test')
    sys.exit(77)

script=Path(__file__).resolve().parents[1]/'scripts/analysis/select_sptrsv_narrow_candidates.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);factors=root/'factors';factors.mkdir()
    chain=diags([[-.25]*39,[2.]*40],[-1,0],shape=(40,40),format='csr')
    mmwrite(factors/'chain-L.mtx',chain,symmetry='general')
    mmwrite(factors/'diagonal-L.mtx',diags([2.]*40,format='csr'),symmetry='general')
    (root/'manifest.json').write_text(json.dumps({'names':['chain','diagonal','missing']}))
    subprocess.run([sys.executable,str(script),'--manifest',str(root/'manifest.json'),'--factors',str(factors),'--output',str(root/'out')],check=True,capture_output=True,text=True)
    output=root/'out';selection=json.loads((output/'manifest.json').read_text());assert selection['names']==['chain']
    with (output/'features.csv').open() as f:r={x['name']:x for x in csv.DictReader(f)}
    assert r['chain']['levels']=='40' and r['chain']['csr_bytes']=='2076'
    assert r['diagonal']['selected']=='False' and r['missing']['status']=='factor_missing'
