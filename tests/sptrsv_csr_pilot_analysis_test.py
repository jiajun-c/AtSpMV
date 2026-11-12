#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_csr_pilot.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['cached','fallback','bad']}))
    for name in ('cached','fallback','bad'):
        for order in ('forward','reverse'):
            e='0.1' if name=='bad' else '1e-13';selected='selector-cusparse' if name=='fallback' else 'selector-cached'
            stdout=('n,levels,variant,solve_us,max_error\n'
                    f'128,64,cusparse,12,1e-13\n128,64,{selected},10,1e-13\n'
                    f'128,64,cached-tiles-512,10,1e-13\n128,64,cached-csr-32-256,5,{e}\n')
            (root/'cases'/f'{name}-{order}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:r={v['name']:v for v in csv.DictReader(f)}
    assert r['cached']['baseline']=='cached-tiles-512' and float(r['cached']['speedup'])==2
    assert r['fallback']['baseline']=='cusparse' and float(r['fallback']['speedup'])==2.4
    assert r['bad']['status']=='invalid_or_unsupported'
