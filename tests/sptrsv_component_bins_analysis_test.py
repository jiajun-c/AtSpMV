#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_component_bins.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['good','bad']}))
    for name in ('good','bad'):
        for order in ('tile-first','components-first'):
            err='1' if name=='bad' else '1e-13'
            stdout=('n,variant,solve_us,max_error\n'
                    '128,cusparse,20,1e-13\n128,cached-components-512,10,1e-13\n'
                    f'128,binned-components-128,5,{err}\n128,binned-components-256,8,1e-13\n')
            (root/'cases'/f'{name}-{order}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['good']['status']=='ok' and float(rows['good']['bin128_speedup'])==2
    assert rows['bad']['status']=='invalid_or_unsupported'
