#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_component_pairs.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['cached','fallback','bad']}))
    for name in ('cached','fallback','bad'):
        n=5000 if name=='fallback' else 1024
        for order in ('tile-first','components-first'):
            err='1' if name=='bad' else '1e-13'
            stdout=('n,levels,variant,solve_us,max_error\n'
                    f'{n},16,cusparse,12,1e-13\n'
                    f'{n},16,cached-tiles-1024,10,1e-13\n'
                    f'{n},16,cached-components-512,5,{err}\n')
            (root/'cases'/f'{name}-{order}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f: rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['cached']['baseline']=='cached-tiles-1024' and float(rows['cached']['speedup'])==2
    assert rows['fallback']['baseline']=='cusparse' and float(rows['fallback']['speedup'])==2.4
    assert rows['bad']['status']=='invalid_or_unsupported'
