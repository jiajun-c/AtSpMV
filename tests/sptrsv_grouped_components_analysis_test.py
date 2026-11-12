#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_grouped_components.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['good','bad']}))
    for name in ('good','bad'):
        for order in ('tile-first','components-first'):
            err='1' if name=='bad' else '1e-13'
            stdout=('n,levels,variant,solve_us,max_error\n'
                    '1024,16,cusparse,20,1e-13\n1024,16,cached-tiles-1024,12,1e-13\n'
                    f'1024,16,cached-components-512,10,1e-13\n1024,16,grouped-components-512,5,{err}\n')
            (root/'cases'/f'{name}-{order}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['good']['status']=='ok' and float(rows['good']['grouped_vs_ungrouped'])==2
    assert rows['good']['baseline']=='cached-tiles-1024' and rows['bad']['status']=='invalid_or_unsupported'
