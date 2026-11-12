#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_shape_pair.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['ok','bad']}))
    for name in ('ok','bad'):
        for rnd in range(2):
            for mode in ('usual','narrow'):
                err='0.1' if name=='bad' and mode=='narrow' else '1e-13'
                stdout=f'n,variant,solve_us,max_error\n128,cached-tiles-512,{10 if mode=="usual" else 5},{err}\n'
                (root/'cases'/f'{name}-{mode}-{rnd}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['ok']['status']=='ok' and float(rows['ok']['speedup'])==2
    assert rows['bad']['status']=='invalid_or_unsupported'
