#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_cusparse_from_sota.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir()
    (root/'analysis.csv').write_text('name,status\ngood,ok\nbad,ok\nmissing,factor_missing\n')
    for name in ('good','bad'):
        for rnd in range(2):
            err='1' if name=='bad' else '1e-13'
            stdout=('mode,backend,rows,solve_us,max_error\n'
                    f'default,cusparse,128,10,{err}\nmodel,cached,128,5,1e-13\n')
            (root/'cases'/f'{name}-plan-{rnd}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'cusparse_analysis.csv').open() as f:r={x['name']:x for x in csv.DictReader(f)}
    assert r['good']['status']=='ok' and float(r['good']['speedup'])==2
    assert r['bad']['status']=='plan_invalid_or_missing' and r['missing']['status']=='factor_missing'
