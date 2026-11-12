#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_plan_pair.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['ok','bad']}))
    for name in ('ok','bad'):
        for rnd in range(2):
            for which in ('fixed','adaptive'):
                err='1e-2' if name=='bad' and which=='adaptive' else '1e-13'
                threads=',cached_threads\n' if which=='adaptive' else '\n'
                record=f'model,cached,128,{10 if which=="fixed" else 5},{err}' + (',1024\n' if which=='adaptive' else '\n')
                stdout='mode,backend,rows,solve_us,max_error'+threads+record
                (root/'cases'/f'{name}-{which}-{rnd}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f: rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['ok']['status']=='ok' and float(rows['ok']['speedup'])==2
    assert rows['bad']['status']=='invalid'
    for rnd in range(2):
        path=root/'cases'/f'ok-adaptive-{rnd}.json';d=json.loads(path.read_text());d['stdout']=d['stdout'].replace('model,cached,','model,cached-csr,');path.write_text(json.dumps(d))
    subprocess.run([sys.executable,str(script),str(root),'--allow-backend-change'],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['ok']['status']=='ok' and rows['ok']['fixed_backend']=='cached' and rows['ok']['backend']=='cached-csr'
