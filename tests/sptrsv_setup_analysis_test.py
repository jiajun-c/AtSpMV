#!/usr/bin/env python3
import csv,json,subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_sptrsv_setup.py'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'cases').mkdir();(root/'manifest.json').write_text(json.dumps({'names':['good','bad']}))
    for name in ('good','bad'):
        for order in ('forward','reverse'):
            count=11 if name=='bad' else 12
            stdout=('mode,backend,setup_median_us,destroy_median_us,iterations\n'
                    f'default,cusparse,10,1,{count}\nmodel,cached-csr,5,1,{count}\n')
            (root/'cases'/f'{name}-{order}.json').write_text(json.dumps({'returncode':0,'stdout':stdout}))
    subprocess.run([sys.executable,str(script),str(root)],check=True,capture_output=True,text=True)
    with (root/'analysis.csv').open() as f:rows={r['name']:r for r in csv.DictReader(f)}
    assert rows['good']['status']=='ok' and float(rows['good']['speedup'])==2
    assert rows['bad']['status']=='invalid'
