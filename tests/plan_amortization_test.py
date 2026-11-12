#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'status').mkdir();(p/'raw').mkdir()
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_tuned'}))
    (p/'manifest.json').write_text(json.dumps([{'dataset':'matrix'}]))
    (p/'status/matrix.json').write_text(json.dumps({'dataset':'matrix','status':'ok'}))
    (p/'raw/matrix.log').write_text('BENCH_ORDER,baseline_first\nPLAN_PREP,atbsr,10\nPLAN_PREP,auto_tuned,110\nBENCH_ORDER,selection_first\nPLAN_PREP,auto_tuned,130\nPLAN_PREP,atbsr,10\n')
    rows=[]
    for order in ('baseline_first','selection_first'):
        for backend,us in (('atbsr',5),('auto_tuned',3)):
            for trial in range(3):rows.append({'order':order,'backend':backend,'trial':trial,'us':us})
    with (p/'raw/matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_amortization.py'),str(p)],check=True,capture_output=True)
    row=next(csv.DictReader((p/'amortization.csv').open()))
    assert float(row['extra_plan_setup_us'])==110 and float(row['saved_us_per_call'])==2 and int(row['break_even_calls'])==55
    report=json.loads((p/'amortization.json').read_text())
    # Pair each order before averaging the ratio: sqrt((15/113)*(15/133)).
    assert abs(report['reuse_scenarios']['1']['geomean_speedup']-((15/113)*(15/133))**.5)<1e-12
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_fast_tuned','compare_full_tune':True}))
    (p/'raw/matrix.log').write_text('BENCH_ORDER,baseline_first\nPLAN_PREP,atbsr,10\nPLAN_PREP,auto_tuned,110\nPLAN_PREP,auto_fast_tuned,20\nBENCH_ORDER,selection_first\nPLAN_PREP,auto_fast_tuned,30\nPLAN_PREP,auto_tuned,130\nPLAN_PREP,atbsr,10\n')
    for order in ('baseline_first','selection_first'):
        for trial in range(3):rows.append({'order':order,'backend':'auto_fast_tuned','trial':trial,'us':4})
    with (p/'raw/matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_amortization.py'),str(p)],check=True,capture_output=True)
    report=json.loads((p/'amortization.json').read_text())
    assert report['versus_full_tune']['median_full_minus_fast_setup_us']==95
    assert abs(report['versus_full_tune']['reuse_scenarios']['1']['geomean_speedup']-((113/24)*(133/34))**.5)<1e-12
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model','compare_auto_model':True}))
    (p/'raw/matrix.log').write_text('BENCH_ORDER,baseline_first\nPLAN_PREP,atbsr,10\nPLAN_PREP,auto_model,20\nPLAN_PREP,auto_vector_model,30\nBENCH_ORDER,selection_first\nPLAN_PREP,auto_vector_model,40\nPLAN_PREP,auto_model,30\nPLAN_PREP,atbsr,10\n')
    for order in ('baseline_first','selection_first'):
        for backend,value in [('auto_model',4),('auto_vector_model',3)]:
            for trial in range(3):rows.append({'order':order,'backend':backend,'trial':trial,'us':value})
    with (p/'raw/matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_amortization.py'),str(p)],check=True,capture_output=True)
    report=json.loads((p/'amortization.json').read_text())
    assert report['versus_auto_model']['median_model_minus_selected_setup_us']==-10
    assert abs(report['versus_auto_model']['reuse_scenarios']['1']['geomean_speedup']-((24/33)*(34/43))**.5)<1e-12
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model','compare_auto_model':True,'warm_setup_anchor':True}))
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_amortization.py'),str(p)],check=True,capture_output=True)
    report=json.loads((p/'amortization.json').read_text())
    assert not report['baseline_setup_comparison_valid'] and report['median_break_even_calls'] is None
    assert 'reuse_scenarios' not in report and 'versus_auto_model' in report
print('plan amortization analysis passed')
