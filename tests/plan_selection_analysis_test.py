#!/usr/bin/env python3
"""Verify paired-order speedup is computed per order, then geometrically averaged."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'status').mkdir();(p/'raw').mkdir()
    (p/'manifest.json').write_text(json.dumps([{'dataset':'example','nnz':4}]))
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_model'}))
    (p/'status/example.json').write_text(json.dumps({'status':'ok'}))
    rows=[]
    for order,base,auto,cusparse in [('baseline_first',9,10,20),('selection_first',11,10,20)]:
        for backend,value in [('atbsr',base),('auto_model',auto),('cusparse',cusparse)]:
            for trial in range(3):rows.append({'backend':backend,'config':'atbsr/0/0' if backend=='auto_model' else 'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_default']['geomean'],math.sqrt(.9*1.1),rel_tol=1e-12)
    assert math.isclose(result['speedup_vs_cusparse']['geomean'],2.,rel_tol=1e-12)
    assert result['status_counts']=={'ok':1}
    # The extended comparator is selected separately in each construction
    # order before averaging; it must not include the selected model itself.
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_model','extended_baselines':True}))
    for order in ('baseline_first','selection_first'):
        for backend,value in [('atbsr_m8',8),('atbsr_m1_balanced',12),('csr_dsm_fixed',13),('csr_two_stage_fixed',14)]:
            for trial in range(3):rows.append({'backend':backend,'config':'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_extended_baseline']['geomean'],.8,rel_tol=1e-12)
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_model','extended_baselines':True,'vector_candidates':True}))
    for order in ('baseline_first','selection_first'):
        for width in (1,2,4,8,16,32):
            for trial in range(3):rows.append({'backend':f'csr_vec{width}','config':'test','trial':trial,'us':4+width,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_best_vector_vs_extended']['geomean'],8/5,rel_tol=1e-12)
    assert math.isclose(result['speedup_oracle_extended_plus_vector']['geomean'],8/5,rel_tol=1e-12)
    # Full and fast tuning are separate plans; compare their per-order times,
    # not the ratio of pooled medians or the fixed baseline.
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_fast_tuned','compare_full_tune':True}))
    for order,fast,full in [('baseline_first',10,8),('selection_first',10,18)]:
        for backend,value in [('auto_fast_tuned',fast),('auto_tuned',full)]:
            for trial in range(3):rows.append({'backend':backend,'config':'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_full_tune']['geomean'],math.sqrt(.8*1.8),rel_tol=1e-12)
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model','compare_auto_model':True}))
    for order,value in [('baseline_first',8),('selection_first',10)]:
        for trial in range(3):rows.append({'backend':'auto_vector_model','config':'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_auto_model']['geomean'],math.sqrt(1.25),rel_tol=1e-12)
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model_2pct','compare_auto_model':True,'compare_ungated_vector':True}))
    for order,value in [('baseline_first',10),('selection_first',5)]:
        for trial in range(3):rows.append({'backend':'auto_vector_model_2pct','config':'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_ungated_vector']['geomean'],math.sqrt(.8*2),rel_tol=1e-12)
    assert math.isclose(result['speedup_ungated_vs_auto_model']['geomean'],math.sqrt(1.25),rel_tol=1e-12)
    # Mixed tuning, vector prediction and bounded refinement are compared per
    # construction order rather than taking a pooled median.
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_tuned_mixed', 'compare_auto_model':True,
                                               'compare_vector_model':True, 'compare_refined_model':True}))
    for order, mixed, refined in [('baseline_first', 5, 6), ('selection_first', 10, 8)]:
        for backend, value in [('auto_tuned_mixed', mixed), ('auto_model_refined', refined)]:
            for trial in range(3):
                rows.append({'backend':backend,'config':'test','trial':trial,'us':value,'order':order})
    with (p/'raw/example.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(root/'scripts/analysis/analyze_plan_selection.py'),str(p)],check=True,capture_output=True)
    result=json.loads((p/'analysis.json').read_text())
    assert math.isclose(result['speedup_vs_refined_model']['geomean'],math.sqrt(6/5*8/10),rel_tol=1e-12)
    assert math.isclose(result['speedup_refined_vs_vector_model']['geomean'],math.sqrt(8/6*10/8),rel_tol=1e-12)
print('paired plan selection analysis passed')
