#!/usr/bin/env python3
"""Validate strict predicted-margin gate against measured timings."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_vector_gate.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'raw').mkdir()
    (p/'frozen-model-evaluation.json').write_text(json.dumps({'complete_success':True,'manifest_total':2}))
    (p/'frozen-model-results.csv').write_text('dataset,chosen,predicted_ratio_to_default,fallback_predicted_ratio_to_default\na,csr_vec4,0.98,1\nb,csr_vec8,0.90,1\n')
    for name,vector in [('a',20),('b',5)]:
        with (p/'raw'/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['order','backend','us']);w.writeheader()
            for order in ('baseline_first','selection_first'):
                for backend in ('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse','auto_model'):
                    w.writerow({'order':order,'backend':backend,'us':10})
                w.writerow({'order':order,'backend':'csr_vec4' if name=='a' else 'csr_vec8','us':vector})
    subprocess.run([sys.executable,str(script),str(p),'--output',str(p/'gate.json')],check=True,capture_output=True)
    result=json.loads((p/'gate.json').read_text())['margins']
    assert result['0.0']['selected_vector']==2 and result['0.0']['regressions_vs_model']==1
    assert result['0.02']['selected_vector']==1 and result['0.02']['regressions_vs_model']==0
    assert result['0.1']['selected_vector']==0
print('vector model gate analysis passed')
