#!/usr/bin/env python3
"""Check refinement candidate membership with matched and missing winners."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    (d/'status').mkdir(); (d/'raw').mkdir()
    (d/'manifest.json').write_text('[{"dataset":"one"}]')
    (d/'metadata.json').write_text('{"compare_refined_model":true,"compare_full_tune":true}')
    (d/'status/one.json').write_text('{"status":"ok"}')
    with (d/'raw/one.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['order','backend','config','us']); w.writeheader()
        for order, tile in [('baseline_first', 1), ('selection_first', 2)]:
            w.writerow({'order':order,'backend':'auto_tuned','config':f'atbsr/{tile}/0/8/128/1/1024/vw=0','us':1})
            w.writerow({'order':order,'backend':'auto_model_refined','config':'csr_vector/-1/-1/8/128/1/1024/vw=4/mc=0','us':2})
    with (d/'raw/one.log').open('w') as f:
        for order in ('baseline_first', 'selection_first'):
            f.write(f'BENCH_ORDER,{order}\nPLAN_PREP,auto_model_refined,100\nTUNING_BEGIN selected=csr_vector\n')
            f.write('TUNING,atbsr,1,0,8,128,1,1024,0,1,0\nTUNING_END\n')
    subprocess.run(['python3', str(root/'scripts/analysis/analyze_refined_misses.py'), str(d)], check=True, capture_output=True)
    data = json.loads((d/'refined-misses.json').read_text())
    assert data['cases_over_cutoff'] == 1
    assert data['candidate_coverage_per_order'] == {'winner_in_refinement_candidates':1, 'winner_not_in_refinement_candidates':1}
print('refinement miss analysis passed')
