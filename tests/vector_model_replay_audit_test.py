#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/audit_vector_model_replay.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);a=p/'actual';r=p/'replay';(a/'status').mkdir(parents=True);(a/'raw').mkdir();r.mkdir()
    (a/'manifest.json').write_text(json.dumps([{'dataset':'one'},{'dataset':'two'}]))
    (a/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model'}))
    (r/'frozen-model-results.csv').write_text('dataset,chosen,predicted_ratio_to_default,fallback_predicted_ratio_to_default\none,csr_vec8,0.9,1\ntwo,auto_model,1,1\n')
    for name in ('one','two'):
        (a/'status'/f'{name}.json').write_text(json.dumps({'status':'ok'}))
        with (a/'raw'/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['order','backend','config']);w.writeheader()
            for order in ('baseline_first','selection_first'):
                baseline='atbsr/3/0/vw=0';vector='csr_vector/-1/-1/vw=8' if name=='one' else baseline
                w.writerow({'order':order,'backend':'auto_model','config':baseline})
                w.writerow({'order':order,'backend':'auto_vector_model','config':vector})
    cmd=[sys.executable,str(script),str(a),str(r)]
    subprocess.run(cmd,check=True,capture_output=True)
    report=json.loads((a/'replay-audit.json').read_text());assert report['decisions_checked']==4 and not report['issues']
    (r/'frozen-model-results.csv').write_text('dataset,chosen,predicted_ratio_to_default,fallback_predicted_ratio_to_default\none,csr_vec4,0.9,1\ntwo,auto_model,1,1\n')
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('vector model replay audit passed')
