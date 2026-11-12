#!/usr/bin/env python3
"""Ensure benchmark audit detects missing or incorrect shared Plan markers."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/audit_plan_sweep.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'raw').mkdir();(p/'status').mkdir()
    (p/'manifest.json').write_text(json.dumps([{'dataset':'case'}]))
    (p/'metadata.json').write_text(json.dumps({'selection':'auto_vector_model_2pct','compare_auto_model':True,'compare_ungated_vector':True,'alias_identical_models':True,'counterbalance':True,'trials':1}))
    (p/'status/case.json').write_text(json.dumps({'status':'ok'}))
    with (p/'raw/case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['order','backend','config','trial','us']);w.writeheader()
        for order in ('baseline_first','selection_first'):
            for name,config in [('atbsr','a'),('cusparse','c'),('auto_model','same'),('auto_vector_model','same'),('auto_vector_model_2pct','different')]:
                w.writerow({'order':order,'backend':name,'config':config,'trial':0,'us':1})
    good=''.join(f'BENCH_ORDER,{order}\nPLAN_ALIAS,auto_model,owner\nPLAN_ALIAS,auto_vector_model,shared\nPLAN_ALIAS,auto_vector_model_2pct,owner\n' for order in ('baseline_first','selection_first'))
    (p/'raw/case.log').write_text(good)
    cmd=[sys.executable,str(script),str(p)]
    subprocess.run(cmd,check=True,capture_output=True)
    (p/'raw/case.log').write_text(good.replace('auto_vector_model,shared','auto_vector_model,owner'))
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    meta=json.loads((p/'metadata.json').read_text());meta.update(alias_identical_graphs=True,internal_graph_events=True)
    (p/'metadata.json').write_text(json.dumps(meta))
    graph_good=''.join(f'BENCH_ORDER,{order}\nPLAN_ALIAS,auto_model,owner\nPLAN_ALIAS,auto_vector_model,shared\nPLAN_ALIAS,auto_vector_model_2pct,owner\nTIMING_MODE,internal_graph_events\nGRAPH_ALIAS,auto_vector_model,auto_model\n' for order in ('baseline_first','selection_first'))
    (p/'raw/case.log').write_text(graph_good)
    subprocess.run(cmd,check=True,capture_output=True)
    (p/'raw/case.log').write_text(graph_good.replace('GRAPH_ALIAS,auto_vector_model,auto_model','GRAPH_ALIAS,auto_vector_model,auto_vector_model_2pct'))
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    meta.update(compare_vector_model=True,compare_vector_model_v2=True,compare_refined_model=True)
    (p/'metadata.json').write_text(json.dumps(meta))
    with (p/'raw/case.csv').open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['order','backend','config','trial','us'])
        for order in ('baseline_first','selection_first'):
            w.writerow({'order':order,'backend':'auto_vector_model_v2','config':'same','trial':0,'us':1})
            w.writerow({'order':order,'backend':'auto_model_refined','config':'refined','trial':0,'us':1})
    v2_log=graph_good.replace('PLAN_ALIAS,auto_vector_model_2pct,owner',
                              'PLAN_ALIAS,auto_vector_model_v2,shared\nPLAN_ALIAS,auto_vector_model_2pct,owner')
    v2_log=v2_log.replace('GRAPH_ALIAS,auto_vector_model,auto_model',
                          'GRAPH_ALIAS,auto_vector_model,auto_model\nGRAPH_ALIAS,auto_vector_model_v2,auto_model')
    (p/'raw/case.log').write_text(v2_log)
    subprocess.run(cmd,check=True,capture_output=True)
    meta['log_model_predictions']=True
    (p/'metadata.json').write_text(json.dumps(meta))
    model_log=''
    for label in ('auto_vector_model','auto_vector_model_v2'):
        model_log+=f'MODEL_REFERENCE,{label},1,1\n'
        model_log+=''.join(f'MODEL_PREDICTION,{label},csr_vector,-1,-1,8,128,1,1024,{width},1,1\n' for width in (1,2,4,8,16,32))
    with_predictions=v2_log.replace('TIMING_MODE,internal_graph_events\n',model_log+'TIMING_MODE,internal_graph_events\n')
    (p/'raw/case.log').write_text(with_predictions)
    subprocess.run(cmd,check=True,capture_output=True)
    (p/'raw/case.log').write_text(with_predictions.replace('MODEL_PREDICTION,auto_vector_model_v2,csr_vector,-1,-1,8,128,1,1024,32,1,1\n',''))
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('shared Plan marker audit passed')
