#!/usr/bin/env python3
"""Validate the Graph/clock factorial report and its mode markers."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_graph_timing_controls.py'
with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'metadata.json').write_text(json.dumps({'matrices':{'case':'hash'}}))
    for shared in (0,1):
        for internal in (0,1):
            for order in ('forward','reverse'):
                label=f'case-graph{shared}-event{internal}-{order}'
                with (p/(label+'.csv')).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['backend','config','us']);w.writeheader()
                    for backend,value in [('auto_vector_model',2),('auto_vector_model_2pct',1)]:
                        for _ in range(21):w.writerow({'backend':backend,'config':'same','us':value})
                log='PLAN_ALIAS,auto_vector_model_2pct,shared\n'
                if shared:log+='GRAPH_ALIAS,auto_vector_model_2pct,auto_vector_model\n'
                if internal:log+='TIMING_MODE,internal_graph_events\n'
                (p/(label+'.log')).write_text(log)
    cmd=[sys.executable,str(script),str(p),'--output',str(p/'out.json')]
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads((p/'out.json').read_text())
    assert all(abs(v['geomean_ratio']-2)<1e-12 for v in result['groups'].values())
    (p/'case-graph1-event1-forward.log').write_text('PLAN_ALIAS,auto_vector_model_2pct,shared\n')
    assert subprocess.run(cmd,capture_output=True).returncode!=0
print('Graph timing controls passed')
