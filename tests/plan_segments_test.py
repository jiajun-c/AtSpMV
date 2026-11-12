#!/usr/bin/env python3
"""Check structural bucketing, feature coverage, and geometric summaries."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

script=Path(__file__).resolve().parents[1]/'scripts/analysis/analyze_plan_segments.py'
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory)
    with (p/'analysis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','nnz','selected','speedup_vs_extended_baseline']);w.writeheader()
        w.writerows([{'dataset':'a','nnz':100,'selected':'atbsr','speedup_vs_extended_baseline':2},
                     {'dataset':'b','nnz':200,'selected':'atbsr','speedup_vs_extended_baseline':.5},
                     {'dataset':'missing','nnz':200,'selected':'csr_vector','speedup_vs_extended_baseline':1}])
    with (p/'features.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','threshold','rows','nnz','long_rows']);w.writeheader()
        w.writerows([{'dataset':'a','threshold':1024,'rows':100,'nnz':100,'long_rows':0},
                     {'dataset':'b','threshold':1024,'rows':100,'nnz':200,'long_rows':10},
                     {'dataset':'b','threshold':128,'rows':100,'nnz':200,'long_rows':50}])
    cmd=[sys.executable,str(script),str(p),'--features',str(p/'features.csv'),'--output',str(p/'out.json')]
    subprocess.run(cmd,check=True,capture_output=True)
    result=json.loads((p/'out.json').read_text())
    assert result['matched_cases']==2 and result['missing_features']==['missing']
    assert result['groups']['nnz']['<10k']['geomean_speedup']==1
    assert result['groups']['nnz']['<10k']['wins']==1
    assert result['groups']['long_rows']['>=10% rows >=1024']['worst_dataset']=='b'
print('plan segmentation passed')
