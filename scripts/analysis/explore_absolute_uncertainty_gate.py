#!/usr/bin/env python3
"""Explore a development-selected reference fallback on frozen v2 logs.

The forest's log-time tree spread is not a calibrated confidence interval.
This retrospective analysis never modifies the production Plan.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import joblib
import numpy as np
from train_vector_accuracy import enriched

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--features',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--threshold',type=float,help='Use a previously frozen threshold instead of selecting on dev')
p.add_argument('--exclude-cases',type=Path,help='Exclude matrices present in an earlier evaluated snapshot')
a=p.parse_args()
evaluation_bytes=a.evaluation.read_bytes()
rows=list(csv.DictReader(io.StringIO(evaluation_bytes.decode())))
excluded_bytes=a.exclude_cases.read_bytes() if a.exclude_cases else None
if excluded_bytes is not None:
    excluded={r['dataset'] for r in csv.DictReader(io.StringIO(excluded_bytes.decode()))}
    rows=[r for r in rows if r['dataset'] not in excluded]
if not rows:raise ValueError('no cases remain after exclusion')
with a.features.open(newline='') as f:features={(r['dataset'],int(r['threshold'])):r for r in csv.DictReader(f)}
model=joblib.load(a.model)
x=np.array([enriched(features,r['dataset'],1)[:38] for r in rows])
spread=np.std(np.stack([tree.predict(x) for tree in model.estimators_]),axis=0)
for r,s in zip(rows,spread):r['tree_log_std']=float(s)
groups=defaultdict(list)
for r in rows:groups[r['split']].append(r)
groups['all']=rows
if a.threshold is None and not groups['dev']:raise ValueError('development subset is empty')
if a.threshold is not None and (not math.isfinite(a.threshold) or a.threshold<0):raise ValueError('invalid threshold')
thresholds=[.08,.10,.12,.15,.18,.20,.25,math.inf]
def calculate(group,threshold):
    chosen=[float(r['old_reference_mape'] if r['tree_log_std']>threshold else r['new_reference_mape']) for r in group]
    return {'cases':len(group),'mape':statistics.mean(chosen),
            'max_ape':max(chosen),'over_20pct':sum(v>.2 for v in chosen),
            'fallbacks':sum(r['tree_log_std']>threshold for r in group)}
# Ties prefer fewer fallbacks; this decision uses development data only.
best=(a.threshold if a.threshold is not None else min(thresholds,key=lambda t:(calculate(groups['dev'],t)['mape'],calculate(groups['dev'],t)['fallbacks'])))
report={'method':('Frozen threshold applied to later cases excluded from the earlier gate snapshot. ' if a.threshold is not None else 'Retrospective threshold chosen on development matrices only. ')+'Tree log-time spread is a heuristic, not a confidence interval. Groups can share approximate families and previously observed data; timing noise remains. No Plan behavior is changed.',
        'evaluation_sha256':hashlib.sha256(evaluation_bytes).hexdigest(),
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'excluded_cases_sha256':hashlib.sha256(excluded_bytes).hexdigest() if excluded_bytes is not None else None,
        'threshold_log_std':None if math.isinf(best) else best,'metrics':{}}
for split,group in sorted(groups.items()):
    report['metrics'][split]={'learned_only':calculate(group,math.inf),'selected_gate':calculate(group,best)}
a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'cases.csv').open('w',newline='') as f:
    fields=['dataset','split','nnz','tree_log_std','old_reference_mape','new_reference_mape']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
