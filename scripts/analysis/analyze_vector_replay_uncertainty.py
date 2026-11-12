#!/usr/bin/env python3
"""Family-cluster bootstrap for already-observed vector-model replay."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--decisions', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=20261029)
p.add_argument('--samples', type=int, default=5000)
a = p.parse_args()
if a.samples < 100: raise ValueError('at least 100 bootstrap samples required')
data = json.loads(a.decisions.read_text())
result = {'method':'Retrospective replay of independently timed existing candidates; cluster bootstrap samples approximate name-based families with replacement, preserving all matrices in each sampled family. These samples were already inspected; intervals do not account for tuning, timing noise, other model overlap, or family misclassification.',
          'decisions_sha256':hashlib.sha256(a.decisions.read_bytes()).hexdigest(),
          'seed':a.seed,'bootstrap_samples':a.samples,'datasets':{}}
for name, cases in data.items():
    groups=defaultdict(list)
    for case in cases:
        ds=case['dataset'].lower();family=re.sub(r'[\d_\-]+$', '', ds) or ds
        pred=case['predictions']; old=float(pred['deployed']['speedup']);new=float(pred['extended_deep']['speedup'])
        if not (math.isfinite(old) and old>0 and math.isfinite(new) and new>0):raise ValueError('invalid replay ratio')
        groups[family].append(math.log(new/old))
    if not groups: continue
    families=sorted(groups);rng=random.Random(a.seed)
    sample=[]
    for _ in range(a.samples):
        selected=[rng.choice(families) for _ in families]
        vals=[x for family in selected for x in groups[family]]
        sample.append(math.exp(statistics.mean(vals)))
    sample.sort();n=len(sample)
    values=[x for family in families for x in groups[family]]
    result['datasets'][name]={'matrices':len(values),'approximate_families':len(families),
        'geomean_v2_speedup_vs_v1':math.exp(statistics.mean(values)),
        'bootstrap_95pct_interval':[sample[int(.025*(n-1))],sample[int(.975*(n-1))]],
        'v2_faster_5pct':sum(x>math.log(1.05) for x in values),
        'v2_slower_5pct':sum(x< -math.log(1.05) for x in values)}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
