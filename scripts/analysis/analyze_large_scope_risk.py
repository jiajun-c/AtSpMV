#!/usr/bin/env python3
"""Summarize historical default fallback on the out-of-model-range suite."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--large-index',type=Path,required=True)
parser.add_argument('--historical-analysis',type=Path,required=True)
parser.add_argument('--in-scope-count',type=int,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.in_scope_count<=0:raise ValueError('in-scope count must be positive')
index=list(csv.DictReader(args.large_index.open()));names=[r['dataset'] for r in index]
if not names or len(names)!=len(set(names)):raise ValueError('empty or duplicated large index')
history={r['dataset']:r for r in csv.DictReader(args.historical_analysis.open())}
missing=[n for n in names if n not in history]
if missing:raise ValueError(f'{len(missing)} large cases missing historical measurements')
values=[]
for name in names:
    row=history[name];best=float(row['best_nonhybrid_us']);default=float(row['atbsr_default_us'])
    if not math.isfinite(best) or not math.isfinite(default) or best<=0 or default<=0:raise ValueError(f'{name}: invalid timing')
    values.append(best/default)
geo=math.exp(statistics.mean(map(math.log,values)))
report={'large_cases':len(values),'in_scope_count':args.in_scope_count,'historical_large_default_vs_fixed_best':{'geomean':geo,'median':statistics.median(values),'minimum':min(values),'below_0_9':sum(v<.9 for v in values)},
        'required_in_scope_geomean_for_combined_one_if_historical_large_ratio_held':math.exp(-len(values)*math.log(geo)/args.in_scope_count),
        'inputs':{'large_index_sha256':hashlib.sha256(args.large_index.read_bytes()).hexdigest(),'historical_analysis_sha256':hashlib.sha256(args.historical_analysis.read_bytes()).hexdigest()},
        'method':'Historical warm-Graph fixed-candidate timings; ratio best measured nonhybrid/default AtBSR. The required in-scope value is only an algebraic scenario, not a forecast: historical data predates counterbalanced construction, current code, and independent large-case remeasurement.'}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
