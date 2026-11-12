#!/usr/bin/env python3
"""Summarize ABBA process-order initialization comparisons."""
import argparse
import json
import math
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
result={'method':'ABBA process order; geometric mean of two initialization measurements per version, then previous/current ratio. Small selected sample, process order cannot remove every system or allocation effect. Not a GPU kernel speedup.', 'cases':{}}
names=sorted({path.name.removesuffix('-0-previous.log') for path in args.directory.glob('*-0-previous.log')})
if not names:raise SystemExit('no complete ABBA cases')
for name in names:
    values={}
    for i,kind in enumerate(('previous','current','current','previous')):
        path=args.directory/f'{name}-{i}-{kind}.log'
        matches=[float(line.split(',')[2]) for line in path.read_text().splitlines() if line.startswith('PLAN_PREP,auto_tuned,')]
        if len(matches)!=1 or not math.isfinite(matches[0]) or matches[0]<=0:raise ValueError(f'{path}: bad setup timing')
        values.setdefault(kind,[]).append(matches[0])
    previous,current=(math.exp(statistics.mean(map(math.log,values[k]))) for k in ('previous','current'))
    result['cases'][name]={'previous_setup_us':previous,'current_setup_us':current,'setup_speedup':previous/current,'samples_us':values}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
