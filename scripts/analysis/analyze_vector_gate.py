#!/usr/bin/env python3
"""Retrospectively compare margins using frozen vector candidate timings."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.directory
rows=list(csv.DictReader((root/'frozen-model-results.csv').open()))
status=json.loads((root/'frozen-model-evaluation.json').read_text())
if not status['complete_success'] or len(rows)!=status['manifest_total']:
    raise ValueError('requires complete successful frozen evaluation')
baselines=('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse')
orders=('baseline_first','selection_first')
report={'cases':len(rows),'method':'Post-hoc threshold sweep over the same measured candidate times. Selected predictor and training stay fixed; margins were inspected after this sample and are not independent estimates. Warm Graph excludes setup.', 'margins':{}}
for margin in (0.,.02,.05,.1):
    model_ratios=[];fixed_ratios=[];selected=regressions=0
    for row in rows:
        samples=collections.defaultdict(list)
        for item in csv.DictReader((root/'raw'/f"{row['dataset']}.csv").open()):samples[item['order'],item['backend']].append(float(item['us']))
        times={k:statistics.median(v) for k,v in samples.items()}
        choose=row['chosen']!='auto_model' and float(row['predicted_ratio_to_default']) < (1-margin)*float(row['fallback_predicted_ratio_to_default'])
        kind=row['chosen'] if choose else 'auto_model';selected+=choose
        speed=math.exp(statistics.mean(math.log(times[o,'auto_model']/times[o,kind]) for o in orders))
        fixed=math.exp(statistics.mean(math.log(min(times[o,b] for b in baselines)/times[o,kind]) for o in orders))
        model_ratios.append(speed);fixed_ratios.append(fixed);regressions+=choose and speed<1
    report['margins'][str(margin)]={'selected_vector':selected,'regressions_vs_model':regressions,
        'geomean_vs_model':math.exp(statistics.mean(map(math.log,model_ratios))),
        'geomean_vs_fixed_oracle':math.exp(statistics.mean(map(math.log,fixed_ratios))),
        'worst_vs_model':min(model_ratios)}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
