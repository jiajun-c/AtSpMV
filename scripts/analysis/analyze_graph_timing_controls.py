#!/usr/bin/env python3
"""Summarize 2x2 Graph sharing and timing-event controls."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.directory
meta=json.loads((root/'metadata.json').read_text());names=sorted(meta['matrices']);groups={}
for shared in (0,1):
    for internal in (0,1):
        cases=[]
        for name in names:
            ratios=[]
            for order in ('forward','reverse'):
                label=f'{name}-graph{shared}-event{internal}-{order}'
                rows=list(csv.DictReader((root/(label+'.csv')).open()));values={}
                for backend in ('auto_vector_model','auto_vector_model_2pct'):
                    subset=[r for r in rows if r['backend']==backend]
                    if len(subset)!=21 or len({r['config'] for r in subset})!=1:raise ValueError(f'{label}: invalid samples')
                    values[backend]=(subset[0]['config'],statistics.median(float(r['us']) for r in subset))
                if values['auto_vector_model'][0]!=values['auto_vector_model_2pct'][0]:raise ValueError(f'{label}: configs differ')
                log=(root/(label+'.log')).read_text()
                if 'PLAN_ALIAS,auto_vector_model_2pct,shared' not in log or ('GRAPH_ALIAS,auto_vector_model_2pct,' in log)!=bool(shared) or ('TIMING_MODE,internal_graph_events' in log)!=bool(internal):raise ValueError(f'{label}: mode mismatch')
                ratios.append(values['auto_vector_model'][1]/values['auto_vector_model_2pct'][1])
            ratio=math.exp(statistics.mean(map(math.log,ratios)))
            cases.append({'dataset':name,'ratio':ratio,'order_ratios':dict(zip(('forward','reverse'),ratios))})
        groups[f'graph_shared={shared},events_inside={internal}']={'geomean_ratio':math.exp(statistics.mean(math.log(c['ratio']) for c in cases)),
            'median_absolute_deviation_from_one':statistics.median(abs(c['ratio']-1) for c in cases),'cases':cases}
report={'method':'Equal model configurations execute the same Plan; compare same vs distinct Graph executables and outside vs inside Graph event placement. Two construction orders and 21 samples per case. Four selected examples, so do not generalize without broader tests.', 'groups':groups}
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
