#!/usr/bin/env python3
"""Summarize matched Plan comparisons against three cuSPARSE algorithms."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from benchmark_dsm_atbsr_full import write_json
from benchmark_optimization_ablation import analyze, sample_key

PLANS = ('atbsr_control','atbsr_grouped_model','auto_refined_v2','auto_refined_v2_grouped')
CUSPARSE = ('cusparse','cusparse_default','cusparse_alg2')


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def metrics(values):
    return {'n':len(values), 'geomean':geomean(values), 'median':statistics.median(values),
            'wins_over_1pct':sum(v>1.01 for v in values),
            'losses_over_1pct':sum(v<1/1.01 for v in values),
            'min':min(values), 'max':max(values)} if values else {'n':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args(); p=args.output
    manifest=json.loads((p/'manifest.json').read_text());coverage=analyze(p,manifest)
    results=[];selected={label:Counter() for label in PLANS};setup={label:[] for label in PLANS}
    sample_counts=0
    for item in manifest['matrices']:
        name=item['dataset'];status_file=p/'status'/f'{name}.json'
        if not status_file.exists() or json.loads(status_file.read_text())['status']!='ok':continue
        orders=[];layouts=[];configs=[];setups=[]
        for order in (0,1):
            raw=list(csv.DictReader((p/'raw'/f'{name}-{order}.csv').open()));sample_counts+=len(raw)
            if set(r['backend'] for r in raw)!=set(PLANS+CUSPARSE):raise ValueError('candidate coverage changed')
            times={};resolved={}
            for label in PLANS+CUSPARSE:
                values=[float(r['us']) for r in raw if r['backend']==label]
                if len(values)!=manifest['trials']:raise ValueError('trial coverage changed')
                times[label]=statistics.median(values)
                choices={r['config'] for r in raw if r['backend']==label}
                if len(choices)!=1:raise ValueError('configuration changed within process')
                resolved[label]=choices.pop()
            layout={};prep={}
            for line in (p/'raw'/f'{name}-{order}.log').read_text().splitlines():
                fields=line.split(',')
                if fields[0]=='PLAN_LAYOUT':layout[fields[1]]=list(map(int,fields[2:]))
                if fields[0]=='PLAN_PREP':prep[fields[1]]=float(fields[2])
            if set(layout)!=set(PLANS) or set(prep)!=set(PLANS):raise ValueError('missing Plan metadata')
            orders.append(times);layouts.append(layout);configs.append(resolved);setups.append(prep)
            for label in PLANS:
                backend = 'atbsr' if label in ('atbsr_control','atbsr_grouped_model') else resolved[label].split('/')[0]
                selected[label][backend]+=1
                setup[label].append(prep[label])
        row={'dataset':name,'nnz':int(item['nnz']),'rows':int(item['rows']),
             'columns':int(item['columns']),'size_bin':item['size_bin'],'has_long_row':int(item['has_long_row'])}
        for label in PLANS:
            row[label+'_us']=geomean(t[label] for t in orders)
            row[label+'_vs_cusparse_best']=geomean(min(t[b] for b in CUSPARSE)/t[label] for t in orders)
            row[label+'_vs_cusparse_alg1']=geomean(t['cusparse']/t[label] for t in orders)
            row[label+'_setup_us']=geomean(s[label] for s in setups)
            row[label+'_config_order0']=configs[0][label];row[label+'_config_order1']=configs[1][label]
            row[label+'_grouped_orders']=sum(layout[label][5] for layout in layouts)
        row['cusparse_best_us']=geomean(min(t[b] for b in CUSPARSE) for t in orders)
        row['grouped_vs_original_atbsr']=geomean(t['atbsr_control']/t['atbsr_grouped_model'] for t in orders)
        row['unified_grouped_vs_unified']=geomean(t['auto_refined_v2']/t['auto_refined_v2_grouped'] for t in orders)
        results.append(row)
    if results:
        with (p/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={'coverage':coverage,'raw_timings':sample_counts,'full_vector_checks':sample_counts//manifest['trials']*3,
             'scope':manifest['scope'],'plans':{},'strata':{},'selected_backend_counts_per_process':selected,
             'ratios':{name:metrics([r[name] for r in results]) for name in ['grouped_vs_original_atbsr','unified_grouped_vs_unified']},
             'measurement':'FP64 warm Graph, best of three measured cuSPARSE algorithms per process, paired opposite construction orders. Setup and tuning excluded from speedups; no global-best claim.'}
    for label in PLANS:
        summary['plans'][label]={'vs_best_cusparse':metrics([r[label+'_vs_cusparse_best'] for r in results]),
                                 'vs_alg1':metrics([r[label+'_vs_cusparse_alg1'] for r in results]),
                                 'median_setup_us':statistics.median(setup[label]) if setup[label] else None,
                                 'grouped_plan_instances':sum(r[label+'_grouped_orders'] for r in results)}
    for size in range(4):
        subset=[r for r in results if r['size_bin']==size]
        summary['strata'][str(size)]={label:metrics([r[label+'_vs_cusparse_best'] for r in subset]) for label in PLANS}
    write_json(p/'comparison-summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
