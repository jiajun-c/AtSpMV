#!/usr/bin/env python3
"""Analyze same-process cuSPARSE/Plan Graph rows from valid paired SOTA runs."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
source=list(csv.DictReader((root/'analysis.csv').open()));rows=[]
for source_row in source:
    row={'name':source_row['name'],'status':source_row['status'],'backend':'','default_us':'','model_us':'','speedup':''};runs=[]
    if row['status']!='ok':rows.append(row);continue
    for rnd in range(2):
        try:
            d=json.loads((root/'cases'/f"{row['name']}-plan-{rnd}.json").read_text())
            if d.get('returncode')!=0:raise ValueError('plan failed')
            modes={x['mode']:x for x in csv.DictReader(io.StringIO(d['stdout']))};pair=[]
            for mode in ('default','model'):
                r=modes[mode];n=int(r['rows']);err=float(r['max_error']);t=float(r['solve_us'])
                scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.
                if n<=0 or not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale) or not math.isfinite(t) or t<=0:raise ValueError('invalid result')
                pair.append(t)
            runs.append((modes['model']['backend'],*pair))
        except (OSError,KeyError,ValueError,TypeError):row['status']='plan_invalid_or_missing';break
    if row['status']=='ok':
        if runs[0][0]!=runs[1][0]:row['status']='inconsistent'
        else:
            row['backend']=runs[0][0];row['default_us']=statistics.median(x[1] for x in runs);row['model_us']=statistics.median(x[2] for x in runs);row['speedup']=row['default_us']/row['model_us']
    rows.append(row)
with (root/'cusparse_analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];v=[r['speedup'] for r in ok];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),
    'geomean_cusparse_over_plan':math.exp(statistics.mean(map(math.log,v))) if v else None,
    'plan_wins':sum(x>1 for x in v),'losses_gt_5pct':sum(x<1/1.05 for x in v),
    'note':'warm CUDA Graph solve, median of two process runs; gated by the source SOTA comparison; excludes factorization, Plan construction and Graph creation'}
(root/'cusparse_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
