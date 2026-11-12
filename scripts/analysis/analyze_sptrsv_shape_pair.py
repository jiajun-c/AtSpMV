#!/usr/bin/env python3
"""Analyze counterbalanced usual vs narrow-level K=32 tile packing."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','usual_us':'','narrow_us':'','speedup':''};runs=[]
    for rnd in range(2):
        vals={}
        for mode in ('usual','narrow'):
            f=root/'cases'/f'{name}-{mode}-{rnd}.json'
            if not f.exists():row['status']='missing';break
            d=json.loads(f.read_text())
            if d.get('returncode')!=0:row['status']='failed';break
            try:
                modes={r['variant']:r for r in csv.DictReader(io.StringIO(d['stdout']))};r=modes['cached-tiles-512'];n=int(r['n'])
                scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.;t=float(r['solve_us']);err=float(r['max_error'])
                if not math.isfinite(t) or t<=0 or not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale):raise ValueError()
                vals[mode]=t
            except (KeyError,ValueError,TypeError):row['status']='invalid_or_unsupported';break
        if row['status']!='ok':break
        runs.append(vals)
    if row['status']=='ok':
        row['usual_us']=statistics.median(v['usual'] for v in runs);row['narrow_us']=statistics.median(v['narrow'] for v in runs);row['speedup']=row['usual_us']/row['narrow_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];v=[r['speedup'] for r in ok];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),
    'geomean_usual_over_narrow':math.exp(statistics.mean(map(math.log,v))) if v else None,'wins_gt_5pct':sum(x>1.05 for x in v),'losses_gt_5pct':sum(x<1/1.05 for x in v)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
