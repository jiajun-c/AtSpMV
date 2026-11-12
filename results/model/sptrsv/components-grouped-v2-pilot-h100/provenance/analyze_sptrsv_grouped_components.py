#!/usr/bin/env python3
"""Compare grouped-singleton and one-CTA-per-component solves."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','baseline':'','baseline_us':'','ungrouped_us':'','grouped_us':'','cusparse_us':'','grouped_vs_ungrouped':'','grouped_vs_baseline':''};runs=[]
    for order in ('tile-first','components-first'):
        path=root/'cases'/f'{name}-{order}.json'
        if not path.exists():row['status']='missing';break
        d=json.loads(path.read_text())
        if d.get('returncode')!=0:row['status']='failed';break
        try:
            modes={r['variant']:r for r in csv.DictReader(io.StringIO(d['stdout']))};group=modes['grouped-components-512'];n=int(group['n']);levels=int(group['levels'])
            tile='cached-tiles-1024' if n//levels>=32 else 'cached-tiles-512'
            baseline=tile if n<=4000 and levels>=8 and n/levels<=256 and tile in modes else 'cusparse'
            names=(baseline,'cached-components-512','grouped-components-512','cusparse');v=[]
            scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.
            for k in names:
                t=float(modes[k]['solve_us']);err=float(modes[k]['max_error'])
                if not math.isfinite(t) or t<=0 or not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale):raise ValueError()
                v.append(t)
            runs.append((baseline,*v))
        except (KeyError,ValueError,TypeError,ZeroDivisionError):row['status']='invalid_or_unsupported';break
    if row['status']=='ok':
        if runs[0][0]!=runs[1][0]:row['status']='inconsistent';rows.append(row);continue
        row['baseline']=runs[0][0]
        for i,k in enumerate(('baseline_us','ungrouped_us','grouped_us','cusparse_us'),start=1):row[k]=statistics.median(v[i] for v in runs)
        row['grouped_vs_ungrouped']=row['ungrouped_us']/row['grouped_us'];row['grouped_vs_baseline']=row['baseline_us']/row['grouped_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok)}
for key in ('grouped_vs_ungrouped','grouped_vs_baseline'):
    v=[r[key] for r in ok];summary[key+'_geomean']=math.exp(statistics.mean(map(math.log,v))) if v else None
    summary[key+'_wins_gt_5pct']=sum(x>1.05 for x in v);summary[key+'_losses_gt_5pct']=sum(x<1/1.05 for x in v)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
