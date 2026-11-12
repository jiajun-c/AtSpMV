#!/usr/bin/env python3
"""Compare cached components to the currently selected Plan, in two orders."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','baseline':'','baseline_us':'','components_us':'','speedup':'','cusparse_us':''};values=[]
    for order in ('tile-first','components-first'):
        path=root/'cases'/f'{name}-{order}.json'
        if not path.exists():row['status']='missing';break
        data=json.loads(path.read_text())
        if data.get('returncode')!=0:row['status']='failed';break
        try:
            modes={r['variant']:r for r in csv.DictReader(io.StringIO(data['stdout']))}
            component=modes['cached-components-512'];cu=modes['cusparse']
            n=int(component['n']);levels=int(component['levels'])
            tile=('cached-tiles-1024' if n//levels>=32 else 'cached-tiles-512')
            # H100's Plan also rejects insufficient depth, overly wide levels,
            # rows > 4000 and packs that do not fit the shared-memory limit.
            baseline=tile if n<=4000 and levels>=8 and n/levels<=256 and tile in modes else 'cusparse'
            base=modes[baseline]
            scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.
            for r in (component,cu,base):
                err=float(r['max_error']);t=float(r['solve_us'])
                if not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale) or not math.isfinite(t) or t<=0:raise ValueError()
            values.append((baseline,float(base['solve_us']),float(component['solve_us']),float(cu['solve_us'])))
        except (KeyError,ValueError,TypeError,StopIteration,ZeroDivisionError):row['status']='invalid_or_unsupported';break
    if row['status']=='ok':
        if values[0][0]!=values[1][0]:row['status']='inconsistent';rows.append(row);continue
        row['baseline']=values[0][0];row['baseline_us']=statistics.median(v[1] for v in values);row['components_us']=statistics.median(v[2] for v in values);row['cusparse_us']=statistics.median(v[3] for v in values);row['speedup']=row['baseline_us']/row['components_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),
    'geomean_baseline_over_components':math.exp(statistics.mean(math.log(r['speedup']) for r in ok)) if ok else None,
    'wins_gt_5pct':sum(r['speedup']>1.05 for r in ok),'losses_gt_5pct':sum(r['speedup']<1/1.05 for r in ok)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
