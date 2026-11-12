#!/usr/bin/env python3
"""Analyze two-order component-bin experiments against a single launch."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','cusparse_us':'','single_us':'','bin128_us':'','bin256_us':'','bin128_speedup':'','bin256_speedup':''};runs=[]
    for order in ('tile-first','components-first'):
        f=root/'cases'/f'{name}-{order}.json'
        if not f.exists():row['status']='missing';break
        d=json.loads(f.read_text())
        if d.get('returncode')!=0:row['status']='failed';break
        try:
            m={r['variant']:r for r in csv.DictReader(io.StringIO(d['stdout']))};keys=('cusparse','cached-components-512','binned-components-128','binned-components-256')
            n=int(m['cusparse']['n']);scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.;vals=[]
            for k in keys:
                t=float(m[k]['solve_us']);err=float(m[k]['max_error'])
                if not math.isfinite(t) or t<=0 or not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale):raise ValueError()
                vals.append(t)
            runs.append(vals)
        except (KeyError,ValueError,TypeError):row['status']='invalid_or_unsupported';break
    if row['status']=='ok':
        for i,k in enumerate(('cusparse_us','single_us','bin128_us','bin256_us')):row[k]=statistics.median(v[i] for v in runs)
        for k in (128,256):row[f'bin{k}_speedup']=row['single_us']/row[f'bin{k}_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok)}
for k in (128,256):
    vals=[r[f'bin{k}_speedup'] for r in ok];summary[f'bin{k}_vs_single_geomean']=math.exp(statistics.mean(map(math.log,vals))) if vals else None
    summary[f'bin{k}_wins_gt_5pct']=sum(x>1.05 for x in vals);summary[f'bin{k}_losses_gt_5pct']=sum(x<1/1.05 for x in vals)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
