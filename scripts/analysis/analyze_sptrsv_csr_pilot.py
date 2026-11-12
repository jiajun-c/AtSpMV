#!/usr/bin/env python3
"""Compare fixed cached CSR configuration to the current Plan choice."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','baseline':'','baseline_us':'','csr_us':'','speedup':'','cusparse_us':''};runs=[]
    for order in ('forward','reverse'):
        f=root/'cases'/f'{name}-{order}.json'
        if not f.exists():row['status']='missing';break
        d=json.loads(f.read_text())
        if d.get('returncode')!=0:row['status']='failed';break
        try:
            modes={r['variant']:r for r in csv.DictReader(io.StringIO(d['stdout']))};csr=modes['cached-csr-32-256'];n=int(csr['n']);levels=int(csr['levels']);scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.
            if 'selector-cached' in modes:
                if n//levels>=32:raise ValueError('current Plan uses 1024 threads; unavailable in this pilot')
                baseline='cached-tiles-512'
            elif 'selector-cusparse' in modes:baseline='cusparse'
            else:raise ValueError('selector missing')
            vals=[]
            for key in (baseline,'cached-csr-32-256','cusparse'):
                r=modes[key];t=float(r['solve_us']);e=float(r['max_error'])
                if not math.isfinite(t) or t<=0 or not math.isfinite(e) or e<0 or e>1e-10*max(1.,scale):raise ValueError('invalid result')
                vals.append(t)
            runs.append((baseline,*vals))
        except (KeyError,ValueError,TypeError,ZeroDivisionError):row['status']='invalid_or_unsupported';break
    if row['status']=='ok':
        if runs[0][0]!=runs[1][0]:row['status']='inconsistent';rows.append(row);continue
        row['baseline']=runs[0][0];row['baseline_us']=statistics.median(x[1] for x in runs);row['csr_us']=statistics.median(x[2] for x in runs);row['cusparse_us']=statistics.median(x[3] for x in runs);row['speedup']=row['baseline_us']/row['csr_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];v=[r['speedup'] for r in ok];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),
    'geomean_baseline_over_csr':math.exp(statistics.mean(map(math.log,v))) if v else None,'wins_gt_5pct':sum(x>1.05 for x in v),'losses_gt_5pct':sum(x<1/1.05 for x in v)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
