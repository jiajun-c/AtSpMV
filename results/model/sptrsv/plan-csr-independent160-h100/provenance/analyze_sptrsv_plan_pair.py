#!/usr/bin/env python3
"""Analyze paired full Plan benchmarks with a common forward-error gate."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--allow-backend-change',action='store_true');a=p.parse_args();root=a.directory
rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','backend':'','fixed_backend':'','threads':'','fixed_us':'','adaptive_us':'','speedup':''};values=[]
    for round_index in range(2):
        pair={}
        for which in ('fixed','adaptive'):
            path=root/'cases'/f'{name}-{which}-{round_index}.json'
            if not path.exists():row['status']='factor_missing' if (root/'cases'/f'{name}-skip.json').exists() else 'missing';break
            data=json.loads(path.read_text())
            if data.get('returncode')!=0:row['status']='failed';break
            try:
                modes={r['mode']:r for r in csv.DictReader(io.StringIO(data['stdout']))};m=modes['model']
                n=int(m['rows']);err=float(m['max_error']);t=float(m['solve_us'])
                scale=.5+max((37*i)%101 for i in range(min(n,101)))/101.
                if n<=0 or not math.isfinite(err) or err<0 or err>1e-10*max(1.,scale) or not math.isfinite(t) or t<=0:raise ValueError()
                pair[which]=m
            except (KeyError,ValueError,TypeError):row['status']='invalid';break
        if row['status']!='ok':break
        if not a.allow_backend_change and pair['fixed']['backend']!=pair['adaptive']['backend']:row['status']='backend_mismatch';break
        values.append(pair)
    if row['status']=='ok':
        backends={p['adaptive']['backend'] for p in values};fixed_backends={p['fixed']['backend'] for p in values};threads={p['adaptive']['cached_threads'] for p in values}
        if len(backends)!=1 or len(fixed_backends)!=1 or len(threads)!=1:row['status']='inconsistent';rows.append(row);continue
        row['backend']=backends.pop();row['fixed_backend']=fixed_backends.pop();row['threads']=threads.pop()
        row['fixed_us']=statistics.median(float(p['fixed']['solve_us']) for p in values)
        row['adaptive_us']=statistics.median(float(p['adaptive']['solve_us']) for p in values)
        row['speedup']=row['fixed_us']/row['adaptive_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
    'valid_pairs':len(ok),'adaptive_1024':sum(r['threads']=='1024' and r['backend']=='cached' for r in ok),'cached_csr':sum(r['backend']=='cached-csr' for r in ok),
    'geomean_fixed_over_adaptive':math.exp(statistics.mean(math.log(r['speedup']) for r in ok)) if ok else None,
    'wins_gt_5pct':sum(r['speedup']>1.05 for r in ok),'losses_gt_5pct':sum(r['speedup']<1/1.05 for r in ok)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
