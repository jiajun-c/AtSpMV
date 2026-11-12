#!/usr/bin/env python3
"""Summarize warm, repeated Plan construction times across process orders."""
import argparse,csv,io,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory;rows=[]
for name in json.loads((root/'manifest.json').read_text())['names']:
    row={'name':name,'status':'ok','backend':'','default_setup_us':'','model_setup_us':'','speedup':'','default_destroy_us':'','model_destroy_us':''};runs=[]
    for order in ('forward','reverse'):
        f=root/'cases'/f'{name}-{order}.json'
        if not f.exists():row['status']='missing';break
        d=json.loads(f.read_text())
        if d.get('returncode')!=0:row['status']='failed';break
        try:
            m={r['mode']:r for r in csv.DictReader(io.StringIO(d['stdout']))}
            for mode in ('default','model'):
                for key in ('setup_median_us','destroy_median_us'):
                    x=float(m[mode][key]);
                    if not math.isfinite(x) or x<=0:raise ValueError('invalid timing')
                if int(m[mode]['iterations'])!=12:raise ValueError('wrong iteration count')
            runs.append(m)
        except (KeyError,ValueError,TypeError):row['status']='invalid';break
    if row['status']=='ok':
        b={x['model']['backend'] for x in runs}
        if len(b)!=1:row['status']='inconsistent';rows.append(row);continue
        row['backend']=b.pop()
        for mode in ('default','model'):
            row[f'{mode}_setup_us']=statistics.median(float(x[mode]['setup_median_us']) for x in runs)
            row[f'{mode}_destroy_us']=statistics.median(float(x[mode]['destroy_median_us']) for x in runs)
        row['speedup']=row['default_setup_us']/row['model_setup_us']
    rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];summary={'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),'by_backend':{}}
for b in sorted({r['backend'] for r in ok}):
    q=[r for r in ok if r['backend']==b];summary['by_backend'][b]={'count':len(q),'default_median_us':statistics.median(r['default_setup_us'] for r in q),'model_median_us':statistics.median(r['model_setup_us'] for r in q),'geomean_default_over_model':math.exp(statistics.mean(math.log(r['speedup']) for r in q))}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
