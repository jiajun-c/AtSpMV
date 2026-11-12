#!/usr/bin/env python3
"""Analyze paired forward/reverse Plan comparisons, including skipped cases."""
import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--forward-tolerance',type=float,default=None);a=p.parse_args();root=a.directory
if a.forward_tolerance is not None and (not math.isfinite(a.forward_tolerance) or a.forward_tolerance <= 0):
    p.error('--forward-tolerance must be finite and positive')
names=json.loads((root/'manifest.json').read_text())['names'];rows=[]
for name in names:
    row={'name':name,'status':'','backend':'','default_us':'','model_us':'','speedup':'',
         'default_setup_us':'','model_setup_us':'','break_even_calls':'', 'model_setup_delta_us':''}
    values=[]
    for order in ('forward','reverse'):
        path=root/'cases'/(name+'-'+order+'.json')
        if not path.exists(): break
        data=json.loads(path.read_text())
        if data.get('returncode')!=0: break
        variants={r['mode']:r for r in csv.DictReader(io.StringIO(data.get('stdout',''))) if r['mode'] in ('default','model')}
        if set(variants)!={'default','model'}:break
        values.append(variants)
    if len(values)!=2:
        row['status']='factor_missing' if (root/'cases'/(name+'-skip.json')).exists() else 'benchmark_failed_or_incomplete'
        rows.append(row);continue
    if a.forward_tolerance is not None:
        try:
            checks=[(int(v[mode]['rows']),float(v[mode]['max_error'])) for v in values for mode in ('default','model')]
            good=all(n>0 and math.isfinite(err) and err>=0 and err<=a.forward_tolerance*max(1.,.5+max((37*i)%101 for i in range(min(n,101)))/101.) for n,err in checks)
        except (KeyError,ValueError,TypeError):
            good=False
        if not good:
            row['status']='forward_error_exceeds_tolerance';rows.append(row);continue
    if values[0]['model']['backend']!=values[1]['model']['backend']:
        row['status']='inconsistent_backend';rows.append(row);continue
    row['status']='ok';row['backend']=values[0]['model']['backend']
    for mode in ('default','model'):
        row[mode+'_us']=statistics.median(float(v[mode]['solve_us']) for v in values)
        row[mode+'_setup_us']=statistics.median(float(v[mode]['setup_us']) for v in values)
    row['speedup']=row['default_us']/row['model_us'];row['model_setup_delta_us']=row['model_setup_us']-row['default_setup_us']
    if row['model_us']<row['default_us']:
        row['break_even_calls']=max(0.,row['model_setup_delta_us'])/(row['default_us']-row['model_us'])
    rows.append(row)
suffix='-forward' if a.forward_tolerance is not None else ''
with (root/('analysis'+suffix+'.csv')).open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];cached=[r for r in ok if r['backend']=='cached'];cached_csr=[r for r in ok if r['backend']=='cached-csr']
sumry={'manifest_count':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
       'cached':len(cached),'cached_csr':len(cached_csr),'fallback':sum(r['backend']=='cusparse' for r in ok),
       'solve_geomean':math.exp(sum(math.log(r['speedup']) for r in ok)/len(ok)) if ok else None,
       'cached_wins':sum(r['speedup']>1 for r in cached),
       'cached_slower_gt_5pct':sum(r['speedup']<1/1.05 for r in cached),
       'cached_csr_wins':sum(r['speedup']>1 for r in cached_csr),
       'cached_csr_slower_gt_5pct':sum(r['speedup']<1/1.05 for r in cached_csr),
       'cached_setup_delta_median_us':statistics.median(r['model_setup_delta_us'] for r in cached) if cached else None,
       'note':'median of two process runs, opposite constructor order; warm-library host setup and warm Graph solve. Does not include factorization or graph construction. Factorization failures retained.'}
if a.forward_tolerance is not None:sumry['forward_tolerance']=a.forward_tolerance
(root/('summary'+suffix+'.json')).write_text(json.dumps(sumry,indent=2)+'\n');print(json.dumps(sumry,indent=2))
