#!/usr/bin/env python3
"""Analyze paired forward/reverse Plan comparisons, including skipped cases."""
import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
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
        variants={r['mode']:r for r in csv.DictReader(io.StringIO(data.get('stdout','')))}
        if set(variants)!={'default','model'}:break
        values.append(variants)
    if len(values)!=2:
        row['status']='factor_missing' if (root/'cases'/(name+'-skip.json')).exists() else 'benchmark_failed_or_incomplete'
        rows.append(row);continue
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
with (root/'analysis.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
ok=[r for r in rows if r['status']=='ok'];cached=[r for r in ok if r['backend']=='cached']
sumry={'manifest_count':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
       'cached':len(cached),'fallback':len(ok)-len(cached),
       'solve_geomean':math.exp(sum(math.log(r['speedup']) for r in ok)/len(ok)) if ok else None,
       'cached_wins':sum(r['speedup']>1 for r in cached),
       'cached_slower_gt_5pct':sum(r['speedup']<1/1.05 for r in cached),
       'cached_setup_delta_median_us':statistics.median(r['model_setup_delta_us'] for r in cached) if cached else None,
       'note':'median of two process runs, opposite constructor order; warm-library host setup and warm Graph solve. Does not include factorization or graph construction. Factorization failures retained.'}
(root/'summary.json').write_text(json.dumps(sumry,indent=2)+'\n');print(json.dumps(sumry,indent=2))
