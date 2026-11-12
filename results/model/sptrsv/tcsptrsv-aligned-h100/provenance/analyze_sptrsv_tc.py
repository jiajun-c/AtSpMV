#!/usr/bin/env python3
"""Conservative paired comparison; reject TC results failing reference check."""
import argparse,csv,io,json,math,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--aligned',action='store_true');a=p.parse_args();root=a.directory
names=json.loads((root/'manifest.json').read_text())['names'];rows=[]
for name in names:
 row={'name':name,'status':'','plan_backend':'','plan_host_us':'','tc_host_us':'','speedup':'','max_error':''}
 if (root/'cases'/(name+'-skip.json')).exists():row['status']='factor_missing';rows.append(row);continue
 vals=[];reason=''
 for rnd in range(2):
  pf=root/'cases'/f'{name}-plan-{rnd}.json';tf=root/'cases'/f'{name}-tc-{rnd}.json'
  if not pf.exists() or not tf.exists():reason='pending_or_missing';break
  pp=json.loads(pf.read_text());tt=json.loads(tf.read_text())
  if pp.get('returncode')!=0:reason='plan_failed';break
  if tt.get('returncode')!=0:reason='tc_failed_or_timeout';break
  modes={r['mode']:r for r in csv.DictReader(io.StringIO(pp.get('stdout','')))}
  if 'model-host' not in modes:reason='plan_timing_missing';break
  text=tt.get('stdout',''); m=re.search(r'time_cuda_solve = ([0-9.eE+-]+) ms',text)
  v=re.search(r'VALIDATION finite=(\d+) max_error=([0-9.eE+-]+) reference_scale=([0-9.eE+-]+)',text)
  if not m or not v:reason='tc_output_missing';break
  err,scale=float(v.group(2)),float(v.group(3))
  if v.group(1)!='1' or not math.isfinite(err) or err>1e-10*max(1.,scale):reason='tc_incorrect';break
  vals.append((modes['model-host']['backend'],float(modes['model-host']['solve_us']),float(m.group(1))*1000,err))
 if reason:row['status']=reason;rows.append(row);continue
 if vals[0][0]!=vals[1][0]:row['status']='inconsistent_backend';rows.append(row);continue
 row.update(status='ok',plan_backend=vals[0][0],plan_host_us=statistics.median(v[1] for v in vals),tc_host_us=statistics.median(v[2] for v in vals),max_error=max(v[3] for v in vals))
 row['speedup']=row['tc_host_us']/row['plan_host_us'];rows.append(row)
with (root/'analysis.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ok=[r for r in rows if r['status']=='ok']; summary={'manifest_count':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'valid_pairs':len(ok),'plan_wins':sum(r['speedup']>1 for r in ok),'geomean_tc_over_plan':math.exp(sum(math.log(r['speedup']) for r in ok)/len(ok)) if ok else None,'caveat':('locally modified TC-SpTRSV and Plan CUDA 12.6; identical unit-diagonal factors padded to 16; synchronized host timing includes per-solve state, excludes preprocessing; two rounds, invalid omitted' if a.aligned else 'locally modified TC-SpTRSV, CUDA 12.6 vs Plan CUDA 12.4; unit-diagonal factors, TC pads to multiple of 16; synchronized host timing includes TC per-solve state initialization, excludes preprocessing; two rounds, invalid omitted')}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
