#!/usr/bin/env python3
"""Analyze successful full-suite cases without hiding missing and failed cases."""
import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path
import statistics


def geo(xs):return math.exp(statistics.mean(map(math.log,xs))) if xs else None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text());meta=json.loads((root/'metadata.json').read_text());out=[];counts=collections.Counter();failed=[]
    for m in manifest:
        name=m['dataset'];p=root/'status'/f'{name}.json'
        if not p.exists():counts['pending']+=1;continue
        status=json.loads(p.read_text());kind=status['status'];counts[kind]+=1
        if kind!='ok':failed.append({'dataset':name,'status':kind,'returncode':status.get('returncode')});continue
        grouped=collections.defaultdict(list)
        with (root/'raw'/f'{name}.csv').open() as f:
            for r in csv.DictReader(f):grouped[(r['backend'],r['config'])].append(float(r['us']))
        best={};allmed={}
        for (backend,config),values in grouped.items():
            if len(values)!=meta['trials'] or not all(math.isfinite(x) and x>0 for x in values):raise ValueError(f'{name}: bad sample')
            t=statistics.median(values);allmed[backend,config]=t
            if backend not in best or t<best[backend][0]:best[backend]=(t,config)
        for k in ('atbsr','csr_dsm','csr_two_stage','hybrid_dsm_atbsr','cusparse'):
            if k not in best:raise ValueError(f'{name}: missing {k}')
        h,hc=best['hybrid_dsm_atbsr'];base,bk=min((v[0],k) for k,v in best.items() if k!='hybrid_dsm_atbsr')
        orig=next((v for (k,c),v in allmed.items() if k=='atbsr' and c.startswith('-1/-1/')),None)
        out.append({'dataset':name,'rows':m['rows'],'cols':m['columns'],'nnz':m['nnz'],'max_row_nnz':m['longest_row_nnz'],'max_row_ratio':m['ratio'],
            'hybrid_us':h,'hybrid_config':hc,'atbsr_default_us':orig,'atbsr_best_us':best['atbsr'][0],
            'dsm_us':best['csr_dsm'][0],'two_stage_us':best['csr_two_stage'][0],'cusparse_us':best['cusparse'][0],
            'best_nonhybrid':bk,'best_nonhybrid_us':base,'speedup_vs_best_nonhybrid':base/h,
            'speedup_vs_atbsr_default':orig/h if orig else None,'elapsed_s':status['elapsed_s']})
    if out:
        tmp=root/f'analysis.{os.getpid()}.csv.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        os.replace(tmp,root/'analysis.csv')
        h=[r['speedup_vs_best_nonhybrid'] for r in out];defh=[r['speedup_vs_atbsr_default'] for r in out if r['speedup_vs_atbsr_default'] is not None]
        groups={}
        for label,func in [('no_long',lambda x:int(x['max_row_nnz'])<1024),('has_long',lambda x:int(x['max_row_nnz'])>=1024)]:
            values=[r['speedup_vs_best_nonhybrid'] for r in out if func(r)]
            groups[label]={'count':len(values),'wins':sum(v>1 for v in values),'median':statistics.median(values) if values else None,'geomean':geo(values)}
        metrics={'cases':len(out),'hybrid_wins':sum(v>1 for v in h),'wins_at_least_5pct_faster':sum(v>=1/.95 for v in h),'slower_more_5pct':sum(v<1/1.05 for v in h),'median':statistics.median(h),'geomean':geo(h),'min':min(h),'max':max(h),'vs_default_atbsr_geomean':geo(defh),'groups':groups}
    else:metrics={}
    payload={'status_counts':dict(counts),'total_manifest':len(manifest),'metrics':metrics,'failures':failed,
      'method':'Best of fixed hybrid shortlist vs best fixed non-hybrid shortlist and cuSPARSE CSR_ALG1; 9 interleaved trials of 8 warm Graph calls per config. No preprocessing cost. Early results are sorted by nnz and are not representative of the whole suite.'}
    tmp=root/f'analysis.{os.getpid()}.json.tmp';tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,root/'analysis.json');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
