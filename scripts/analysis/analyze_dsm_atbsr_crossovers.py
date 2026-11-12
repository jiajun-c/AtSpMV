#!/usr/bin/env python3
"""Summarize completed crossover retests, preserving old/new candidate distinction."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--full-config',type=Path,required=True);a=p.parse_args();root=a.directory
    meta=json.loads((root/'manifest.json').read_text())
    if not meta.get('complete'):raise RuntimeError('crossover validation has not completed')
    old=set()
    for row in csv.reader(a.full_config.open()):
        if row and row[0]=='hybrid_dsm_atbsr':old.add('/'.join(row[1:]))
    failed={x['dataset'] for x in meta.get('failures',[])};out=[]
    for case in meta['cases']:
        name=case['dataset']
        if name in failed:continue
        groups=collections.defaultdict(list)
        for seed in meta['seeds']:
            samples=collections.defaultdict(list)
            with (root/f'{name}-{seed}.csv').open() as f:
                for r in csv.DictReader(f):samples[(r['backend'],r['config'])].append(float(r['us']))
            for k,v in samples.items():
                if len(v)!=21:raise RuntimeError(f'incomplete trials: {name}, {seed}, {k}')
                groups[k].append(statistics.median(v))
        configs={k:statistics.median(v) for k,v in groups.items() if len(v)==len(meta['seeds'])}
        if len(configs)!=len(groups):raise RuntimeError('incomplete seeds for '+name)
        non=min((v,k) for k,v in configs.items() if k[0]!='hybrid_dsm_atbsr')
        previous=min((v,k) for k,v in configs.items() if k[0]=='hybrid_dsm_atbsr' and k[1] in old)
        new=min((v,k) for k,v in configs.items() if k[0]=='hybrid_dsm_atbsr' and k[1] not in old)
        best=min(previous,new)
        out.append({'dataset':name,'reason':case['reason'],'max_row_nnz':case['max_row_nnz'],'first_pass_ratio':case['initial_speedup'],
                    'nonhybrid_us':non[0],'nonhybrid_backend':non[1][0],'nonhybrid_config':non[1][1],
                    'old_hybrid_us':previous[0],'old_hybrid_config':previous[1][1],'new_hybrid_us':new[0],'new_hybrid_config':new[1][1],
                    'best_hybrid_us':best[0],'old_speedup':non[0]/previous[0],'new_speedup':non[0]/best[0],
                    'old_vs_new_hybrid':previous[0]/new[0],
                    'new_us_per_seed':groups[new[1]],'old_us_per_seed':groups[previous[1]],'nonhybrid_us_per_seed':groups[non[1]]})
    (root/'summary.json').write_text(json.dumps({'rows':out,'failures':meta.get('failures',[]),'notes':'Ratios use median over seed medians, with the best fixed configuration chosen after retest. Crossover cases were selected from the full pass and are not a random holdout; optimization changes group/threads/tile as well as threshold.'},indent=2)+'\n')
    if out:
        scalar=[k for k,v in out[0].items() if not isinstance(v,list)]
        with (root/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=scalar,extrasaction='ignore');w.writeheader();w.writerows(out)
        print('completed',len(out),'old wins',sum(r['old_speedup']>1 for r in out),'new wins',sum(r['new_speedup']>1 for r in out),'new configuration improves hybrid >=5%',sum(r['old_vs_new_hybrid']>=1/.95 for r in out))
        for r in sorted(out,key=lambda x:x['new_speedup'],reverse=True)[:10]:print(r['dataset'],round(r['old_speedup'],3),round(r['new_speedup'],3),r['nonhybrid_backend'])
if __name__=='__main__':main()
