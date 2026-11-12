#!/usr/bin/env python3
"""Summarize isolated crossover validation and alternating Auto plan tests."""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--auto-only',action='store_true');p.add_argument('--skip-auto',action='store_true');p.add_argument('--auto-names',nargs='+',default=['t60k','c-53','a0nsdsil']);p.add_argument('--case-names',nargs='+',default=['c-53','rajat20','a0nsdsil']);a=p.parse_args();root=a.directory
    meta=json.loads((root/'metadata.json').read_text())
    if not meta.get('complete'):raise RuntimeError('validation has not completed')
    result={}
    for name in (() if a.auto_only else a.case_names):
        vals=collections.defaultdict(list)
        for seed in (202610281,202610282,202610283):
            groups=collections.defaultdict(list)
            with (root/f'{name}-{seed}.out').open() as f:
                for r in csv.DictReader(f):groups[(r['backend'],r['config'])].append(float(r['us']))
            for k,v in groups.items():
                if len(v)!=21:raise ValueError(f'missing trials: {name} {k}')
                vals[k].append(statistics.median(v))
        result[name]={'candidates':[{'backend':b,'config':c,'us':statistics.median(v),'per_seed_us':v} for (b,c),v in sorted(vals.items(),key=lambda kv:statistics.median(kv[1]))]}
    auto={}
    for name in (() if a.skip_auto else a.auto_names):
        auto[name]={}
        for mode in ('baseline','optimized'):
            runs=[]
            for rep in range(3):
                with (root/f'{name}-auto-{mode}-{rep}.out').open() as f:
                    line=f.readline();rows=list(csv.DictReader(f))
                setup=float(re.search(r'setup_us=([\d.eE+-]+)',line).group(1))
                selected=re.search(r'selected=([^,]+)',line).group(1)
                runs.append({'setup_us':setup,'selected':selected,'candidates':len(rows),'best_candidate_us':min(float(r['median_us']) for r in rows)})
            auto[name][mode]={'runs':runs,'median_setup_us':statistics.median(r['setup_us'] for r in runs),'median_best_candidate_us':statistics.median(r['best_candidate_us'] for r in runs)}
    out={'crossover':result,'automatic':auto,'notes':'Fixed crossover candidates, 3 independent order seeds and 21 warm graph batches, using the optimized executable. Auto setup includes selection and is not included in per-call execution time. Best reported per dataset is within listed candidates only.'}
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,v in result.items():
        print(name,'best candidates',[(x['backend'],x['config'],round(x['us'],3)) for x in v['candidates'][:4]])
    for name,v in auto.items():print(name,'median setup µs',round(v['baseline']['median_setup_us']),round(v['optimized']['median_setup_us']))
if __name__=='__main__':main()
