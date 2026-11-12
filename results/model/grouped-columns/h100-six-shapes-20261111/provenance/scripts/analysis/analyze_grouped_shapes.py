#!/usr/bin/env python3
"""Compare grouped and original columns at each identical ordinary tile shape."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from benchmark_dsm_atbsr_full import write_json
from benchmark_optimization_ablation import analyze


def gm(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def key(config):
    return config[0]+':'+('auto' if config[0]=='atbsr_control' else '/'.join(map(str,config[1:])))


def aggregate(values):
    return {'n':len(values),'geomean':gm(values) if values else None,
            'wins_over_1pct':sum(v>1.01 for v in values),'losses_over_1pct':sum(v<1/1.01 for v in values)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);a=parser.parse_args()
    p=a.output;manifest=json.loads((p/'manifest.json').read_text());coverage=analyze(p,manifest)
    raw=list(csv.DictReader((p/'timings.csv').open()));shapes=[];matrices=[]
    for item in manifest['matrices']:
        name=item['dataset'];data={r['candidate']:r for r in raw if r['dataset']==name}
        if not data:continue
        orders=[{k:float(r[f'order{o}_us']) for k,r in data.items()} for o in (0,1)]
        layouts=[]
        for o in (0,1):
            configs=item['configs'] if o==0 else list(reversed(item['configs']))
            lines=[l.split(',') for l in (p/'raw'/f'{name}-{o}.log').read_text().splitlines() if l.startswith('PLAN_LAYOUT,')]
            if len(lines)!=len(configs):raise ValueError('layout coverage differs')
            layout={}
            for c,line in zip(configs,lines):
                if c[0]!=line[1]:raise ValueError('layout order differs')
                layout[key(c)]=list(map(int,line[2:]))
            layouts.append(layout)
        if layouts[0]!=layouts[1]:raise ValueError('layout differs between orders')
        bases=[];groups=[]
        for tile in range(6):
            b=key(['atbsr',tile,0,8,128,1,1024]);g=key(['atbsr_grouped',tile,0,8,128,1,1024]);bases.append(b);groups.append(g)
            if layouts[0][b][:5]!=layouts[0][g][:5]:raise ValueError('shape/balance changed')
            shapes.append({'dataset':name,'tile_index':tile,'shape':f'{1<<tile}x{32>>tile}',
                           'grouped_active':layouts[0][g][5],'column_bytes':layouts[0][g][7],
                           'original_us':gm(t[b] for t in orders),'grouped_us':gm(t[g] for t in orders),
                           'speedup':gm(t[b]/t[g] for t in orders),
                           'order0_speedup':orders[0][b]/orders[0][g], 'order1_speedup':orders[1][b]/orders[1][g]})
        b='atbsr_control:auto';g=key(['atbsr_grouped_model',-1,-1,8,128,1,1024])
        if layouts[0][b][:5]!=layouts[0][g][:5]:raise ValueError('automatic shape changed')
        matrices.append({'dataset':name,'original_shape':f'{1<<layouts[0][b][3]}x{32>>layouts[0][b][3]}',
                         'balanced':layouts[0][b][4],'grouped_active':layouts[0][g][5],
                         'original_auto_us':gm(t[b] for t in orders),'grouped_auto_us':gm(t[g] for t in orders),
                         'auto_speedup':gm(t[b]/t[g] for t in orders),
                         'best_grouped6_vs_original6':gm(min(t[k] for k in bases)/min(t[k] for k in groups) for t in orders),
                         'auto_vs_best_cusparse':gm(min(t[k] for k in ['cusparse:CSR_ALG1','cusparse_default:cusparse_default','cusparse_alg2:cusparse_alg2'])/t[g] for t in orders)})
    for filename,rows in [('shapes.csv',shapes),('matrices.csv',matrices)]:
        if rows:
            with (p/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={f'{1<<tile}x{32>>tile}':aggregate([r['speedup'] for r in shapes if r['tile_index']==tile and r['grouped_active']]) for tile in range(6)}
    result={'coverage':coverage,'same_shape_active':stats,
            'automatic_all':aggregate([r['auto_speedup'] for r in matrices]),
            'automatic_active':aggregate([r['auto_speedup'] for r in matrices if r['grouped_active']]),
            'automatic_vs_best_cusparse':aggregate([r['auto_vs_best_cusparse'] for r in matrices]),
            'ordinary_six_shape_oracle':aggregate([r['best_grouped6_vs_original6'] for r in matrices]),
            'note':'All ratios use paired process medians then geomean. Oracle is post-hoc best among six ordinary shapes only. Fallbacks excluded from per-shape active statistics, retained in automatic-all. No initialization cost included.'}
    write_json(p/'shape-summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
