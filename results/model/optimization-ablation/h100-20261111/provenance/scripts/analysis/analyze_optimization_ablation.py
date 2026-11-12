#!/usr/bin/env python3
"""Reduce explicit ablation pairs; retain regressions and inactive fallbacks."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from benchmark_dsm_atbsr_full import write_json
from benchmark_optimization_ablation import analyze


def key(backend, tile=-1, balanced=-1, parts=8, threads=128, group=1, threshold=1024):
    return backend+':'+('/'.join(map(str, (tile,balanced,parts,threads,group,threshold))))


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def reduce_main(output):
    manifest=json.loads((output/'manifest.json').read_text())
    summary=analyze(output,manifest)
    timings=list(csv.DictReader((output/'timings.csv').open()))
    rows=[]
    for item in manifest['matrices']:
        name=item['dataset']; measurements={r['candidate']:r for r in timings if r['dataset']==name}
        if not measurements: continue
        per_order=[{k:float(v[f'order{o}_us']) for k,v in measurements.items()} for o in (0,1)]
        layouts=[]
        for order in (0,1):
            layout={}
            for line in (output/'raw'/f'{name}-{order}.log').read_text().splitlines():
                if line.startswith('PLAN_LAYOUT,'):
                    fields=line.split(',');layout[fields[1]]=[int(x) for x in fields[2:]]
            layouts.append(layout)
        for label in ['atbsr_control','atbsr_grouped','atbsr_compressed','atbsr_grouped_model']:
            if layouts[0][label]!=layouts[1][label]: raise ValueError('layout mismatch')
        ordinary=[key('atbsr',t,0) for t in range(6)]
        fixed=[key('atbsr',t,b) for t in range(6) for b in (0,1)]
        base32=key('atbsr',5,0);bal32=key('atbsr',5,1)
        row={'dataset':name,'grouped_active':layouts[0]['atbsr_grouped'][5],
             'grouped_model_active':layouts[0]['atbsr_grouped_model'][5]}
        metrics={
            'shape_oracle_vs_32':lambda t:t[base32]/min(t[k] for k in ordinary),
            'balance_oracle_vs_ordinary':lambda t:min(t[k] for k in ordinary)/min(t[k] for k in fixed),
            'balance_at_32':lambda t:t[base32]/t[bal32],
            'prefetch_12config_gmean':lambda t:geomean(t[key('atbsr_historical',i,b)]/t[key('atbsr',i,b)] for i in range(6) for b in (0,1)),
            'remove_clear_32':lambda t:t[key('atbsr_clear',5,0)]/t[base32],
            'cta_merge_32_balanced':lambda t:t[bal32]/t[key('atbsr_tuning',5,1,8,4,2,1)],
            'cap4_32_balanced':lambda t:t[bal32]/t[key('atbsr_tuning',5,1,4,4,2,0)],
            'cap16_32_balanced':lambda t:t[bal32]/t[key('atbsr_tuning',5,1,16,4,2,0)],
            'warps2_32':lambda t:t[base32]/t[key('atbsr_tuning',5,0,8,2,2,0)],
            'warps8_32':lambda t:t[base32]/t[key('atbsr_tuning',5,0,8,8,2,0)],
            'depth4_vs2_32':lambda t:t[base32]/t[key('atbsr_tuning',5,0,8,4,4,0)],
            'mixed_split8_vs_unsplit':lambda t:t[key('mixed_tile',parts=0)]/t[key('mixed_tile',parts=8)],
            'mixed_oracle_vs_fixed12':lambda t:min(t[k] for k in fixed)/min(t[key('mixed_tile',parts=c)] for c in (0,8)),
            'vector16_vs_auto_atbsr':lambda t:t['atbsr_control:auto']/t[key('csr_vector',parts=16)],
        }
        if row['grouped_active']:
            metrics.update(grouped_vs_32=lambda t:t[base32]/t[key('atbsr_grouped',5,0)],
                           grouped_vs_old_compression=lambda t:t[key('atbsr_compressed',5,0)]/t[key('atbsr_grouped',5,0)],
                           old_compression_vs_32=lambda t:t[base32]/t[key('atbsr_compressed',5,0)])
        if any(c[0]=='atbsr_i32' for c in item['configs']):
            metrics['int16_vs_int32_32']=lambda t:t[key('atbsr_i32',5,0)]/t[base32]
        if key('csr_dsm') in measurements:
            metrics['dsm_vs_two_stage']=lambda t:t[key('csr_two_stage')]/t[key('csr_dsm')]
            metrics['hybrid_vs_auto_atbsr']=lambda t:t['atbsr_control:auto']/t[key('hybrid_dsm_atbsr')]
        for metric,fn in metrics.items(): row[metric]=geomean(fn(t) for t in per_order)
        rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'factors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    factors={}
    for metric in fields[3:]:
        values=[r[metric] for r in rows if metric in r]
        factors[metric]={'n':len(values),'geomean':geomean(values),'wins_over_1pct':sum(v>1.01 for v in values),'losses_over_1pct':sum(v<1/1.01 for v in values)}
    write_json(output/'factor-summary.json',{'coverage':summary,'factors':factors})


def reduce_rows(output):
    manifest=json.loads((output/'manifest.json').read_text());analyze(output,manifest)
    timings=list(csv.DictReader((output/'timings.csv').open()));rows=[]
    for item in manifest['matrices']:
        name=item['dataset'];ts={r['candidate']:r for r in timings if r['dataset']==name}
        if not ts: continue
        slots=[]
        for o in (0,1):
            order={}
            for line in (output/'raw'/f'{name}-{o}.log').read_text().splitlines():
                if line.startswith('PLAN_SLOTS,'):
                    _,b,t,s=line.split(',');order[b+':'+t]=int(s)
            slots.append(order)
        if slots[0]!=slots[1]: raise ValueError('packing differs across orders')
        for tile in (3,5):
            original=ts[key('atbsr_rows_original',tile,0)];sorted_=ts[key('atbsr_rows_sorted',tile,0)]
            a=slots[0]['atbsr_rows_original:'+str(tile)];b=slots[0]['atbsr_rows_sorted:'+str(tile)]
            rows.append({'dataset':name,'shape':f'{1<<tile}x{32>>tile}', 'original_slots':a,'sorted_slots':b,
                         'density_gain':a/b,'sort_speedup':geomean(float(original[f'order{o}_us'])/float(sorted_[f'order{o}_us']) for o in (0,1))})
    with (output/'row-order-factors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--row-order',action='store_true');a=p.parse_args()
    (reduce_rows if a.row_order else reduce_main)(a.output)


if __name__=='__main__':main()
