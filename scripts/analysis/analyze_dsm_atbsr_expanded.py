#!/usr/bin/env python3
"""Summarize a completed stratified hybrid validation, including online selection."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args();root=args.directory
    manifest=json.loads((root/'manifest.json').read_text());metadata=json.loads((root/'final/metadata.json').read_text())
    summary=json.loads((root/'final/summary.json').read_text());byname={x['name']:x for x in manifest['manifest']};out=[]
    seeds=metadata['seeds']
    for result in summary:
        name=result['dataset'];entry=byname[name];screen=root/'screen'/f'{name}.csv'
        with screen.open() as f:
            headline=f.readline();screenrows=list(csv.DictReader(f))
        best=min(screenrows,key=lambda r:float(r['median_us']))
        config='/'.join(best[x] for x in ('tile_index','balanced','parts','threads','rows_per_group','threshold'))
        seedmed=[]
        for seed in seeds:
            matches=[]
            for row in csv.DictReader((root/'final'/f'{name}-{seed}.csv').open()):
                if row['backend']==best['backend'] and row['config']==config:matches.append(float(row['us']))
            if not matches:raise RuntimeError(f'missing selected candidate {name} {best["backend"]} {config}')
            seedmed.append(statistics.median(matches))
        chosen=statistics.median(seedmed)
        out.append({'dataset':name,'group':entry['group'],'rows':entry['rows'],'cols':entry['cols'],'nnz':entry['nnz'],'max_row':entry['max_row'],'max_row_fraction':entry['max_row_fraction'],
            'hybrid_us':result['hybrid']['median_us'],'hybrid_config':result['hybrid']['config'],
            'best_nonhybrid_us':result['best_nonhybrid_us'],'best_nonhybrid':result['best_nonhybrid'],
            'hybrid_speedup':result['speedup_vs_best_nonhybrid'],'atbsr_us':result['backends']['atbsr']['median_us'],
            'cusparse_us':result['backends']['cusparse']['median_us'],'auto_screen_backend':best['backend'],
            'auto_config':config,'auto_recheck_us':chosen,'auto_vs_best_nonhybrid':result['best_nonhybrid_us']/chosen})
    def stats(items):
        h=[r['hybrid_speedup'] for r in items];auto=[r['auto_vs_best_nonhybrid'] for r in items]
        return {'count':len(items),'hybrid_wins':sum(x>1 for x in h),'hybrid_wins_5pct':sum(x>=1.05 for x in h),
                'hybrid_slower_5pct':sum(x<1/1.05 for x in h),'hybrid_geo_mean':math.exp(statistics.mean(math.log(x) for x in h)),
                'hybrid_median':statistics.median(h),'hybrid_min':min(h),'hybrid_max':max(h),
                'auto_hybrid_selected':sum(r['auto_screen_backend']=='hybrid_dsm_atbsr' for r in items),
                'auto_geo_mean_vs_best_nonhybrid':math.exp(statistics.mean(math.log(x) for x in auto))}
    grouping={group:stats([r for r in out if r['group']==group]) for group in ('high','medium','low','very_low','no_long') if any(r['group']==group for r in out)}
    info={'overall':stats(out),'groups':grouping,'note':'Stratified exploratory sample, not representative of the full database; baseline is the best of three shortlisted configurations per non-hybrid backend and one cuSPARSE algorithm; no inference for unmeasured configurations.'}
    (root/'final'/'analysis.json').write_text(json.dumps(info,indent=2)+'\n')
    with (root/'final'/'analysis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    ordered=sorted(out,key=lambda r:r['hybrid_speedup']); colors={'high':'#dc6446','medium':'#d5a028','low':'#2a8f88','very_low':'#5375b8','no_long':'#9098a0'}
    fig,ax=plt.subplots(figsize=(10,11));ax.barh([r['dataset'] for r in ordered],[r['hybrid_speedup'] for r in ordered],color=[colors[r['group']] for r in ordered]);ax.axvline(1,color='#303030',lw=1,ls='--')
    ax.set_xlabel('Speedup vs best measured non-hybrid (higher is better)');ax.set_title('Hybrid DSM + AtBSR: stratified SuiteSparse sample, H100 PCIe');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=g.replace('_',' ')) for g,c in colors.items()],title='Longest row fraction',loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(root/'final'/'speedups.png',dpi=180);plt.close(fig)
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
