#!/usr/bin/env python3
"""Plot completed per-matrix outcomes; the input CSV contains successes only."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
    with (root/'analysis.csv').open() as f:rows=list(csv.DictReader(f))
    if not rows:raise RuntimeError('no results')
    n=np.array([int(r['nnz']) for r in rows]);ratio=np.array([float(r['speedup_vs_best_nonhybrid']) for r in rows]);maximum=np.array([int(r['max_row_nnz']) for r in rows]);valid=np.isfinite(ratio)&(ratio>0)&(n>0)
    fig,(scatter,hist)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[2,1]})
    for mask,color,label in [(maximum<128,'#7a91a1','longest row <128'),((maximum>=128)&(maximum<1024),'#de9f35','128–1023'),(maximum>=1024,'#227a86','≥1024')]:
        use=mask&valid;scatter.scatter(n[use],ratio[use],s=9,alpha=.5,c=color,label=f'{label} (n={int(use.sum())})',rasterized=True)
    scatter.axhline(1,color='#333333',ls='--',lw=1);scatter.set_xscale('log');scatter.set_yscale('log');scatter.set_xlabel('Matrix nonzeros');scatter.set_ylabel('Best non-hybrid time / hybrid time');scatter.legend(fontsize=8);scatter.grid(alpha=.2)
    vals=np.log2(ratio[valid]);hist.hist(vals,bins=np.linspace(min(-4,float(vals.min())),max(2,float(vals.max())),31),color='#507c9e');hist.axvline(0,color='#333333',ls='--',lw=1);hist.set_xlabel('log₂ speedup');hist.set_ylabel('Number of matrices');hist.grid(axis='y',alpha=.2)
    fig.suptitle(f'H100: fixed-candidate warm-graph sweep (n={len(rows)} successful matrices)');fig.tight_layout();fig.savefig(root/'speedups.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
