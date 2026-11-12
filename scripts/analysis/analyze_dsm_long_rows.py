#!/usr/bin/env python3
"""Summarize measured DSM long-row wins, losses, and matched controls."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in args.directories:
        meta = json.loads((directory / 'metadata.json').read_text())
        real_names = {Path(p).stem for p in meta['inputs']}
        for case in json.loads((directory / 'summary.json').read_text()):
            name = case['case']
            dsm = case['best']['dsm_async']
            samples = []
            for seed in meta['seeds']:
                samples.extend(csv.DictReader((directory / f'{name}-{seed}.csv').open()))
            def matched(kind):
                rows = [r for r in samples if r['kind'] == kind and all(r[k] == dsm[k] for k in ['threads', 'parts', 'rows_per_group', 'threshold'])]
                return statistics.median(float(r['median_us']) for r in rows) if rows else None
            baseline = case['best'][case['best_non_dsm_kind']]
            ratios = list(case['fixed_dsm_vs_best_non_dsm_each_seed'].values())
            records.append({
                'case': name, 'real_matrix': name in real_names,
                'rows': int(samples[0]['m']), 'cols': int(samples[0]['n']), 'nnz': int(samples[0]['nnz']),
                'dsm_us': dsm['median_us'], 'best_non_dsm_us': baseline['median_us'],
                'best_non_dsm_kind': baseline['kind'], 'speedup': case['speedup'],
                'repeat_min_speedup': min(ratios), 'repeat_max_speedup': max(ratios),
                'all_repeats_above_1_05': min(ratios) > 1.05,
                'matched_pull_us': matched('dsm_pull'), 'matched_atomic_us': matched('atomic'),
                'matched_two_stage_us': matched('two_stage'), 'config': dsm,
                'source': str(directory),
            })
    (args.output / 'analysis.json').write_text(json.dumps(records, indent=2) + '\n')
    columns = [k for k in records[0] if k != 'config']
    with (args.output / 'analysis.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, max(4, len(records)*.31)))
    y = np.arange(len(records))
    values = [r['speedup'] for r in records]
    ax.barh(y, values, color=['#2878a5' if r['real_matrix'] else '#cf8a36' for r in records])
    # The dots/spans show individual-repeat range, not a confidence interval.
    for i, r in enumerate(records):
        ax.plot([r['repeat_min_speedup'], r['repeat_max_speedup']], [i, i], color='black', linewidth=1.5)
        ax.text(max(r['speedup'], r['repeat_max_speedup'])+.015, i, f"{r['speedup']:.2f}x", va='center', fontsize=8)
    ax.set_yticks(y, [r['case'] for r in records]); ax.invert_yaxis()
    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Best measured non-DSM time / async DSM time (higher is better)')
    ax.set_title('H100 PCIe FP64 single SpMV, warm CUDA Graph\nOrange: synthetic; blue: real matrix; black span: three-repeat range')
    ax.set_xlim(0, max(max(values), max(r['repeat_max_speedup'] for r in records))*1.13)
    ax.grid(axis='x', alpha=.2); fig.tight_layout()
    fig.savefig(args.output/'speedups.png', dpi=180); fig.savefig(args.output/'speedups.svg'); plt.close(fig)
    print(f'{len(records)} cases; {sum(r["all_repeats_above_1_05"] for r in records)} exceed 1.05x in all three repeats')


if __name__ == '__main__':
    main()
