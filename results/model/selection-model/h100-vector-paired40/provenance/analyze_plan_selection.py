#!/usr/bin/env python3
"""Summarize a benchmark_plan_selection run, retaining failures separately."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    root = a.directory
    manifest = json.loads((root/'manifest.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    selection = metadata.get('selection','auto_tuned')
    extended = metadata.get('extended_baselines',False)
    vectors = metadata.get('vector_candidates',False)
    compare_full = metadata.get('compare_full_tune',False)
    compare_model = metadata.get('compare_auto_model',False)
    compare_vector_model = metadata.get('compare_vector_model',False)
    compare_ungated = metadata.get('compare_ungated_vector',False)
    rows = []
    counts = collections.Counter()
    selected = collections.Counter()
    for item in manifest:
        name = item['dataset']
        status = root/'status'/f'{name}.json'
        if not status.exists():
            counts['pending'] += 1
            continue
        data = json.loads(status.read_text())
        counts[data['status']] += 1
        if data['status'] != 'ok':
            continue
        values = collections.defaultdict(list)
        configs = {}
        with (root/'raw'/f'{name}.csv').open() as f:
            for record in csv.DictReader(f):
                order=record.get('order','baseline_first')
                values[(order,record['backend'])].append(float(record['us']))
                configs[(order,record['backend'])] = record['config']
        orders=sorted({k[0] for k in values})
        times = {k: statistics.median(v) for k, v in values.items()}
        choices = [configs[(order,selection)].split('/')[0] for order in orders]
        choice = choices[0] if len(set(choices))==1 else '|'.join(choices)
        selected[choice] += 1
        geo=lambda seq:math.exp(statistics.mean(math.log(x) for x in seq))
        row={'dataset': name, 'nnz': item['nnz'], 'selected': choice, 'selected_config': '|'.join(configs[(o,selection)] for o in orders),
                     'auto_us': geo(times[(o,selection)] for o in orders), 'default_us': geo(times[(o,'atbsr')] for o in orders), 'cusparse_us': geo(times[(o,'cusparse')] for o in orders),
                     'speedup_vs_default': geo(times[(o,'atbsr')]/times[(o,selection)] for o in orders),
                     'speedup_vs_cusparse': geo(times[(o,'cusparse')]/times[(o,selection)] for o in orders),
                     'speedup_vs_best_baseline': geo(min(times[(o,'atbsr')], times[(o,'cusparse')])/times[(o,selection)] for o in orders)}
        if extended:
            baselines=('atbsr','atbsr_m8','atbsr_m1_balanced','csr_dsm_fixed','csr_two_stage_fixed','cusparse')
            row['speedup_vs_extended_baseline']=geo(min(times[(o,b)] for b in baselines)/times[(o,selection)] for o in orders)
            if vectors:
                vector_names=[f'csr_vec{x}' for x in (1,2,4,8,16,32)]
                row['speedup_best_vector_vs_extended']=geo(min(times[(o,b)] for b in baselines)/min(times[(o,v)] for v in vector_names) for o in orders)
                row['speedup_oracle_extended_plus_vector']=geo(max(1.,min(times[(o,b)] for b in baselines)/min(times[(o,v)] for v in vector_names)) for o in orders)
        if compare_full:
            row['speedup_vs_full_tune']=geo(times[(o,'auto_tuned')]/times[(o,selection)] for o in orders)
        if compare_model:
            row['speedup_vs_auto_model']=geo(times[(o,'auto_model')]/times[(o,selection)] for o in orders)
        if compare_vector_model:
            row['speedup_vector_vs_auto_model']=geo(times[(o,'auto_model')]/times[(o,'auto_vector_model')] for o in orders)
            row['speedup_vs_vector_model']=geo(times[(o,'auto_vector_model')]/times[(o,selection)] for o in orders)
        if compare_ungated:
            row['speedup_vs_ungated_vector']=geo(times[(o,'auto_vector_model')]/times[(o,selection)] for o in orders)
            if compare_model:
                row['speedup_ungated_vs_auto_model']=geo(times[(o,'auto_model')]/times[(o,'auto_vector_model')] for o in orders)
        rows.append(row)
    result = {'manifest_total': len(manifest), 'status_counts': dict(counts), 'selected_backends': dict(selected),
              'selection': selection, 'method': 'Actual Plan selection, independently timed against default AtBSR and cuSPARSE; warm Graph, excludes setup. Ratios across construction orders use the geometric mean. Randomized manifest prefix is not the complete suite.'}
    if metadata.get('warm_setup_anchor',False):
        result['method'] += ' Default AtBSR is constructed first in both orders; the two model plans swap relative order after this anchor.'
    if metadata.get('alias_identical_models',False):
        result['method'] += ' Model labels with identical resolved configurations execute the same Plan instance.'
    if metadata.get('alias_identical_graphs',False):
        result['method'] += ' Those labels also execute the same Graph executable.'
    if metadata.get('internal_graph_events',False):
        result['method'] += ' CUDA event nodes inside each Graph measure the GPU interval; this includes event-node overhead.'
    if rows:
        for key in ('speedup_vs_default', 'speedup_vs_cusparse', 'speedup_vs_best_baseline')+(('speedup_vs_extended_baseline',) if extended else ())+(('speedup_best_vector_vs_extended','speedup_oracle_extended_plus_vector') if vectors else ())+(('speedup_vs_full_tune',) if compare_full else ())+(('speedup_vs_auto_model',) if compare_model else ())+(('speedup_vector_vs_auto_model','speedup_vs_vector_model') if compare_vector_model else ())+(('speedup_vs_ungated_vector',) if compare_ungated else ())+(('speedup_ungated_vs_auto_model',) if compare_model and compare_ungated else ()):
            v = [r[key] for r in rows]
            result[key] = {'geomean': math.exp(statistics.mean(math.log(x) for x in v)), 'median': statistics.median(v), 'wins': sum(x > 1 for x in v), 'cases': len(v)}
        with (root/'analysis.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (root/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
