#!/usr/bin/env python3
"""Compare the model's measured time to the best measured Plan in each order."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

LABELS = ('auto_model', 'auto_tuned', 'auto_tuned_mixed', 'atbsr', 'cusparse')


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    manifest = json.loads((root / 'manifest.json').read_text())
    metadata = json.loads((root / 'metadata.json').read_text())
    counts, rows = Counter(), []
    for entry in manifest:
        name = entry['dataset']
        status_path = root / 'status' / f'{name}.json'
        if not status_path.exists():
            counts['pending'] += 1
            continue
        status = json.loads(status_path.read_text())['status']
        counts[status] += 1
        if status != 'ok':
            continue
        groups, configs = defaultdict(list), {}
        with (root / 'raw' / f'{name}.csv').open(newline='') as source:
            for row in csv.DictReader(source):
                key = row['order'], row['backend']
                if row['backend'] in LABELS:
                    groups[key].append(float(row['us']))
                    configs[key] = row['config']
        orders = sorted({order for order, _ in groups})
        expected = {(order, label) for order in orders for label in LABELS}
        if len(orders) != 2 or set(groups) != expected:
            raise ValueError(f'{name}: missing comparison candidate/order')
        if any(len(v) != metadata['trials'] or any(not math.isfinite(x) or x <= 0 for x in v)
               for v in groups.values()):
            raise ValueError(f'{name}: invalid measurements')
        times = {key: statistics.median(v) for key, v in groups.items()}
        model_over_best, model_over_tuned, model_over_mixed = [], [], []
        best_labels = []
        for order in orders:
            best = min(LABELS, key=lambda label: times[(order, label)])
            best_labels.append(best)
            model = times[(order, 'auto_model')]
            model_over_best.append(model / times[(order, best)])
            model_over_tuned.append(model / times[(order, 'auto_tuned')])
            model_over_mixed.append(model / times[(order, 'auto_tuned_mixed')])
        rows.append({'dataset': name, 'nnz': entry['nnz'],
                     'model_over_observed_best': geometric_mean(model_over_best),
                     'model_over_tuned': geometric_mean(model_over_tuned),
                     'model_over_tuned_mixed': geometric_mean(model_over_mixed),
                     'model_config': '|'.join(configs[(o, 'auto_model')] for o in orders),
                     'best_label': '|'.join(best_labels),
                     'tuned_mixed_config': '|'.join(configs[(o, 'auto_tuned_mixed')] for o in orders)})
    summary = {'manifest_total': len(manifest), 'status_counts': dict(counts),
               'scope': 'Independent paired Graph timings; best of five observed Plan results per order, not an exhaustive kernel oracle. Geomean across orders; setup excluded.'}
    if rows:
        values = [row['model_over_observed_best'] for row in rows]
        summary.update({'cases': len(rows), 'geomean_model_over_observed_best': geometric_mean(values),
                        'median_model_over_observed_best': statistics.median(values),
                        'model_within_5pct_cases': sum(v <= 1.05 for v in values),
                        'model_more_than_10pct_slower_cases': sum(v > 1.10 for v in values),
                        'max_model_over_observed_best': max(values)})
        with (root / 'oracle-analysis.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (root / 'oracle-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
