#!/usr/bin/env python3
"""Join the experimental Graph-model comparison to the full dataset inventory."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import tempfile


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/model/selection-model'))
    args = parser.parse_args()
    root = args.root
    inventory = read_csv(root/'all-datasets-summary/all-datasets.csv')
    names = [row['dataset'] for row in inventory]
    if not inventory or len(set(names)) != len(names):
        raise ValueError('empty inventory or duplicate dataset')

    graph = {}
    metrics = ('graph_speedup_vs_v2', 'graph_speedup_vs_cusparse',
               'v2_width_ratio_mape', 'graph_width_ratio_mape')
    for group, directory in (('evaluation', 'h100-vector-graph-663'),
                             ('train_dev', 'h100-vector-graph-train-dev')):
        run = root/directory
        manifest = json.loads((run/'manifest.json').read_text())
        comparison = run/'graph-comparison.csv'
        results = read_csv(comparison) if comparison.exists() else []
        by_name = {row['dataset']: row for row in results}
        if len(by_name) != len(results):
            raise ValueError(f'duplicate comparison dataset: {run}')
        manifest_names = [item['dataset'] for item in manifest]
        if len(set(manifest_names)) != len(manifest_names) or set(by_name) - set(manifest_names):
            raise ValueError(f'invalid manifest or comparison: {run}')
        for name in manifest_names:
            if name in graph:
                raise ValueError(f'dataset appears in both groups: {name}')
            status_file = run/'status'/f'{name}.json'
            status = json.loads(status_file.read_text())['status'] if status_file.exists() else 'pending'
            result = by_name.get(name)
            if result and status != 'ok':
                raise ValueError(f'comparison without successful status: {name}')
            if status == 'ok' and not result:
                status = 'pending_analysis'
            graph[name] = {'graph_group': group, 'graph_status': status,
                           **{key: result[key] if result else '' for key in metrics}}

    scope = {row['dataset'] for row in inventory if row['scope'] == 'within_calibration_scope'}
    if set(graph) != scope:
        raise ValueError(f'Graph manifests differ from calibration scope: '
                         f'missing={len(scope - set(graph))}, extra={len(set(graph) - scope)}')
    counts = Counter()
    for row in inventory:
        row.update(graph.get(row['dataset'], {'graph_group': '', 'graph_status': 'not_measured',
                                              **dict.fromkeys(metrics, '')}))
        counts[row['graph_status']] += 1
    output = root/'h100-vector-graph-all'
    output.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents readers from seeing a partially written CSV.
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=output, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
        temporary = Path(stream.name)
    temporary.replace(output/'all-datasets-combined.csv')
    print(json.dumps({'total': len(inventory), 'graph_status_counts': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
