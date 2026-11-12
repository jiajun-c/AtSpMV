#!/usr/bin/env python3
"""Compare frozen and Graph-trained Plans on the same H100 runs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    a = parser.parse_args()
    manifest = json.loads((a.run/'manifest.json').read_text())
    metadata = json.loads((a.run/'metadata.json').read_text())
    rows, states = [], Counter()
    for item in manifest:
        name = item['dataset']
        path = a.run/'status'/f'{name}.json'
        if not path.exists():
            states['pending'] += 1
            continue
        status = json.loads(path.read_text())['status']
        states[status] += 1
        if status != 'ok':
            continue
        samples = defaultdict(list)
        with (a.run/'raw'/f'{name}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                v = float(row['us'])
                if not math.isfinite(v) or v <= 0: raise ValueError(f'invalid timing: {name}')
                samples[row['order'],row['backend']].append(v)
        orders = sorted({key[0] for key in samples})
        required = ('atbsr','cusparse','auto_vector_model_v2','auto_vector_model_graph',
                    *(f'csr_vec{w}' for w in (1,2,4,8,16,32)))
        if len(orders) != 2 or any(len(samples[o,label]) != metadata['trials'] for o in orders for label in required):
            raise ValueError(f'incomplete timing: {name}')
        times = {key:statistics.median(v) for key,v in samples.items()}
        predictions = {}
        order = None
        with (a.run/'raw'/f'{name}.log').open(newline='') as f:
            for fields in csv.reader(f):
                if not fields: continue
                if fields[0] == 'BENCH_ORDER': order = fields[1]
                elif len(fields) >= 12 and fields[0] == 'MODEL_PREDICTION' and fields[2] == 'csr_vector' and fields[1] in ('auto_vector_model_v2','auto_vector_model_graph'):
                    key = order,fields[1],int(fields[9])
                    if order is None or key in predictions: raise ValueError(f'duplicate prediction: {name}')
                    predictions[key] = float(fields[10])
        errors = {}
        for label in ('auto_vector_model_v2','auto_vector_model_graph'):
            values = []
            for o in orders:
                for w in (1,2,4,8,16,32):
                    pred = predictions[o,label,w]
                    if not math.isfinite(pred) or pred <= 0: raise ValueError(f'invalid prediction: {name}')
                    values.append(abs(pred/(times[o,f'csr_vec{w}']/times[o,'atbsr'])-1))
            errors[label] = statistics.mean(values)
        rows.append({'dataset':name,
                     'graph_speedup_vs_v2':geo(times[o,'auto_vector_model_v2']/times[o,'auto_vector_model_graph'] for o in orders),
                     'graph_speedup_vs_cusparse':geo(times[o,'cusparse']/times[o,'auto_vector_model_graph'] for o in orders),
                     'v2_width_ratio_mape':errors['auto_vector_model_v2'],
                     'graph_width_ratio_mape':errors['auto_vector_model_graph']})
    result = {'status_counts':dict(states), 'total':len(manifest), 'cases':len(rows),
              'method':'Same-run warm Graph timings in two construction orders; each width uses separate allocations. '
                       'Input matrices were previously inspected and may share families with training. '
                       'Graph model is experimental and setup is excluded.'}
    if rows:
        result.update({'graph_speedup_vs_v2':geo(r['graph_speedup_vs_v2'] for r in rows),
                       'graph_speedup_vs_cusparse':geo(r['graph_speedup_vs_cusparse'] for r in rows),
                       'v2_width_ratio_mape':statistics.mean(r['v2_width_ratio_mape'] for r in rows),
                       'graph_width_ratio_mape':statistics.mean(r['graph_width_ratio_mape'] for r in rows),
                       'graph_faster_5pct':sum(r['graph_speedup_vs_v2'] > 1.05 for r in rows),
                       'graph_slower_5pct':sum(r['graph_speedup_vs_v2'] < 1/1.05 for r in rows)})
        with (a.run/'graph-comparison.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.run/'graph-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
