#!/usr/bin/env python3
"""Compare fixed candidates' initialization timings with independent Graph timings.

The two phases use distinct buffers and possibly different allocation histories;
the ratio cannot by itself identify the cause of a discrepancy.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    meta = json.loads((root/'metadata.json').read_text())
    if meta.get('selection') != 'auto_tuned' or not meta.get('extended_baselines'):
        raise SystemExit('requires full Auto tuning and extended fixed baselines')
    # TUNING fields: backend, tile, balance, parts, threads, group, threshold, width, us.
    signatures = {
        ('atbsr', '3', '0'): 'atbsr_m8',
        ('atbsr', '0', '1'): 'atbsr_m1_balanced',
        ('csr_dsm', '8', '256', '1', '1024'): 'csr_dsm_fixed',
        ('csr_two_stage', '8', '256', '1', '1024'): 'csr_two_stage_fixed',
        ('cusparse',): 'cusparse',
    }
    ratios = collections.defaultdict(list)
    entries = []
    for status_path in sorted((root/'status').glob('*.json')):
        status = json.loads(status_path.read_text())
        if status.get('status') != 'ok':
            continue
        name = status['dataset']
        observed = collections.defaultdict(list)
        with (root/'raw'/f'{name}.csv').open() as f:
            for row in csv.DictReader(f):
                observed[(row['order'], row['backend'])].append(float(row['us']))
        order = None
        inside = False
        seen = set()
        for line in (root/'raw'/f'{name}.log').read_text().splitlines():
            if line.startswith('BENCH_ORDER,'):
                order = line.split(',', 1)[1]
            elif line.startswith('TUNING_BEGIN '):
                if order is None or inside:
                    raise ValueError(f'{name}: invalid tuning begin')
                inside = True
            elif line == 'TUNING_END':
                if not inside:
                    raise ValueError(f'{name}: invalid tuning end')
                inside = False
            elif inside and line.startswith('TUNING,'):
                fields = next(csv.reader([line]))[1:]
                backend = fields[0]
                if backend == 'atbsr':
                    signature = tuple(fields[:3])
                elif backend in ('csr_dsm', 'csr_two_stage'):
                    signature = (backend, *fields[3:7])
                else:
                    signature = (backend,)
                label = signatures.get(signature)
                if label is None or (order, label) not in observed:
                    continue
                key = (order, label)
                if key in seen:
                    raise ValueError(f'{name}: duplicate tuning candidate {key}')
                seen.add(key)
                tuned = float(fields[8])
                tested = statistics.median(observed[key])
                if not math.isfinite(tuned) or not math.isfinite(tested) or tuned <= 0 or tested <= 0:
                    raise ValueError(f'{name}: invalid timing')
                ratio = tested/tuned
                ratios[label].append(ratio)
                entries.append({'dataset': name, 'order': order, 'candidate': label,
                                'tuning_us': tuned, 'independent_us': tested, 'independent_over_tuning': ratio})
        if inside:
            raise ValueError(f'{name}: incomplete tuning block')
    report = {'records': len(entries), 'candidates': {},
              'method': 'Independent fixed candidate median divided by initialization median, per construction order; same named configuration, not necessarily same allocation or input/output buffers. Warm Graph; shared-device polling cannot exclude all interference. Sample is incomplete and was used in development.'}
    for label, values in sorted(ratios.items()):
        ordered = sorted(values)
        report['candidates'][label] = {'observations': len(values),
            'geomean_ratio': math.exp(statistics.mean(map(math.log, values))),
            'median_ratio': statistics.median(values),
            'p10_ratio': ordered[int((len(ordered)-1)*.1)],
            'p90_ratio': ordered[int((len(ordered)-1)*.9)],
            'above_1_10': sum(x > 1.1 for x in values),
            'below_0_90': sum(x < .9 for x in values)}
    (root/'tuning-drift.json').write_text(json.dumps(report, indent=2)+'\n')
    if entries:
        with (root/'tuning-drift.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
