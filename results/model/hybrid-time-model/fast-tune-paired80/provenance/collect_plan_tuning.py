#!/usr/bin/env python3
"""Export logged initialization timings; these are tuning, not test timings."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.directory
metadata=json.loads((root/'metadata.json').read_text())
expected_blocks=2 if metadata.get('counterbalance') else 1
rows=[];errors=[];cases=0
for status_file in sorted((root/'status').glob('*.json')):
    status=json.loads(status_file.read_text())
    if status['status']!='ok':continue
    name=status['dataset'];blocks=[];current=None;last_plan=None;block_plan=None
    for line in (root/'raw'/f'{name}.log').read_text().splitlines():
        if line.startswith('PLAN_PREP,'):
            last_plan=line.split(',')[1]
        elif line.startswith('TUNING_BEGIN '):
            if current is not None:errors.append(f'{name}: nested block')
            current=[];block_plan=last_plan or metadata.get('selection','auto_tuned')
        elif line=='TUNING_END':
            if current is None:errors.append(f'{name}: end without block')
            else:
                if block_plan==metadata.get('selection','auto_tuned'):blocks.append(current)
                current=None;last_plan=None;block_plan=None
        elif line.startswith('TUNING,'):
            fields=next(csv.reader([line]))
            if current is None or len(fields)!=10:errors.append(f'{name}: malformed candidate');continue
            _,backend,tile,balanced,parts,threads,group,threshold,width,us=fields
            try:
                row={'dataset':name,'block':len(blocks),'backend':backend,'tile_index':int(tile),'balanced':int(balanced),
                     'parts':int(parts),'threads':int(threads),'rows_per_group':int(group),'threshold':int(threshold),
                     'vector_width':int(width),'median_us':float(us)}
                if not 0<row['median_us']<float('inf'):raise ValueError('non-positive timing')
                current.append(row)
            except ValueError:errors.append(f'{name}: invalid candidate')
    if current is not None or len(blocks)!=expected_blocks or any(not b for b in blocks):
        errors.append(f'{name}: expected {expected_blocks} complete nonempty blocks, found {len(blocks)}')
        continue
    rows.extend(r for block in blocks for r in block);cases+=1
if errors:raise SystemExit('\n'.join(errors))
if rows:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print(json.dumps({'cases':cases,'candidate_rows':len(rows),'blocks_per_case':expected_blocks,
                  'note':'Initialization measurements; selected-candidate bias and device interference apply. Not independent test timing.'}))
