#!/bin/bash
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/all-2888-ada
if [ ! -f "$out/metadata.json" ]; then echo 'Waiting for runner initialization'; exit 0; fi
python3 "$out/provenance/analyze_mixed_tile.py" "$out" > "$out/live-summary.log" || exit 1
python3 - "$out" <<'PY'
import csv,json,sys,statistics,math
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads((p/'analysis.json').read_text()); print('status:',x['status_counts'],'/ total:',x['manifest_total'])
if not (p/'analysis.csv').exists():raise SystemExit
r=list(csv.DictReader((p/'analysis.csv').open()));
if not r:raise SystemExit
mixed=[int(v['nnz'])/int(v['mixed_slots']) for v in r]
fixed=[d*float(v['mixed_slots_vs_fewest_fixed']) for d,v in zip(mixed,r)]
gains=[float(v['best_mixed_vs_fastest_fixed']) for v in r]
print('successful:',len(r),'mixed / fastest AtBSR geomean:',round(math.exp(statistics.mean(map(math.log,gains))),4),'faster:',sum(v>1 for v in gains))
print('per-matrix mean density: mixed {:.2%}, best fixed {:.2%}'.format(statistics.mean(mixed),statistics.mean(fixed)))
print('Note: partial results favor small matrices; fastest candidates selected retrospectively.')
PY
printf '\nRecent progress:\n'
tail -5 "$out/runner.log"
