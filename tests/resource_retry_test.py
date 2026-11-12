#!/usr/bin/env python3
"""Retry only explicit resource failures; preserve other failures."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d=Path(temporary);run=d/'source';(run/'status').mkdir(parents=True);(run/'raw').mkdir()
    names=['memory','timeout','cusparse','incorrect','valid']
    manifest=[dict(dataset=n,rows='1',columns='1',nnz='1',longest_row='1',longest_row_nnz='1',ratio='1',status='ok') for n in names]
    (run/'manifest.json').write_text(json.dumps(manifest))
    for name,state in [('memory','error'),('timeout','timeout'),('cusparse','error'),('incorrect','error'),('valid','ok')]:
        (run/'status'/f'{name}.json').write_text(json.dumps({'status':state}))
    (run/'raw/memory.log').write_text('out of memory\n')
    (run/'raw/timeout.log').write_text('BENCH_ORDER,baseline_first\n')
    (run/'raw/cusparse.log').write_text('cuSPARSE status 2\n')
    (run/'raw/incorrect.log').write_text('old allocation: out of memory\nincorrect result\n')
    subprocess.run(['python3',str(root/'scripts/analysis/prepare_resource_retry.py'),'--run',str(run),'--output',str(d/'out')],check=True,capture_output=True)
    with (d/'out/index.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    assert {r['dataset'] for r in rows}=={'memory','timeout','cusparse'}
    assert [r['reason'] for r in json.loads((d/'out/retry-evidence.json').read_text())['retries']]==['gpu_memory_error','timeout','gpu_memory_error']
print('resource retry classification passed')
