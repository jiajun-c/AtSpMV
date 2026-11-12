#!/usr/bin/env python3
"""Verify that extra benchmarks only receive scanned, unchanged inputs."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temporary:
    d = Path(temporary)
    scan, matrices, out = d/'scan', d/'matrices', d/'out'
    (scan/'provenance').mkdir(parents=True)
    (scan/'status').mkdir()
    (matrices/'valid').mkdir(parents=True)
    matrix = matrices/'valid/valid.mtx'
    matrix.write_text('test')
    with (scan/'provenance/input-status.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'status'])
        writer.writeheader(); writer.writerows([{'dataset':'valid', 'status':'unverified'}, {'dataset':'bad', 'status':'unverified'}])
    stat = matrix.stat()
    features = dict(dataset='valid', rows='1', columns='1', nnz='1000001', longest_row='1',
                    longest_row_nnz='1000001', ratio='1', status='ok')
    (scan/'status/valid.json').write_text(json.dumps({'status':'ok', 'structure':features,
        'file_bytes':stat.st_size, 'file_mtime_ns':stat.st_mtime_ns}))
    (scan/'status/bad.json').write_text(json.dumps({'status':'invalid_data'}))
    cmd = ['python3', str(root/'scripts/analysis/finalize_unindexed_selection.py'), '--scan', str(scan),
           '--matrix-root', str(matrices), '--output', str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    with (out/'index.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['dataset'] == 'valid'
    matrix.write_text('changed')
    assert subprocess.run(cmd, capture_output=True).returncode != 0
print('unindexed selection finalization passed')
