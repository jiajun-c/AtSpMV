#!/usr/bin/env python3
"""Exercise profile task persistence without requiring a GPU or Nsight."""
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    fake_ncu = d/'fake-ncu'
    fake_ncu.write_text('#!/bin/sh\nprintf "sm__inst_executed_pipe_tensor.sum\\n"\n')
    fake_ncu.chmod(0o755)
    exe = d/'benchmark'
    exe.write_text('frozen binary')
    out = d/'output'
    cmd = ['python3', str(root/'scripts/analysis/profile_cusparse_spmm_cores.py'),
           '--exe', str(exe), '--out', str(out), '--ncu', str(fake_ncu)]
    for _ in range(2):
        subprocess.run(cmd, check=True, capture_output=True)
    assert len(list(out.glob('*.csv'))) == 12
    records = [json.loads(p.read_text()) for p in out.glob('b*.json')]
    assert len(records) == 12 and all(r['status'] == 'ok' and r['has_metric_records'] for r in records)
    exact_out = d/'exact-output'
    exact_cmd = ['python3', str(root/'scripts/analysis/profile_cusparse_spmm_cores.py'),
                 '--exe', str(exe), '--out', str(exact_out), '--ncu', str(fake_ncu), '--exact-one']
    subprocess.run(exact_cmd, check=True, capture_output=True)
    assert len(list(exact_out.glob('*.csv'))) == 15
    assert (exact_out/'b64-d0-n32-blocked_ell.json').exists()
    exe.write_text('changed binary')
    assert subprocess.run(cmd, capture_output=True).returncode != 0
print('instruction profile resumption and provenance passed')
