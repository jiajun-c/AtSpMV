#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/plan-setup-pilot-h100
while tmux has-session -t atsptrsv_plansetup 2>/dev/null; do sleep 15; done
python3 "$out/provenance/analyze_sptrsv_setup.py" "$out" >> "$out/runner.log" 2>&1
