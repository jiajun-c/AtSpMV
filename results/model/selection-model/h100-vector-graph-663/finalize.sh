#!/bin/bash
set -euo pipefail
cd /staff/chengjiajun/workspace/AtSpmv
out=results/model/selection-model/h100-vector-graph-663
while tmux has-session -t atspmv_graph_663 2>/dev/null; do sleep 30; done
if ! grep -q '^runner exit code: 0$' "$out/runner.log" 2>/dev/null; then
    printf 'Graph comparison did not complete successfully; inspect runner.log.\n' >&2
    exit 1
fi
python3 "$out/provenance/analyze_vector_graph_comparison.py" "$out"
