#!/bin/bash
set -euo pipefail
cd /staff/chengjiajun/workspace/AtSpmv
while tmux has-session -t atspmv_graph_663 2>/dev/null || tmux has-session -t atspmv_graph_train_dev 2>/dev/null; do sleep 30; done
root=results/model/selection-model
python3 "$root/h100-vector-graph-train-dev/provenance/audit_vector_graph_inputs.py" \
    --reference "$root/h100-v2-full-2247" --run "$root/h100-vector-graph-663" \
    --run "$root/h100-vector-graph-train-dev" --output "$root/h100-vector-graph-train-dev/input-audit.json"
