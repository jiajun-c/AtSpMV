#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-vector-graph-all
while true; do
    printf '\n%s\n' "$(date -Is)" >> "$out/join.log"
    python3 "$out/provenance/join_graph_dataset_results.py" >> "$out/join.log" 2>&1 || printf 'join failed; will retry if jobs remain\n' >> "$out/join.log"
    if ! tmux has-session -t atspmv_graph_train_dev 2>/dev/null && \
       ! tmux has-session -t atspmv_graph_input_audit 2>/dev/null && \
       ! tmux has-session -t atspmv_final_summary 2>/dev/null && \
       ! tmux has-session -t atspmv_summary 2>/dev/null; then
        python3 "$out/provenance/join_graph_dataset_results.py" >> "$out/join.log" 2>&1
        exit $?
    fi
    sleep 300
done
