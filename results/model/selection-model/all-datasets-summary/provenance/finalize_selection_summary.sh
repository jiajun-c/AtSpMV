#!/bin/bash
# Run after the periodic collector exits, so the final total includes the
# absolute-reference evaluation produced by the last collection pass.
set -euo pipefail
cd /staff/chengjiajun/workspace/AtSpmv
out=results/model/selection-model/all-datasets-summary
while tmux has-session -t atspmv_summary 2>/dev/null; do sleep 30; done
for session in atspmv_full_selection atspmv_v2_queued atspmv_medium_queued \
               atspmv_outside_queued atspmv_extra_queued atspmv_resource_retry; do
    if tmux has-session -t "$session" 2>/dev/null; then
        printf 'Collector exited while %s is still running; refusing to finalize.\n' "$session" >&2
        exit 1
    fi
done
# First pass refreshes the absolute-reference evaluation; the second folds it
# into the total. The periodic script also reports failed/pending datasets.
bash "$out/provenance/refresh.sh"
bash "$out/provenance/refresh.sh"
v2=results/model/selection-model/h100-v2-full-2247
cp "$v2/v2-paired-analysis.csv" "$out/final-v2-bootstrap-input.csv"
python3 "$out/provenance/analyze_v2_uncertainty.py" \
    "$out/final-v2-bootstrap-input.csv" "$out/final-v2-bootstrap.json"
printf 'Final refresh completed; check summary and audits for pending or failed cases.\n'
