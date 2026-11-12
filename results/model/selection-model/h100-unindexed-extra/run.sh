#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-unindexed-extra
scan=results/model/selection-model/unindexed-25-scan
prior=results/model/selection-model/h100-outside-scope-550
while tmux has-session -t atspmv_input_audit 2>/dev/null || tmux has-session -t atspmv_outside_queued 2>/dev/null; do sleep 30; done
if ! grep -q '^scan completed$' "$scan/runner.log" 2>/dev/null || ! grep -q '^runner exit code: 0$' "$prior/runner.log" 2>/dev/null; then
    printf 'Input scan or previous H100 run did not finish successfully.\n' >> "$out/runner.log"
    exit 1
fi
python3 "$out/provenance/finalize_unindexed_selection.py" --scan "$scan" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" >> "$out/runner.log" 2>&1 || exit 1
if [ "$(wc -l < "$out/index.csv")" -le 1 ]; then
    printf 'No indexable matrices; runner exit code: 0\n' >> "$out/runner.log"
    exit 0
fi
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 2 --selection auto_model_refined --compare-full-tune --compare-mixed-tune --counterbalance --internal-graph-events --retry-shared-storage-on-oom --timeout 3600 >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then
    python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
