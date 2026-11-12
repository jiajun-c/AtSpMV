#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-resource-retry
outside=results/model/selection-model/h100-outside-scope-550
extra=results/model/selection-model/h100-unindexed-extra
while tmux has-session -t atspmv_extra_queued 2>/dev/null || tmux has-session -t atspmv_outside_queued 2>/dev/null; do sleep 30; done
if ! grep -q '^runner exit code: 0$' "$outside/runner.log" 2>/dev/null || ! grep -q '^runner exit code: 0$' "$extra/runner.log" 2>/dev/null; then
    printf 'Source runs did not finish successfully.\n' >> "$out/runner.log"
    exit 1
fi
python3 "$out/provenance/prepare_resource_retry.py" --run "$outside" --run "$extra" --output "$out" >> "$out/runner.log" 2>&1 || exit 1
if [ "$(wc -l < "$out/index.csv")" -le 1 ]; then
    printf 'No eligible resource failures.\nrunner exit code: 0\n' >> "$out/runner.log"
    exit 0
fi
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root "$outside/matrix-overlay" --output "$out" --gpu-index 2 --selection auto_model_refined_vector --counterbalance --internal-graph-events --timeout 3600 >> "$out/runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then
    python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
