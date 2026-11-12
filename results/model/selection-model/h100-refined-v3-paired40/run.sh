#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/h100-refined-v3-paired40
uuid=$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader) || exit 1
printf 'Waiting for exclusive H100 GPU 2\n' > "$out/precheck-status.txt"
while true; do
 apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || { sleep 30; continue; }
 util=$(nvidia-smi -i 2 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || { sleep 30; continue; }
 if printf '%s\n' "$apps" | grep -Fxq "$uuid" || [ "$util" != 0 ]; then sleep 30; continue; fi
 sleep 30
 apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || continue
 if ! printf '%s\n' "$apps" | grep -Fxq "$uuid"; then break; fi
done
printf 'Running H100 Plan correctness checks\n' > "$out/precheck-status.txt"
CUDA_VISIBLE_DEVICES=2 "$out/provenance/spmv_plan_test" --quick > "$out/correctness.log" 2>&1 || { printf 'Correctness failed\n' > "$out/precheck-status.txt"; exit 1; }
printf 'Correctness passed; starting paired experiment\n' > "$out/precheck-status.txt"
while true; do
 python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 2 --selection auto_tuned_mixed --compare-auto-model --compare-vector-model --compare-refined-model --compare-full-tune --counterbalance --warm-setup-anchor --alias-identical-models --alias-identical-graphs --internal-graph-events >> "$out/runner.log" 2>&1
 rc=$?
 [ "$rc" -eq 75 ] || break
done
if [ "$rc" -eq 0 ]; then
 python3 "$out/provenance/analyze_plan_selection.py" "$out" > "$out/summary.log" 2>&1
 python3 "$out/provenance/analyze_selection_oracle.py" "$out" > "$out/oracle-summary.log" 2>&1
fi
printf 'runner exit code: %s\n' "$rc" >> "$out/runner.log"
exit "$rc"
