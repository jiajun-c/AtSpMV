#!/bin/bash
set -eu
cd /staff/chengjiajun/workspace/AtSpmv
out=results/model/hybrid-time-model/shared-atbsr-paired80
previous=results/model/hybrid-time-model/unified-policy-full-2794
pid=$(cat "$previous/queue.pid")
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'unified-policy-full-2794/run-after-vector-full.sh'; do sleep 30; done
# Avoid disturbing another process between the queued sweep and correctness.
idle=0
while [ "$idle" -lt 2 ]; do
    state=$(nvidia-smi -i 3 --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits)
    if awk -F, '{exit !($1 == 0 && $2 > 70000)}' <<< "$state"; then idle=$((idle+1)); else idle=0; fi
    sleep 15
done
CUDA_VISIBLE_DEVICES=3 "$out/provenance/spmv_plan_test" > "$out/correctness.log" 2>&1
CUDA_VISIBLE_DEVICES=3 LD_PRELOAD="$out/provenance/cuda_malloc_fault.so" ATSPMV_ENABLE_FAIL_MALLOC_TEST=1 compute-sanitizer --tool memcheck --error-exitcode 99 "$out/provenance/spmv_plan_test" --quick > "$out/memcheck.log" 2>&1
while true; do
    rc=0
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 --monitor-resident --selection auto_tuned_shared --compare-full-tune --counterbalance --extended-baselines --internal-graph-events >> "$out/runner.log" 2>&1 || rc=$?
    if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/audit_plan_sweep.py" "$out"
python3 "$out/provenance/analyze_plan_selection.py" "$out"
