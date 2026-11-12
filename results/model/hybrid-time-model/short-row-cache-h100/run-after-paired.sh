#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/hybrid-time-model/short-row-cache-h100
previous=results/model/hybrid-time-model/shared-row-sort-paired80
pid=$(cat "$previous/queue.pid")
while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | rg -q 'shared-row-sort-paired80/run-after-correctness.sh'; do sleep 30; done
for mode in direct memcheck; do
    if [ "$mode" = direct ]; then prefix=(); else prefix=(/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 99); fi
    while true; do
        python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 bash -c 'target=$1; shift; exec "$@" > "$target.tmp" 2>&1' _ "$out/$mode.log" "${prefix[@]}" "$out/provenance/spmv_plan_test" --quick
        rc=$?
        if [ "$rc" -eq 75 ]; then rm -f "$out/$mode.log.tmp"; continue; fi
        if [ "$rc" -ne 0 ]; then mv "$out/$mode.log.tmp" "$out/$mode.failed.log"; exit "$rc"; fi
        mv "$out/$mode.log.tmp" "$out/$mode.log" || exit 1
        break
    done
done
while true; do
    python3 "$out/provenance/benchmark_plan_selection.py" --exe "$out/provenance/spmv_hybrid_validation" --index "$out/index.csv" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out" --gpu-index 3 --monitor-resident --selection auto_tuned --counterbalance --extended-baselines >> "$out/runner.log" 2>&1
    rc=$?
    if [ "$rc" -ne 75 ]; then break; fi
done
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
python3 "$out/provenance/audit_plan_sweep.py" "$out" || exit $?
python3 "$out/provenance/audit_tuned_choice.py" "$out" || exit $?
python3 "$out/provenance/analyze_plan_selection.py" "$out" || exit $?
python3 "$out/provenance/analyze_plan_amortization.py" "$out" || exit $?
