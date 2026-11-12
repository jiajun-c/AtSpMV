#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/fixed-shapes-examples-h100
while true; do
 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_fixed_shapes.py" --exe "$out/provenance/bench" --factors results/model/sptrsv/tcsptrsv-independent160-h100/factors --output "$out" >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
