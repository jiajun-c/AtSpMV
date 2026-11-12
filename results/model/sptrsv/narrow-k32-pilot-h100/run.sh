#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/narrow-k32-pilot-h100
while tmux has-session -t atsptrsv_narrowthreads 2>/dev/null; do sleep 15; done
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_shape_pair.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/cases" --exe "$out/provenance/bench" --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
