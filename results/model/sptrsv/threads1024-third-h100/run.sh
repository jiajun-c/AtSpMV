#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/threads1024-third-h100
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_threads.py" --manifest "$out/manifest.json" --factors results/model/sptrsv/ilu-expanded-h100/factors --output "$out/cases" --exe "$out/provenance/bench" --include-1024 --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/provenance/analyze_sptrsv_threads.py" "$out" --include-1024 >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
