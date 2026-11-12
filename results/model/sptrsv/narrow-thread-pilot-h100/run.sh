#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/narrow-thread-pilot-h100
while tmux has-session -t atsptrsv_slowdiagnosis 2>/dev/null; do sleep 15; done
while true; do
 python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_threads.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/cases" --exe "$out/provenance/bench" --narrow --resume >> "$out/runner.log" 2>&1
 status=$?
 if [ "$status" -ne 75 ]; then break; fi
done
python3 "$out/provenance/analyze_sptrsv_threads.py" "$out" --narrow >> "$out/runner.log" 2>&1
printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
exit "$status"
