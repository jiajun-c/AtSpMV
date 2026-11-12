#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
while tmux has-session -t atsptrsv_capelliniwarm 2>/dev/null; do sleep 15; done
for item in 50 160; do
 out="results/model/sptrsv/capellini-clean$item-h100"
 if [ "$item" = 50 ]; then factors=results/model/sptrsv/tcsptrsv-aligned-h100/factors; else factors=results/model/sptrsv/tcsptrsv-independent160-h100/factors; fi
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_tc.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --plan "$out/provenance/plan" --tc "$out/provenance/capellini" --tc-workdir "$out/work" --timeout 25 --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 python3 "$out/provenance/analyze_sptrsv_capellini.py" "$out" --parser "$out/provenance/analyze_sptrsv_tc.py" --warmup-count 5 --poison-outside-timing >> "$out/runner.log" 2>&1
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
