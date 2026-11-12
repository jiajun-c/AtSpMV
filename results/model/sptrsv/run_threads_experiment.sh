#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
# The first probe has a live idle-GPU monitor. Wait for it before rebuilding
# the executable or starting another GPU monitor.
while kill -0 507914 2>/dev/null && ps -p 507914 -o args= | grep -q 'bcsstk04-L.mtx'; do sleep 15; done
PATH=/usr/bin:/bin:/usr/local/cuda-12.6/bin cmake --build /tmp/atspmv-sptrsv-cuda126 --target sptrsv_level_bench -j 4 > results/model/sptrsv/threads-build.log 2>&1 || exit 1
for item in train holdout; do
 out="results/model/sptrsv/threads-$item-h100"
 if [ "$item" = train ]; then factors=results/model/sptrsv/ilu-selector-holdout-h100/factors; else factors=results/model/sptrsv/ilu-second-holdout-h100/factors; fi
 mkdir -p "$out/provenance"
 cp /tmp/atspmv-sptrsv-cuda126/sptrsv_level_bench "$out/provenance/bench"
 cp bench/sptrsv_level_bench.cu bench/sptrsv_*.cuh "$out/provenance/"
 cp include/detail/{sptrsvPacking.h,sptrsvShared.cuh} include/sptrsvSelection.h "$out/provenance/"
 cp scripts/analysis/{benchmark_sptrsv_threads.py,analyze_sptrsv_threads.py,run_on_idle_gpu.py,benchmark_dsm_atbsr_full.py} "$out/provenance/"
 find "$out/provenance" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/provenance.sha256"
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_threads.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --exe "$out/provenance/bench" --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 python3 "$out/provenance/analyze_sptrsv_threads.py" "$out" >> "$out/runner.log" 2>&1
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
