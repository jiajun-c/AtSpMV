#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for group in prior second; do
 out="results/model/sptrsv/components-cached-paired-$group-h100"
 if [ "$group" = prior ]; then factors=results/model/sptrsv/ilu-selector-holdout-h100/factors; else factors=results/model/sptrsv/ilu-second-holdout-h100/factors; fi
 mkdir -p "$out/provenance"
 cp /tmp/atspmv-sptrsv-cuda126/sptrsv_level_bench "$out/provenance/bench"
 cp bench/sptrsv_level_bench.cu bench/sptrsv_*.cuh include/detail/{sptrsvPacking.h,sptrsvShared.cuh} include/sptrsvSelection.h "$out/provenance/"
 cp scripts/analysis/{benchmark_sptrsv_components.py,run_on_idle_gpu.py,benchmark_dsm_atbsr_full.py} "$out/provenance/"
 find "$out/provenance" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/provenance.sha256"
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_components.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --exe "$out/provenance/bench" --paired --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
