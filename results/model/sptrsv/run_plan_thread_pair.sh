#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for item in third train holdout; do
 out="results/model/sptrsv/plan-thread-$item-h100"
 case "$item" in
  third) factors=results/model/sptrsv/ilu-expanded-h100/factors;;
  train) factors=results/model/sptrsv/ilu-selector-holdout-h100/factors;;
  holdout) factors=results/model/sptrsv/ilu-second-holdout-h100/factors;;
 esac
 mkdir -p "$out/provenance"
 cp results/model/sptrsv/componentwise-upper-h100/provenance/plan "$out/provenance/fixed"
 cp /tmp/atspmv-sptrsv-cuda126/sptrsv_plan_bench "$out/provenance/adaptive"
 cp src/sptrsvPlan.cu include/{sptrsvPlan.h,sptrsvSelection.h} include/detail/{sptrsvPacking.h,sptrsvShared.cuh} bench/{sptrsv_plan_bench.cu,sptrsv_residual.h} "$out/provenance/"
 cp scripts/analysis/{benchmark_sptrsv_plan_pair.py,analyze_sptrsv_plan_pair.py,run_on_idle_gpu.py,benchmark_dsm_atbsr_full.py} "$out/provenance/"
 find "$out/provenance" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/provenance.sha256"
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_plan_pair.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --fixed "$out/provenance/fixed" --adaptive "$out/provenance/adaptive" --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 python3 "$out/provenance/analyze_sptrsv_plan_pair.py" "$out" >> "$out/runner.log" 2>&1
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
