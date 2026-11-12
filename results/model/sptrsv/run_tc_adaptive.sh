#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for item in first second; do
 out="results/model/sptrsv/tcsptrsv-adaptive-$item-h100"
 if [ "$item" = first ]; then factors=results/model/sptrsv/tcsptrsv-aligned-h100/factors; else factors=results/model/sptrsv/tcsptrsv-aligned-second-h100/factors; fi
 mkdir -p "$out/provenance"
 cp results/model/sptrsv/tcsptrsv-aligned-h100/provenance/{tcsptrsv,tcsptrsv.patch} "$out/provenance/"
 cp /tmp/atspmv-sptrsv-cuda126/sptrsv_plan_bench "$out/provenance/plan"
 cp src/sptrsvPlan.cu include/{sptrsvPlan.h,sptrsvSelection.h} include/detail/{sptrsvPacking.h,sptrsvShared.cuh} bench/{sptrsv_plan_bench.cu,sptrsv_residual.h} "$out/provenance/"
 cp scripts/analysis/{benchmark_sptrsv_tc.py,sptrsv_matrix_contract.py,analyze_sptrsv_tc.py,run_on_idle_gpu.py,benchmark_dsm_atbsr_full.py} "$out/provenance/"
 find "$out/provenance" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/provenance.sha256"
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_tc.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --plan "$out/provenance/plan" --tc "$out/provenance/tcsptrsv" --tc-workdir /tmp/atspmv-tcsptrsv/TC-SpTRSV/TCSpTRSV/script --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 python3 "$out/provenance/analyze_sptrsv_tc.py" "$out" --aligned >> "$out/runner.log" 2>&1
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
