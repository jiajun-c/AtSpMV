#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for group in prior second diverse; do
 out="results/model/sptrsv/components-broad-$group-h100"
 case "$group" in
  prior) factors=results/model/sptrsv/ilu-selector-holdout-h100/factors;;
  second) factors=results/model/sptrsv/ilu-second-holdout-h100/factors;;
  diverse) factors=results/model/sptrsv/components-diverse-h100/factors;;
 esac
 mkdir -p "$out/provenance"
 cp results/model/sptrsv/components-cached-paired-prior-h100/provenance/bench "$out/provenance/bench"
 cp scripts/analysis/{benchmark_sptrsv_components.py,analyze_sptrsv_component_pairs.py,run_on_idle_gpu.py,benchmark_dsm_atbsr_full.py} "$out/provenance/"
 find "$out/provenance" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$out/provenance.sha256"
 while true; do
  python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index 3 python3 "$out/provenance/benchmark_sptrsv_components.py" --manifest "$out/manifest.json" --factors "$factors" --output "$out/cases" --exe "$out/provenance/bench" --paired --resume >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 python3 "$out/provenance/analyze_sptrsv_component_pairs.py" "$out" >> "$out/runner.log" 2>&1
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
