#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for case in coherent-r5 coherent-r7; do
 out=results/applications/mfem-heat/$case
 mkdir -p "$out"
 refine=7
 steps=20
 if [ "$case" = coherent-r5 ]; then refine=5;steps=100;fi
 while true; do
  OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 results/applications/mfem-heat/coherent-provenance/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-coherent --mesh /tmp/atspmv-mfem-v4.8/data/star.mesh --output "$out" --refine "$refine" --steps "$steps" --alpha 0.01 --rounds 4 --prefetch 4 --backends cusparse,mfem-rebuild,mfem-hostcoeff,mfem,atbsr,mixed --gpu-assembly >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
