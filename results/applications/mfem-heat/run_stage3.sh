#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for case in nonlinear-r5-validated nonlinear-r6-full nonlinear-r7-full large-r5-prefetch8; do
 out=results/applications/mfem-heat/$case
 mkdir -p "$out"
 mesh=/tmp/atspmv-mfem-v4.8/data/star.mesh
 refine=5
 alpha=0.01
 steps=20
 rounds=4
 prefetch=0
 backends=cusparse,mfem,atbsr,mixed
 extra=()
 if [ "$case" = nonlinear-r6-full ]; then refine=6;extra=(--gpu-assembly);fi
 if [ "$case" = nonlinear-r7-full ]; then refine=7;rounds=2;extra=(--gpu-assembly);fi
 if [ "$case" = large-r5-prefetch8 ]; then
  mesh=/tmp/atspmv-mfem-v4.8/data/fichera-mixed.mesh
  alpha=0
  steps=100
  rounds=2
  prefetch=8
  backends=cusparse,atbsr
 fi
 while true; do
  OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 results/applications/mfem-heat/stage3-provenance/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-stage3 --mesh "$mesh" --output "$out" --refine "$refine" --steps "$steps" --alpha "$alpha" --rounds "$rounds" --prefetch "$prefetch" --backends "$backends" "${extra[@]}" >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
 if [ "$status" -ne 0 ]; then exit "$status"; fi
done
