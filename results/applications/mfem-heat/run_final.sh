#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for case in final-amr-pilot final-small-full final-large-full; do
 out=results/applications/mfem-heat/$case
 mkdir -p "$out"
 mesh=/tmp/atspmv-mfem-v4.8/data/star.mesh
 refine=5
 steps=20
 rounds=4
 prefetch=0
 backends=cusparse,mfem,atbsr,mixed
 if [ "$case" = final-amr-pilot ]; then
  mesh=/tmp/atspmv-mfem-v4.8/data/amr-hex.mesh
  refine=2
  steps=5
  rounds=1
  backends=cusparse,atbsr,mixed,tuned
 fi
 if [ "$case" = final-large-full ]; then refine=7;prefetch=4;backends=cusparse,mfem,atbsr;fi
 while true; do
  OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 results/applications/mfem-heat/final-provenance/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-final --mesh "$mesh" --output "$out" --refine "$refine" --steps "$steps" --alpha 0.01 --rounds "$rounds" --prefetch "$prefetch" --backends "$backends" --gpu-assembly >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
done
