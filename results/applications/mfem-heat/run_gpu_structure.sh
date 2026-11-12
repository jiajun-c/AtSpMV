#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
for case in gpu-structure-r5 gpu-structure-r7 gpu-structure-r7-ablation; do
 out=results/applications/mfem-heat/$case
 mkdir -p "$out"
 refine=7
 rounds=4
 backends=cusparse,mfem,atbsr,tuned
 extra=()
 if [ "$case" = gpu-structure-r5 ]; then refine=5;backends=cusparse,mfem,atbsr,mixed;fi
 if [ "$case" = gpu-structure-r7-ablation ]; then rounds=2;backends=cusparse,atbsr;extra=(--rebuild-implicit --rebuild-values);fi
 while true; do
  OPENBLAS_NUM_THREADS=1 python3 scripts/analysis/run_on_idle_gpu.py --gpu-index 3 python3 results/applications/mfem-heat/gpu-structure-provenance/benchmark_mfem_heat.py --exe /tmp/atspmv-mfem-gpu-structure --mesh /tmp/atspmv-mfem-v4.8/data/star.mesh --output "$out" --refine "$refine" --steps 20 --alpha 0.01 --rounds "$rounds" --prefetch 4 --backends "$backends" --gpu-assembly "${extra[@]}" >> "$out/runner.log" 2>&1
  status=$?
  if [ "$status" -ne 75 ]; then break; fi
 done
 printf 'runner_exit=%s\n' "$status" >> "$out/runner.log"
done
