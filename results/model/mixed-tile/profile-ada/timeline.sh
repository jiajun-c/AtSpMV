#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/profile-ada
runner=2174390
kill -STOP "$runner" || exit 1
trap 'kill -CONT "$runner" 2>/dev/null || true' EXIT
while ps -o stat= --ppid "$runner" | rg -q '^[^Z]'; do sleep 1; done
for pair in 'Chebyshev1:mixed_bal4' 'Chebyshev1:m1_b0' 'rajat09:mixed' 'rajat09:m32_b1'; do
 matrix=${pair%%:*}; candidate=${pair#*:}; base="$out/${matrix}_${candidate}_timeline"
 CUDA_VISIBLE_DEVICES=1 ATSPMV_MIXED_PROFILE="$candidate" /usr/local/cuda-13.0/bin/nsys profile --trace=cuda --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$base" /tmp/atspmv-dsm-sm90-build/mixed_tile_bench "/staff/chengjiajun/workspace/database/suitsparse/$matrix/$matrix.mtx" 20261028 1 0 > "$base.log" 2>&1
 /usr/local/cuda-13.0/bin/nsys export --type sqlite --force-overwrite true -o "$base.sqlite" "$base.nsys-rep" >> "$base.log" 2>&1
done
