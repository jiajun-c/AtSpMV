#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/mixed-tile/profile-ada
runner=2174390
kill -STOP "$runner" || exit 1
trap 'kill -CONT "$runner" 2>/dev/null || true' EXIT
while ps -o stat= --ppid "$runner" | rg -q '^[^Z]'; do sleep 1; done
for pair in 'trans4:mixed_bal8' 'trans4:m8_b1_cap16' 'rajat09:mixed' 'rajat09:m32_b1' 'Chebyshev1:mixed_bal4' 'Chebyshev1:m1_b0' 'bloweybl:mixed_bal4' 'bloweybl:m8_b1'; do
 matrix=${pair%%:*}; candidate=${pair#*:}
 echo "START $pair $(date -Is)" >> "$out/progress.log"
 CUDA_VISIBLE_DEVICES=1 ATSPMV_MIXED_PROFILE="$candidate" /usr/local/cuda-13.0/bin/ncu --profile-from-start off --cache-control none --clock-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section InstructionStats --section SchedulerStats --section Occupancy --section LaunchStats --csv --log-file "$out/${matrix}_${candidate}.csv" /tmp/atspmv-dsm-sm90-build/mixed_tile_bench "/staff/chengjiajun/workspace/database/suitsparse/$matrix/$matrix.mtx" 20261028 1 0 > "$out/${matrix}_${candidate}.log" 2>&1
 rc=$?
 echo "DONE $pair rc=$rc $(date -Is)" >> "$out/progress.log"
done
