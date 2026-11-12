#!/bin/bash
set -u
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
out=results/model/cusparse-spmm-core-compare/h100-synthetic-v1
gpu_index=${ATSPMV_SPMM_GPU_INDEX:-3}
while true; do
    python3 "$out/provenance/run_on_idle_gpu.py" --gpu-index "$gpu_index" \
        python3 "$out/provenance/profile_cusparse_spmm_cores.py" \
        --exe "$out/provenance/cusparse_spmm_core_compare" --out "$out/instruction-profile" \
        >> "$out/instruction-profile-runner.log" 2>&1
    rc=$?
    [ "$rc" -eq 75 ] || break
done
printf 'profiler runner exit code: %s\n' "$rc" >> "$out/instruction-profile-runner.log"
exit "$rc"
