#!/usr/bin/env bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/sptrsv/components-diverse-h100
OPENBLAS_NUM_THREADS=1 python3 "$out/provenance/generate_sptrsv_factor_manifest.py" --manifest "$out/manifest.json" --matrix-root /staff/chengjiajun/workspace/database/suitsparse --output "$out/factors" --generator "$out/provenance/generate_sptrsv_ilu_factors.py" --resume >> "$out/generation.log" 2>&1 || exit 1
OPENBLAS_NUM_THREADS=1 python3 "$out/provenance/sptrsv_component_stats.py" --manifest "$out/manifest.json" --factors "$out/factors" --output "$out/component-stats.csv" >> "$out/generation.log" 2>&1
