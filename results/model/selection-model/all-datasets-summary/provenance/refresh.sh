#!/bin/bash
set -u
cd /staff/chengjiajun/workspace/AtSpmv || exit 1
out=results/model/selection-model/all-datasets-summary
while true; do
    python3 scripts/analysis/summarize_all_selection_results.py \
        --inventory results/model/selection-model/full-inventory/inventory.csv \
        --baseline results/model/selection-model/h100-full-2247 \
        --v2 results/model/selection-model/h100-v2-full-2247 \
        --outside results/model/selection-model/h100-outside-scope-550 \
        --structural-audit results/model/selection-model/unknown-scope-audit/input-status.csv \
        --additional-scan results/model/selection-model/unindexed-25-scan \
        --additional-outside results/model/selection-model/h100-unindexed-extra \
        --resource-retry results/model/selection-model/h100-resource-retry \
        --absolute-evaluation results/model/selection-model/h100-absolute-reference-v1/reference-evaluation.csv \
        --output "$out" >> "$out/refresh.log" 2>&1
    if [ -f results/model/selection-model/h100-full-2247/manifest.json ]; then
        python3 "$out/provenance/analyze_refined_misses.py" results/model/selection-model/h100-full-2247 >> "$out/refresh.log" 2>&1
        python3 "$out/provenance/analyze_refined_amortization.py" results/model/selection-model/h100-full-2247 >> "$out/refresh.log" 2>&1
    fi
    for run in h100-full-2247 h100-v2-full-2247 h100-outside-scope-550 h100-unindexed-extra h100-resource-retry; do
        directory=results/model/selection-model/$run
        if [ -f "$directory/manifest.json" ]; then
            python3 "$out/provenance/audit_plan_sweep.py" "$directory" >> "$out/refresh.log" 2>&1
        fi
    done
    if [ -f results/model/selection-model/h100-v2-full-2247/manifest.json ]; then
        python3 "$out/provenance/analyze_vector_v2_paired.py" results/model/selection-model/h100-v2-full-2247 \
            --training-dataset "$out/provenance/vector-v2-training.csv" >> "$out/refresh.log" 2>&1
        reference=results/model/selection-model/h100-absolute-reference-v1
        python3 "$reference/provenance/evaluate_h100_absolute_reference.py" \
            --run results/model/selection-model/h100-v2-full-2247 --model "$reference/model.joblib" \
            --training "$reference/training-data.csv" --splits "$reference/provenance/splits.csv" \
            --features "$reference/provenance/features-exact.csv" \
            --additional-feature-hashes "$reference/provenance/additional-feature-hashes.json" \
            --historical-status results/model/hybrid-time-model/plan-tune-2247-v4/status --output "$reference" >> "$out/refresh.log" 2>&1
    fi
    if [ -f results/model/selection-model/h100-medium-holdout38/metadata.json ]; then
        flock results/model/selection-model/h100-medium-holdout38/analysis.lock python3 results/model/selection-model/h100-medium-holdout38/provenance/analyze_medium_unified.py \
            results/model/selection-model/h100-medium-holdout38 \
            --replay results/model/selection-model/h100-medium-holdout38/provenance/replay.csv >> "$out/refresh.log" 2>&1
    fi
    if ! tmux has-session -t atspmv_full_selection 2>/dev/null && \
       ! tmux has-session -t atspmv_v2_queued 2>/dev/null && \
       ! tmux has-session -t atspmv_medium_queued 2>/dev/null && \
       ! tmux has-session -t atspmv_outside_queued 2>/dev/null && \
       ! tmux has-session -t atspmv_input_audit 2>/dev/null && \
       ! tmux has-session -t atspmv_extra_queued 2>/dev/null && \
       ! tmux has-session -t atspmv_resource_retry 2>/dev/null; then
        break
    fi
    sleep 300
done
