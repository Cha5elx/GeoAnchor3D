#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

require_env LLM_PATH
require_env BASELINE_CHECKPOINT
require_env FULL_CHECKPOINT

output_dir="${OUTPUT_DIR:-$REPO_ROOT/reviewer_response/results/efficiency/$(timestamp)}"
mkdir -p "$output_dir"

for model_name in baseline geoanchor3d; do
    if [[ "$model_name" == "baseline" ]]; then
        checkpoint="$BASELINE_CHECKPOINT"
    else
        checkpoint="$FULL_CHECKPOINT"
    fi

    python reviewer_response/benchmark_efficiency.py \
        --model "$model_name" \
        --mode inference \
        --llm-path "$LLM_PATH" \
        --checkpoint "$checkpoint" \
        --output "$output_dir/${model_name}_inference.json" \
        --warmup-batches "${INFERENCE_WARMUP_BATCHES:-20}" \
        --timed-batches "${INFERENCE_TIMED_BATCHES:-200}" \
        --fixed-new-tokens "${FIXED_NEW_TOKENS:-32}" \
        --num-workers "${NUM_WORKERS:-4}"

    python reviewer_response/benchmark_efficiency.py \
        --model "$model_name" \
        --mode training \
        --llm-path "$LLM_PATH" \
        --checkpoint "$checkpoint" \
        --output "$output_dir/${model_name}_training.json" \
        --warmup-batches "${TRAIN_WARMUP_BATCHES:-10}" \
        --timed-batches "${TRAIN_TIMED_BATCHES:-50}" \
        --batch-size "${TRAIN_BATCH_SIZE:-1}" \
        --num-workers "${NUM_WORKERS:-4}"
done
