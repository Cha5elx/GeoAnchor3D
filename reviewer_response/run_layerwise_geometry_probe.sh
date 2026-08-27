#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

require_env LLM_PATH
require_env BASELINE_CHECKPOINT
require_env FULL_CHECKPOINT

output_dir="${OUTPUT_DIR:-$REPO_ROOT/reviewer_response/results/layerwise_geometry_probe/$(timestamp)}"
mkdir -p "$output_dir"

for model_name in baseline geoanchor3d; do
    if [[ "$model_name" == "baseline" ]]; then
        checkpoint="$BASELINE_CHECKPOINT"
    else
        checkpoint="$FULL_CHECKPOINT"
    fi
    python reviewer_response/layerwise_geometry_probe.py \
        --model "$model_name" \
        --llm-path "$LLM_PATH" \
        --checkpoint "$checkpoint" \
        --output "$output_dir/${model_name}.json" \
        --layers "${PROBE_LAYERS:-4,8,12,16,20,24,28,32}" \
        --max-scenes "${PROBE_MAX_SCENES:-100}" \
        --max-objects-per-scene "${PROBE_MAX_OBJECTS_PER_SCENE:-100}" \
        --probe-epochs "${PROBE_EPOCHS:-25}" \
        --seed "${SEED:-42}"
done
