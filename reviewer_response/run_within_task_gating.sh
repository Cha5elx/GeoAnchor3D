#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

require_env LLM_PATH
require_env FULL_CHECKPOINT

output_dir="${OUTPUT_DIR:-$REVIEWER_OUTPUT_ROOT/within_task_gating/$(timestamp)}"
mkdir -p "$output_dir"

run_python tasks/train.py scripts/config.py \
    output_dir "$output_dir/evaluation" \
    pretrained_path "$FULL_CHECKPOINT" \
    evaluate True \
    auto_resume False \
    wandb.enable False \
    gpu_num "${NPROC_PER_NODE:-1}" \
    batch_size "${BATCH_SIZE:-1}" \
    train_tag scanrefer \
    val_tag scanrefer \
    model.llama_model_path "$LLM_PATH" \
    model.add_scene_token False \
    model.max_obj_num "${MAX_OBJ_NUM:-100}" \
    model.gate_granularity per_head \
    model.alpha_ablation_mode 0

prediction_files=("$output_dir"/evaluation/preds_*_scanrefer.json)
prediction_file="${prediction_files[0]:-}"
if [[ ! -f "$prediction_file" ]]; then
    echo "No merged ScanRefer prediction JSON was produced." >&2
    exit 1
fi

python reviewer_response/analyze_within_task_gating.py \
    --predictions "$prediction_file" \
    --output-json "$output_dir/within_task_gating.json" \
    --output-csv "$output_dir/within_task_gating.csv"
