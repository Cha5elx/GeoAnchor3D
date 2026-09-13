#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

resolve_full_checkpoint
require_env FULL_CHECKPOINT

output_root="${OUTPUT_DIR:-$REVIEWER_OUTPUT_ROOT/proposal_robustness/$(timestamp)}"
mkdir -p "$output_root"

for keep_count in ${PROPOSAL_COUNTS:-100 75 50 25}; do
    run_dir="$output_root/k${keep_count}"
    mkdir -p "$run_dir"
    echo "Evaluating GeoAnchor3D with the first ${keep_count} serialized Mask3D proposals"
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-2}" run_python tasks/train.py scripts/config.py \
        output_dir "$run_dir" \
        pretrained_path "$FULL_CHECKPOINT" \
        evaluate True \
        auto_resume False \
        wandb.enable False \
        gpu_num "${TRAIN_NPROC_PER_NODE:-2}" \
        batch_size 8 \
        val_tag "${PROPOSAL_VAL_TAG:-scanrefer#multi3dref}" \
        proposal_keep_count "$keep_count" \
        model.llama_model_path "$LLM_PATH" \
        model.add_scene_token False \
        model.max_obj_num 100 \
        model.gate_granularity per_head \
        model.alpha_ablation_mode 0 \
        model.use_gate_supervision True \
        model.gate_loss_weight 1.0 \
        model.coord_loss_weight 0.1 \
        lora.lora_r "${LORA_RANK:-16}" \
        seed "${SEED:-42}" 2>&1 | tee "$run_dir/eval.log"
done
