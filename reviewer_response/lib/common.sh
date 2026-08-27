#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Required environment variable is missing: $name" >&2
        exit 2
    fi
}

timestamp() {
    date +"%Y%m%d_%H%M%S"
}

run_python() {
    local nproc="${NPROC_PER_NODE:-1}"
    if [[ "$nproc" -gt 1 ]]; then
        torchrun --nproc_per_node="$nproc" --master_port="${MASTER_PORT:-29501}" "$@"
    else
        python "$@"
    fi
}

run_ablation_training() {
    local experiment="$1"
    local granularity="$2"
    local gate_supervision="$3"

    require_env LLM_PATH
    require_env INIT_CHECKPOINT
    require_env TRAIN_TAG
    require_env VAL_TAG

    local train_nproc="${TRAIN_NPROC_PER_NODE:-2}"
    if [[ "$train_nproc" -ne 2 ]]; then
        echo "Reviewer-response full training is configured for exactly 2 GPUs." >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    if [[ "${#visible_devices[@]}" -ne 2 ]]; then
        echo "CUDA_VISIBLE_DEVICES must contain exactly 2 GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
        exit 2
    fi

    local output_dir="${OUTPUT_DIR:-$REPO_ROOT/reviewer_response/results/$experiment/$(timestamp)}"
    local gate_loss_weight="${GATE_LOSS_WEIGHT:-1.0}"
    if [[ "$gate_supervision" == "False" ]]; then
        gate_loss_weight=0.0
    fi
    mkdir -p "$output_dir"

    NPROC_PER_NODE="$train_nproc" run_python tasks/train.py scripts/config.py \
        output_dir "$output_dir" \
        scheduler.epochs "${EPOCHS:-3}" \
        optimizer.lr "${LEARNING_RATE:-5e-6}" \
        pretrained_path "$INIT_CHECKPOINT" \
        evaluate False \
        auto_resume False \
        wandb.enable "${ENABLE_WANDB:-False}" \
        gpu_num "$train_nproc" \
        batch_size "${BATCH_SIZE:-16}" \
        train_tag "$TRAIN_TAG" \
        val_tag "$VAL_TAG" \
        model.llama_model_path "$LLM_PATH" \
        model.max_obj_num "${MAX_OBJ_NUM:-100}" \
        model.gate_granularity "$granularity" \
        model.alpha_ablation_mode 0 \
        model.use_gate_supervision "$gate_supervision" \
        model.gate_loss_weight "$gate_loss_weight" \
        model.coord_loss_weight "${COORD_LOSS_WEIGHT:-0.1}" \
        lora.lora_r "${LORA_RANK:-16}" \
        seed "${SEED:-42}"
}
