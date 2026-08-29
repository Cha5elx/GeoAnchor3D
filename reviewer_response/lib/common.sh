#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVIEWER_OUTPUT_ROOT="${REVIEWER_OUTPUT_ROOT:-/data/lcx/chat-scene01/outputs/reviewer_response}"
LLM_PATH="${LLM_PATH:-/data/lcx/HuggingFace-Download-Accelerator/hf_hub/models--lmsys--vicuna-7b-v1.5}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/data/lcx/chat-scene/Chat-Scene/pretrained_models/ckpt_01_3446.pth}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$BASELINE_CHECKPOINT}"
TRAIN_TAG="${TRAIN_TAG:-scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref}"
VAL_TAG="${VAL_TAG:-scanrefer#multi3dref#scan2cap#scanqa}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export REVIEWER_OUTPUT_ROOT LLM_PATH BASELINE_CHECKPOINT INIT_CHECKPOINT
export TRAIN_TAG VAL_TAG TRAIN_NPROC_PER_NODE CUDA_VISIBLE_DEVICES
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

resolve_full_checkpoint() {
    if [[ -n "${FULL_CHECKPOINT:-}" ]]; then
        return
    fi

    local checkpoint
    checkpoint="$(find "$REVIEWER_OUTPUT_ROOT/full_per_head" -type f -name 'ckpt_02_*.pth' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
    if [[ -z "$checkpoint" ]]; then
        echo "No completed Full checkpoint was found under:" >&2
        echo "  $REVIEWER_OUTPUT_ROOT/full_per_head" >&2
        echo "Finish full_per_head first, or export FULL_CHECKPOINT explicitly." >&2
        exit 2
    fi
    FULL_CHECKPOINT="$checkpoint"
    export FULL_CHECKPOINT
    echo "Using Full checkpoint: $FULL_CHECKPOINT"
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
    if [[ -n "${RESUME_CHECKPOINT:-}" && ! -f "$RESUME_CHECKPOINT" ]]; then
        echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
        exit 2
    fi

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

    local output_dir="${OUTPUT_DIR:-$REVIEWER_OUTPUT_ROOT/$experiment/$(timestamp)}"
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
        resume_checkpoint_path "${RESUME_CHECKPOINT:-}" \
        evaluate False \
        auto_resume False \
        wandb.enable "${ENABLE_WANDB:-False}" \
        gpu_num "$train_nproc" \
        batch_size 8 \
        train_tag "$TRAIN_TAG" \
        val_tag "$VAL_TAG" \
        model.llama_model_path "$LLM_PATH" \
        model.add_scene_token False \
        model.max_obj_num "${MAX_OBJ_NUM:-100}" \
        model.gate_granularity "$granularity" \
        model.alpha_ablation_mode 0 \
        model.use_gate_supervision "$gate_supervision" \
        model.gate_loss_weight "$gate_loss_weight" \
        model.coord_loss_weight "${COORD_LOSS_WEIGHT:-0.1}" \
        lora.lora_r "${LORA_RANK:-16}" \
        seed "${SEED:-42}"
}
