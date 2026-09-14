#!/usr/bin/env bash
export TRAIN_TAG="${TRAIN_TAG:-scanrefer#obj_align#nr3d_caption#scanqa}"
export VAL_TAG="${VAL_TAG:-scanrefer#scanqa}"
export REVIEWER_OUTPUT_ROOT="${REVIEWER_OUTPUT_ROOT:-/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/outputs/reviewer_response}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-/data/ZXMIC/mic_lcx/Chat-Scene/Chat-Scene/pretrained_models/ckpt_01_3446.pth}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ALT_LLM_PATH="${ALT_LLM_PATH:-/data/ZXMIC/mic_lcx/llama-3-8b/Meta-Llama-3-8B-Instruct}"
export BACKBONE_LLAMA3_MODE=True
export BACKBONE_BATCH_SIZE="${BACKBONE_BATCH_SIZE:-1}"
export BACKBONE_ATTN_IMPLEMENTATION="${BACKBONE_ATTN_IMPLEMENTATION:-sdpa}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ ! -f "$ALT_LLM_PATH/config.json" || ! -f "$ALT_LLM_PATH/tokenizer.json" ]]; then
    echo "Expected Hugging Face config.json and tokenizer.json in $ALT_LLM_PATH" >&2
    exit 2
fi
