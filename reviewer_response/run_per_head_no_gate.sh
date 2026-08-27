#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# Full-model retraining: per-head instruction gate without task-type prior loss.
run_ablation_training "per_head_no_gate_loss" "per_head" "False"
