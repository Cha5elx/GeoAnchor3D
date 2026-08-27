#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# Full-model retraining: instruction-conditioned scalar gate shared by all heads.
run_ablation_training "dynamic_scalar" "scalar" "True"
