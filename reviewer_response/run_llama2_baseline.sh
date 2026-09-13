#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# Chat-Scene-style control under the alternative LLaMA-2 backbone.
run_backbone_training "llama2_baseline" False False False 0.0 0.0
