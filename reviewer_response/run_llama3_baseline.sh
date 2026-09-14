#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/llama3.sh"
run_backbone_training "llama3_baseline" False False False 0.0 0.0
