#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# Full GeoAnchor3D under the same alternative LLaMA-2 backbone.
run_backbone_training "llama2_full" True True True 1.0 0.1
