#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

# Full GeoAnchor3D: dynamic per-head IGGA + gate supervision + GATH.
run_ablation_training "full_per_head" "per_head" "True"
