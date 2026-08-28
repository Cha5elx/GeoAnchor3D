#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib/common.sh"

experiment="${1:-}"
case "$experiment" in
    full_per_head)
        script="reviewer_response/run_full_per_head.sh"
        ;;
    dynamic_scalar)
        script="reviewer_response/run_dynamic_scalar.sh"
        ;;
    per_head_no_gate)
        script="reviewer_response/run_per_head_no_gate.sh"
        ;;
    within_task_gating)
        resolve_full_checkpoint
        script="reviewer_response/run_within_task_gating.sh"
        ;;
    layerwise_geometry_probe)
        resolve_full_checkpoint
        script="reviewer_response/run_layerwise_geometry_probe.sh"
        ;;
    efficiency)
        resolve_full_checkpoint
        script="reviewer_response/run_efficiency.sh"
        ;;
    *)
        echo "Usage: bash reviewer_response/launch_background.sh EXPERIMENT" >&2
        echo "Experiments: full_per_head, dynamic_scalar, per_head_no_gate," >&2
        echo "             within_task_gating, layerwise_geometry_probe, efficiency" >&2
        exit 2
        ;;
esac

output_dir="${OUTPUT_DIR:-$REVIEWER_OUTPUT_ROOT/$experiment/$(timestamp)}"
mkdir -p "$output_dir"
export OUTPUT_DIR="$output_dir"

if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required for SSH-independent background execution." >&2
    exit 2
fi

log_file="$output_dir/run.log"
pid_file="$output_dir/run.pid"
nohup setsid bash "$script" >"$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$pid_file"

echo "Started: $experiment"
echo "PID:     $pid"
echo "Output:  $output_dir"
echo "Log:     $log_file"
echo "Watch:   tail -f '$log_file'"
