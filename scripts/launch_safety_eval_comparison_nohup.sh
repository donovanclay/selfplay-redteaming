#!/usr/bin/env bash
#
# Launch full base vs trained safety eval in the background (survives disconnect).
#
# Usage:
#   bash scripts/launch_safety_eval_comparison_nohup.sh
#   bash scripts/launch_safety_eval_comparison_nohup.sh --gpus 1,2,3,4,5,6,7
#
# Monitor:
#   tail -f logs/safety_compare_full.log
#   cat logs/safety_compare.pid
#
# Stop:
#   kill $(cat logs/safety_compare.pid)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
PID_FILE="$LOG_DIR/safety_compare.pid"
LOG_FILE="$LOG_DIR/safety_compare_full.log"

GPUS="${GPUS:-1,2,3,4,5,6,7}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Safety comparison already running (pid $(cat "$PID_FILE"))."
    echo "  tail -f $LOG_FILE"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

cd "$REPO_ROOT"
nohup bash "$SCRIPT_DIR/run_safety_eval_comparison.sh" \
    --skip-smoke \
    --gpus "$GPUS" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "Full safety comparison launched (nohup)."
echo "  PID:     $(cat "$PID_FILE")"
echo "  Log:     $LOG_FILE"
echo "  Monitor: tail -f $LOG_FILE"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
