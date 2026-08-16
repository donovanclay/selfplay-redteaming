#!/usr/bin/env bash
#
# Launch self-play red teaming training in the background.
# Safe to disconnect from SSH/Cursor after starting.
#
# Usage:
#   bash scripts/launch_selfplay_training.sh
#   bash scripts/launch_selfplay_training.sh --num-gpus 1 --tensor-parallel-size 1
#
# Monitor:
#   tail -f logs/reward_model.txt
#   tail -f logs/train.txt
#
# Stop:
#   kill $(cat logs/wildguard.pid) $(cat logs/train.pid) 2>/dev/null
#   ray stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=1
TP_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

HOST_IP="$(hostname -I | awk '{print $1}')"
export REMOTE_RM_URL="${REMOTE_RM_URL:-http://${HOST_IP}:5000/classify}"

echo "Repo:           $REPO_ROOT"
echo "Host IP:        $HOST_IP"
echo "Reward model:   $REMOTE_RM_URL"
echo "WildGuard GPUs: $NUM_GPUS (tensor parallel: $TP_SIZE)"
echo "Logs:           $LOG_DIR"
echo

# Step 1: Ray cluster (daemonizes on its own)
if ray status >/dev/null 2>&1; then
    echo "[1/3] Ray cluster already running."
else
    echo "[1/3] Starting Ray head..."
    ray start --head
fi

# Step 2: WildGuard reward model server
if [[ -f "$LOG_DIR/wildguard.pid" ]] && kill -0 "$(cat "$LOG_DIR/wildguard.pid")" 2>/dev/null; then
    echo "[2/3] WildGuard already running (pid $(cat "$LOG_DIR/wildguard.pid"))."
else
    echo "[2/3] Starting WildGuard reward model..."
    nohup bash "$SCRIPT_DIR/serve_remote_wildguard.sh" \
        --num-gpus "$NUM_GPUS" \
        --tensor-parallel-size "$TP_SIZE" \
        > "$LOG_DIR/reward_model.txt" 2>&1 &
    echo $! > "$LOG_DIR/wildguard.pid"

    echo "      Waiting for WildGuard to start (see $LOG_DIR/reward_model.txt)..."
    for _ in $(seq 1 120); do
        if grep -q "Uvicorn running" "$LOG_DIR/reward_model.txt" 2>/dev/null; then
            echo "      WildGuard server up; waiting for model weights to load..."
            sleep 30
            echo "      WildGuard is ready."
            break
        fi
        sleep 5
    done
fi

# Step 3: Training
if [[ -f "$LOG_DIR/train.pid" ]] && kill -0 "$(cat "$LOG_DIR/train.pid")" 2>/dev/null; then
    echo "[3/3] Training already running (pid $(cat "$LOG_DIR/train.pid"))."
else
    echo "[3/3] Starting training..."
    nohup bash "$SCRIPT_DIR/red_team_game_reinforce_8b.sh" \
        > "$LOG_DIR/train.txt" 2>&1 &
    echo $! > "$LOG_DIR/train.pid"
fi

echo
echo "All services launched. You can disconnect safely."
echo
echo "  tail -f $LOG_DIR/reward_model.txt"
echo "  tail -f $LOG_DIR/train.txt"
echo "  ray status"
echo
echo "PIDs:"
echo "  WildGuard: $(cat "$LOG_DIR/wildguard.pid" 2>/dev/null || echo n/a)"
echo "  Training:  $(cat "$LOG_DIR/train.pid" 2>/dev/null || echo n/a)"
