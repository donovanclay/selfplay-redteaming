#!/usr/bin/env bash
#
# Evaluate base (pretrain) and trained checkpoints, then print a comparison table.
#
# Usage:
#   bash scripts/run_safety_eval_comparison.sh
#   bash scripts/run_safety_eval_comparison.sh --smoke-only
#   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/run_safety_eval_comparison.sh --gpus 1,2,3,4,5,6,7 --skip-smoke
#
# Options mirror run_safety_eval.sh plus:
#   --base_model PATH_OR_HF_ID   (default: mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated)
#   --trained_model PATH         (default: global_step200_hf checkpoint)
#   --output_dir DIR             (default: eval/results/compare_TIMESTAMP)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_BASE_MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
DEFAULT_TRAINED_MODEL="$REPO_ROOT/checkpoints/selfplay_RL_FULL_PTX_SFT_wjbhs_re++_rtg_0805T22:08/ckpt/global_step200_hf"

BASE_MODEL="$DEFAULT_BASE_MODEL"
TRAINED_MODEL="$DEFAULT_TRAINED_MODEL"
CHAT_TEMPLATE="llama3_cot"
OUTPUT_DIR=""
MIN_GPUS_PER_TASK=1
GPUS=""
SMOKE_ONLY=false
SKIP_SMOKE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --trained_model)
            TRAINED_MODEL="$2"
            shift 2
            ;;
        --chat_template)
            CHAT_TEMPLATE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --min_gpus_per_task)
            MIN_GPUS_PER_TASK="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --smoke-only)
            SMOKE_ONLY=true
            shift
            ;;
        --skip-smoke)
            SKIP_SMOKE=true
            shift
            ;;
        -h|--help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/eval/results/compare_$(date +%Y%m%dT%H%M)"
fi
BASE_OUT="$OUTPUT_DIR/base"
TRAINED_OUT="$OUTPUT_DIR/trained"
mkdir -p "$OUTPUT_DIR"

RUN_ARGS=(--chat_template "$CHAT_TEMPLATE" --min_gpus_per_task "$MIN_GPUS_PER_TASK")
if [[ -n "$GPUS" ]]; then
    RUN_ARGS+=(--gpus "$GPUS")
fi
if [[ "$SMOKE_ONLY" == true ]]; then
    RUN_ARGS+=(--smoke-only)
fi
if [[ "$SKIP_SMOKE" == true ]]; then
    RUN_ARGS+=(--skip-smoke)
fi

echo "Comparison output: $OUTPUT_DIR"
echo "  Base model:    $BASE_MODEL"
echo "  Trained model: $TRAINED_MODEL"
echo

echo "========== [1/2] Base model =========="
bash "$SCRIPT_DIR/run_safety_eval.sh" \
    --model_path "$BASE_MODEL" \
    --output_dir "$BASE_OUT" \
    "${RUN_ARGS[@]}"

echo
echo "========== [2/2] Trained model =========="
bash "$SCRIPT_DIR/run_safety_eval.sh" \
    --model_path "$TRAINED_MODEL" \
    --output_dir "$TRAINED_OUT" \
    "${RUN_ARGS[@]}"

BASE_METRICS="$BASE_OUT/smoke_harmbench.json"
TRAINED_METRICS="$TRAINED_OUT/smoke_harmbench.json"
if [[ "$SMOKE_ONLY" != true ]]; then
    BASE_METRICS="$BASE_OUT/metrics.json"
    TRAINED_METRICS="$TRAINED_OUT/metrics.json"
fi

echo
echo "========== Comparison =========="
if [[ ! -f "$BASE_METRICS" || ! -f "$TRAINED_METRICS" ]]; then
    echo "ERROR: expected metrics not found:" >&2
    echo "  $BASE_METRICS" >&2
    echo "  $TRAINED_METRICS" >&2
    exit 1
fi

/root/eval-venv/bin/python "$SCRIPT_DIR/compare_safety_metrics.py" \
    "$BASE_METRICS" "$TRAINED_METRICS" | tee "$OUTPUT_DIR/comparison.txt"

echo
echo "Saved comparison to $OUTPUT_DIR/comparison.txt"
echo "  Base results:    $BASE_OUT"
echo "  Trained results: $TRAINED_OUT"
