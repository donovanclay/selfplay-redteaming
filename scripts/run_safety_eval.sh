#!/usr/bin/env bash
#
# Run safety-eval-fork benchmarks using /root/eval-venv (isolated from training env).
#
# Usage:
#   bash scripts/run_safety_eval.sh
#   bash scripts/run_safety_eval.sh --model_path mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
#   bash scripts/run_safety_eval.sh --smoke-only
#   bash scripts/run_safety_eval.sh --skip-smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_VENV="${EVAL_VENV:-/root/eval-venv}"
SAFETY_EVAL_DIR="$REPO_ROOT/eval/benchmarks/safety-eval-fork"

DEFAULT_MODEL_PATH="$REPO_ROOT/checkpoints/selfplay_RL_FULL_PTX_SFT_wjbhs_re++_rtg_0805T22:08/ckpt/global_step200_hf"
DEFAULT_BASE_MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
MODEL_PATH="$DEFAULT_MODEL_PATH"
CHAT_TEMPLATE="llama3_cot"
OUTPUT_DIR=""
MIN_GPUS_PER_TASK=1
GPUS=""
SMOKE_ONLY=false
SKIP_SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
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
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

VENV_PYTHON="$EVAL_VENV/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: eval venv not found at $EVAL_VENV. Run: bash scripts/setup_eval_venv.sh" >&2
    exit 1
fi

if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: local model path not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/eval/results/safety_eval_$(date +%Y%m%dT%H%M)"
fi
mkdir -p "$OUTPUT_DIR"

if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

export PYTHONPATH="$SAFETY_EVAL_DIR${PYTHONPATH:+:$PYTHONPATH}"
# WildGuard benchmarks do not call OpenAI; dummy key satisfies safety-eval import chain.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-not-used}"

METRICS_PATH="$OUTPUT_DIR/metrics.json"
ALL_RESULTS_PATH="$OUTPUT_DIR/all.json"
SMOKE_METRICS="$OUTPUT_DIR/smoke_harmbench.json"
SMOKE_ALL="$OUTPUT_DIR/smoke_harmbench_all.json"

echo "Model:       $MODEL_PATH"
echo "Template:    $CHAT_TEMPLATE"
echo "Output:      $OUTPUT_DIR"
echo "Eval python: $VENV_PYTHON"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-all visible}"
echo

cd "$SAFETY_EVAL_DIR"

run_smoke() {
    echo "=== HarmBench smoke test ==="
    "$VENV_PYTHON" evaluation/eval.py generators \
        --model_name_or_path "$MODEL_PATH" \
        --model_input_template_path_or_name "$CHAT_TEMPLATE" \
        --tasks harmbench \
        --report_output_path "$SMOKE_METRICS" \
        --save_individual_results_path "$SMOKE_ALL" \
        --use_vllm
    echo "Smoke test passed: $SMOKE_METRICS"
}

run_full_suite() {
    echo "=== Full safety benchmark suite ==="
    "$VENV_PYTHON" evaluation/run_all_generation_benchmarks.py \
        --model_name_or_path "$MODEL_PATH" \
        --model_input_template_path_or_name "$CHAT_TEMPLATE" \
        --report_output_path "$METRICS_PATH" \
        --save_individual_results_path "$ALL_RESULTS_PATH" \
        --min_gpus_per_task "$MIN_GPUS_PER_TASK"
}

if [[ "$SMOKE_ONLY" == true ]]; then
    run_smoke
    exit 0
fi

if [[ "$SKIP_SMOKE" != true ]]; then
    run_smoke
fi

if [[ "$SMOKE_ONLY" != true ]]; then
    run_full_suite
fi

if [[ -f "$METRICS_PATH" ]]; then
  echo "=== Parsing results to CSV ==="
  "$VENV_PYTHON" "$REPO_ROOT/eval/parse_result_to_csv.py" "$OUTPUT_DIR" || true
fi

echo
echo "Done. Results in $OUTPUT_DIR"
