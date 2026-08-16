#!/usr/bin/env bash
#
# Set up /root/eval-venv for safety-eval without modifying the base training Python env.
# Uses system-site-packages so ROCm torch/vllm from /usr/local are inherited.
#
# Usage:
#   bash scripts/setup_eval_venv.sh
#   bash scripts/setup_eval_venv.sh --venv /path/to/eval-venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_VENV="${EVAL_VENV:-/root/eval-venv}"
SAFETY_EVAL_DIR="$REPO_ROOT/eval/benchmarks/safety-eval-fork"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            EVAL_VENV="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

VENV_PYTHON="$EVAL_VENV/bin/python"
VENV_PIP="$EVAL_VENV/bin/pip"

if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Creating venv at $EVAL_VENV with system-site-packages..."
    python3 -m venv --system-site-packages "$EVAL_VENV"
fi

if ! grep -q '^include-system-site-packages = true' "$EVAL_VENV/pyvenv.cfg"; then
    echo "ERROR: $EVAL_VENV must have include-system-site-packages = true" >&2
    echo "Recreate with: python3 -m venv --system-site-packages $EVAL_VENV" >&2
    exit 1
fi

echo "Verifying ROCm torch/vllm inheritance..."
"$VENV_PYTHON" -c "
import torch, vllm
print('  torch:', torch.__version__, 'at', torch.__file__)
print('  vllm:', vllm.__version__)
if 'dist-packages' not in torch.__file__:
    raise SystemExit('torch not from system site-packages — aborting to protect training env')
"

echo "Installing safety-eval-fork into $EVAL_VENV (never touching base pip)..."
cd "$SAFETY_EVAL_DIR"

"$VENV_PIP" install -U pip
"$VENV_PIP" install 'setuptools>=77,<80'

"$VENV_PIP" install -e . --no-deps

"$VENV_PIP" install \
    alpaca-eval fire fastchat fschat datasets pandas scikit-learn scipy \
    tqdm pyyaml openai tenacity peft termcolor regex joblib dill \
    huggingface-hub tokenizers accelerate cloudpickle psutil pydantic \
    requests sympy beaker-py beaker-gantry multiprocess

echo "Verifying safety-eval import..."
# HarmBench uses WildGuard, not OpenAI; dummy key satisfies import-time OpenAI client init.
PYTHONPATH="$SAFETY_EVAL_DIR" OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-not-used}" \
    "$VENV_PYTHON" -c "from evaluation.eval import generators; print('safety-eval import: ok')"

echo
echo "Setup complete."
echo "  Python: $VENV_PYTHON"
echo "  Run eval: bash $REPO_ROOT/scripts/run_safety_eval.sh"
