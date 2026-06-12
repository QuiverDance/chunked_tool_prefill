#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swebench_verified_qwen36_token_timing}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.6-27B}"

exec "$ROOT_DIR/scripts/run_verified_token_timing.sh"
