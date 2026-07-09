#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swebench_verified_mistral_small32_token_timing}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-small32-24b}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506}"

exec "$ROOT_DIR/scripts/run_verified_token_timing.sh"
