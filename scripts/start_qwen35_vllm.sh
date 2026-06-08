#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runs/vllm"
VLLM_ENV="$ROOT_DIR/.conda/vllm-py312"
VLLM_BIN="$VLLM_ENV/bin/vllm"
MODEL_DIR="/home/pjw7200/models/Qwen3.5-27B"
SERVED_MODEL_NAME="qwen35-27b"
HOST="127.0.0.1"

mkdir -p "$LOG_DIR"

start_server() {
  local gpu="$1"
  local port="$2"
  local name="qwen35-27b-gpu${gpu}"
  local pid_file="$LOG_DIR/${name}.pid"
  local log_file="$LOG_DIR/${name}.log"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name is already running with pid $(cat "$pid_file")"
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_PREFIX="$VLLM_ENV" \
  CONDA_DEFAULT_ENV="$VLLM_ENV" \
  PATH="$VLLM_ENV/bin:$PATH" \
  PYTHONNOUSERSITE=1 \
  setsid "$VLLM_BIN" serve "$MODEL_DIR" \
    --host "$HOST" \
    --port "$port" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    > "$log_file" 2>&1 < /dev/null &

  echo "$!" > "$pid_file"
  echo "started $name on $HOST:$port with pid $(cat "$pid_file")"
}

start_server 0 8000
start_server 1 8001
