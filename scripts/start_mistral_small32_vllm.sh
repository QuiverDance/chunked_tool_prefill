#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runs/vllm"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.conda/vllm-py312}"
if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  VLLM_ENV="/data/pjw7200/src/mini-swe-agent/.venv"
fi
VLLM_PYTHON="$VLLM_ENV/bin/python"
MODEL_DIR="/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506"
SERVED_MODEL_NAME="mistral-small32-24b"
HOST="127.0.0.1"
MAX_MODEL_LEN="131072"

mkdir -p "$LOG_DIR"

start_server() {
  local gpu="$1"
  local port="$2"
  local name="mistral-small32-24b-gpu${gpu}"
  local pid_file="$LOG_DIR/${name}.pid"
  local log_file="$LOG_DIR/${name}.log"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name is already running with pid $(cat "$pid_file")"
    return
  fi

  if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
    echo "${HOST}:${port} is already serving another process. Stop it before starting $name."
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_PREFIX="$VLLM_ENV" \
  CONDA_DEFAULT_ENV="$VLLM_ENV" \
  PATH="$VLLM_ENV/bin:$PATH" \
  PYTHONPATH="$ROOT_DIR/scripts:$ROOT_DIR/agent/src${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONNOUSERSITE=1 \
  setsid "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
    --host "$HOST" \
    --port "$port" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --tokenizer-mode mistral \
    --load-format auto \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --language-model-only \
    --middleware vllm_prefill_middleware.prefill_middleware \
    > "$log_file" 2>&1 < /dev/null &

  echo "$!" > "$pid_file"
  echo "started $name on $HOST:$port with pid $(cat "$pid_file")"
}

start_server 0 8000
start_server 1 8001
