#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/runs/vllm"

stop_server() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name has no pid file"
    return
  fi

  local pid
  pid="$(cat "$pid_file")"

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$name is not running"
    return
  fi

  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid"
  echo "stopped $name with pid $pid"
}

stop_server qwen35-27b-gpu0
stop_server qwen35-27b-gpu1

