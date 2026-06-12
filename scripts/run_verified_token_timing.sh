#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swebench_verified_qwen35_token_timing}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
BASE_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench.yaml"
TOKEN_TIMING_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench_token_timing.yaml"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-27b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.5-27B}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

if [[ ! -x "$MINI_EXTRA" ]]; then
  MINI_EXTRA="mini-extra"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "$RUN_DIR/gpu0" "$RUN_DIR/gpu1" "$REPORT_DIR"

check_server() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
  "$PYTHON_BIN" - "$port" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys
import urllib.request

port, expected = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
    model_ids = [item["id"] for item in json.load(response)["data"]]

if expected not in model_ids:
    raise SystemExit(f"expected model {expected!r} on port {port}, got {model_ids!r}")
PY
}

run_half() {
  local label="$1"
  local port="$2"
  local slice_spec="$3"
  local output_dir="$RUN_DIR/$label"
  local log_file="$output_dir/launcher.log"

  "$MINI_EXTRA" swebench \
    --subset verified \
    --split test \
    --slice "$slice_spec" \
    --workers 1 \
    --output "$output_dir" \
    --config "$BASE_CONFIG" \
    --config "$TOKEN_TIMING_CONFIG" \
    --config "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}" \
    --config "model.model_kwargs.api_base=http://127.0.0.1:${port}/v1" \
    --config "agent.tokenizer_path=${TOKENIZER_PATH}" \
    --config "environment.pull_timeout=600" \
    --config "environment.start_attempts=3" \
    --config "environment.start_retry_sleep=10" \
    --config "run.remove_docker_image_after_instance=true" \
    > "$log_file" 2>&1
}

echo "run_dir=$RUN_DIR"
echo "report_dir=$REPORT_DIR"
echo "docker_host=$DOCKER_HOST"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "checking Docker"
docker info >/dev/null

echo "checking vLLM servers"
check_server 8000
check_server 8001

echo "starting gpu0 slice 0:250 on port 8000"
run_half gpu0 8000 "0:250" &
pid0="$!"

echo "starting gpu1 slice 250: on port 8001"
run_half gpu1 8001 "250:" &
pid1="$!"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_token_timing.py" "$RUN_DIR" --output-dir "$REPORT_DIR"

exit "$status"
