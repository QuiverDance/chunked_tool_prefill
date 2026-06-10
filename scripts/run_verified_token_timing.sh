#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-swebench_verified_qwen35_token_timing_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
BASE_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench.yaml"
TOKEN_TIMING_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench_token_timing.yaml"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
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
    --config "model.model_kwargs.api_base=http://127.0.0.1:${port}/v1" \
    --config "run.remove_docker_image_after_instance=true" \
    > "$log_file" 2>&1
}

echo "run_dir=$RUN_DIR"
echo "report_dir=$REPORT_DIR"
echo "docker_host=$DOCKER_HOST"
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
