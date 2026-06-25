#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_LIST="${ARTIFACT_LIST:-${1:-}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-cibench_qwen36_token_timing}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
CIBENCH_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/cibench.yaml"
TOKEN_TIMING_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench_token_timing.yaml"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.6-27B}"
CIBENCH_SLICE_GPU0="${CIBENCH_SLICE_GPU0:-0:50}"
CIBENCH_SLICE_GPU1="${CIBENCH_SLICE_GPU1:-50:}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

if [[ -z "$ARTIFACT_LIST" ]]; then
  echo "Usage: ARTIFACT_LIST=artifacts.txt $0"
  echo "   or: $0 artifacts.txt"
  exit 2
fi
if [[ ! -f "$ARTIFACT_LIST" ]]; then
  echo "Artifact list not found: $ARTIFACT_LIST"
  exit 2
fi
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

run_slice() {
  local label="$1"
  local port="$2"
  local slice_spec="$3"
  local output_dir="$RUN_DIR/$label"
  local log_file="$output_dir/launcher.log"

  "$MINI_EXTRA" cibench \
    --artifact-list "$ARTIFACT_LIST" \
    --slice "$slice_spec" \
    --workers 1 \
    --output "$output_dir" \
    --config "$CIBENCH_CONFIG" \
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
echo "artifact_list=$ARTIFACT_LIST"
echo "docker_host=$DOCKER_HOST"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "checking Docker"
docker info >/dev/null

echo "checking vLLM servers"
check_server 8000
check_server 8001

echo "starting gpu0 slice $CIBENCH_SLICE_GPU0 on port 8000"
run_slice gpu0 8000 "$CIBENCH_SLICE_GPU0" &
pid0="$!"

echo "starting gpu1 slice $CIBENCH_SLICE_GPU1 on port 8001"
run_slice gpu1 8001 "$CIBENCH_SLICE_GPU1" &
pid1="$!"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_token_timing.py" "$RUN_DIR" --output-dir "$REPORT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_cibench.py" "$RUN_DIR" --output-dir "$REPORT_DIR"

exit "$status"
