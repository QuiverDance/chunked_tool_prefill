#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE_LIST="${SWEBENCH_PRO_INSTANCE_LIST:-${1:-}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-swebench_pro_qwen36_token_timing}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
SWEBENCH_PRO_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench_pro.yaml"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.6-27B}"
SWEBENCH_PRO_SLICE_GPU0="${SWEBENCH_PRO_SLICE_GPU0:-0:366}"
SWEBENCH_PRO_SLICE_GPU1="${SWEBENCH_PRO_SLICE_GPU1:-366:731}"
SWEBENCH_PRO_FILTER="${SWEBENCH_PRO_FILTER:-}"
SWEBENCH_PRO_REPO_FILTER="${SWEBENCH_PRO_REPO_FILTER:-}"
SWEBENCH_PRO_LANGUAGE_FILTER="${SWEBENCH_PRO_LANGUAGE_FILTER:-}"
SWEBENCH_PRO_REMOVE_IMAGES="${SWEBENCH_PRO_REMOVE_IMAGES:-true}"
SWEBENCH_PRO_EVALUATE="${SWEBENCH_PRO_EVALUATE:-0}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

if [[ -n "$INSTANCE_LIST" && ! -f "$INSTANCE_LIST" ]]; then
  echo "SWE-bench Pro instance list not found: $INSTANCE_LIST"
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

  args=(
    swebench-pro
    --slice "$slice_spec"
    --workers 1
    --output "$output_dir"
    --config "$SWEBENCH_PRO_CONFIG"
    --config "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}"
    --config "model.model_kwargs.api_base=http://127.0.0.1:${port}/v1"
    --config "agent.tokenizer_path=${TOKENIZER_PATH}"
    --config "environment.pull_timeout=900"
    --config "environment.start_attempts=3"
    --config "environment.start_retry_sleep=10"
    --config "run.remove_docker_image_after_instance=${SWEBENCH_PRO_REMOVE_IMAGES}"
    --tokenizer-path "$TOKENIZER_PATH"
  )

  if [[ -n "$INSTANCE_LIST" ]]; then
    args+=(--instance-list "$INSTANCE_LIST")
  fi
  if [[ -n "$SWEBENCH_PRO_FILTER" ]]; then
    args+=(--filter "$SWEBENCH_PRO_FILTER")
  fi
  if [[ -n "$SWEBENCH_PRO_REPO_FILTER" ]]; then
    args+=(--repo-filter "$SWEBENCH_PRO_REPO_FILTER")
  fi
  if [[ -n "$SWEBENCH_PRO_LANGUAGE_FILTER" ]]; then
    args+=(--language-filter "$SWEBENCH_PRO_LANGUAGE_FILTER")
  fi

  "$MINI_EXTRA" "${args[@]}" > "$log_file" 2>&1
}

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

echo "run_dir=$RUN_DIR"
echo "report_dir=$REPORT_DIR"
echo "instance_list=$INSTANCE_LIST"
echo "docker_host=$DOCKER_HOST"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "slice_gpu0=$SWEBENCH_PRO_SLICE_GPU0"
echo "slice_gpu1=$SWEBENCH_PRO_SLICE_GPU1"
echo "evaluate=$SWEBENCH_PRO_EVALUATE"

echo "checking Docker"
docker info >/dev/null

echo "checking vLLM servers"
check_server 8000
check_server 8001

echo "starting gpu0 slice $SWEBENCH_PRO_SLICE_GPU0 on port 8000"
run_slice gpu0 8000 "$SWEBENCH_PRO_SLICE_GPU0" &
pid0="$!"

echo "starting gpu1 slice $SWEBENCH_PRO_SLICE_GPU1 on port 8001"
run_slice gpu1 8001 "$SWEBENCH_PRO_SLICE_GPU1" &
pid1="$!"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_token_timing.py" "$RUN_DIR" --output-dir "$REPORT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_swebench_pro.py" "$RUN_DIR" --output-dir "$REPORT_DIR"

if truthy "$SWEBENCH_PRO_EVALUATE"; then
  EVAL_WORKERS="$EVAL_WORKERS" REPORT_DIR="$REPORT_DIR" SUMMARY_RUN_DIR="$RUN_DIR" \
    "$ROOT_DIR/scripts/evaluate_swebench_pro_local.sh" "$RUN_DIR/gpu0"
  EVAL_WORKERS="$EVAL_WORKERS" REPORT_DIR="$REPORT_DIR" SUMMARY_RUN_DIR="$RUN_DIR" \
    "$ROOT_DIR/scripts/evaluate_swebench_pro_local.sh" "$RUN_DIR/gpu1"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_swebench_pro.py" "$RUN_DIR" --output-dir "$REPORT_DIR"
fi

exit "$status"
