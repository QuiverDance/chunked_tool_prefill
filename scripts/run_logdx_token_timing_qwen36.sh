#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_LIST="${LOGDX_CASE_LIST:-${1:-}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-logdx_qwen36_token_timing}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
LOGDX_CONFIG="$ROOT_DIR/agent/src/minisweagent/config/benchmarks/logdx.yaml"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.6-27B}"
LOGDX_SPLITS="${LOGDX_SPLITS:-all}"
LOGDX_CORPUS_ROOT="${LOGDX_CORPUS_ROOT:-}"
LOGDX_SLICE_GPU0="${LOGDX_SLICE_GPU0:-0:18}"
LOGDX_SLICE_GPU1="${LOGDX_SLICE_GPU1:-18:}"

if [[ -n "$CASE_LIST" && ! -f "$CASE_LIST" ]]; then
  echo "LogDx case list not found: $CASE_LIST"
  exit 2
fi
if [[ ! -x "$MINI_EXTRA" ]]; then
  MINI_EXTRA="mini-extra"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "$RUN_DIR/gpu0" "$RUN_DIR/gpu1" "$REPORT_DIR"

"$PYTHON_BIN" - <<'PY'
import importlib.util

if importlib.util.find_spec("logdx_ci") is None:
    raise SystemExit("logdx-ci is missing. Install it with: pip install logdx-ci")
PY

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
    logdx
    --splits "$LOGDX_SPLITS"
    --slice "$slice_spec"
    --workers 1
    --output "$output_dir"
    --config "$LOGDX_CONFIG"
    --config "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}"
    --config "model.model_kwargs.api_base=http://127.0.0.1:${port}/v1"
    --config "agent.tokenizer_path=${TOKENIZER_PATH}"
    --tokenizer-path "$TOKENIZER_PATH"
  )

  if [[ -n "$CASE_LIST" ]]; then
    args+=(--case-list "$CASE_LIST")
  fi
  if [[ -n "$LOGDX_CORPUS_ROOT" ]]; then
    args+=(--corpus-root "$LOGDX_CORPUS_ROOT")
  fi

  "$MINI_EXTRA" "${args[@]}" > "$log_file" 2>&1
}

echo "run_dir=$RUN_DIR"
echo "report_dir=$REPORT_DIR"
echo "case_list=$CASE_LIST"
echo "splits=$LOGDX_SPLITS"
echo "corpus_root=$LOGDX_CORPUS_ROOT"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "tokenizer_path=$TOKENIZER_PATH"

echo "checking vLLM servers"
check_server 8000
check_server 8001

echo "starting gpu0 slice $LOGDX_SLICE_GPU0 on port 8000"
run_slice gpu0 8000 "$LOGDX_SLICE_GPU0" &
pid0="$!"

echo "starting gpu1 slice $LOGDX_SLICE_GPU1 on port 8001"
run_slice gpu1 8001 "$LOGDX_SLICE_GPU1" &
pid1="$!"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_token_timing.py" "$RUN_DIR" --output-dir "$REPORT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_logdx.py" "$RUN_DIR" --output-dir "$REPORT_DIR"

exit "$status"
