#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${TRACE_PATH:-/home/pjw7200/traces/swe-chat}"
PYTHON="${PYTHON:-/data/pjw7200/src/mini-swe-agent/.venv/bin/python}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-small32-24b}"
RUN_NAME="${RUN_NAME:-swe_chat_opencode_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/replay_runs/$RUN_NAME}"
LIMIT="${LIMIT:-}"
INCREMENTAL_PORT="${INCREMENTAL_PORT:-8000}"
BASELINE_PORT="${BASELINE_PORT:-8001}"
INCREMENTAL_GPU_LABEL="${INCREMENTAL_GPU_LABEL:-gpu0}"
BASELINE_GPU_LABEL="${BASELINE_GPU_LABEL:-gpu1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

for port in "$INCREMENTAL_PORT" "$BASELINE_PORT"; do
  if ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "vLLM is not healthy on port $port" >&2
    exit 1
  fi
done

if ! curl -fsS \
  -X POST "http://127.0.0.1:${INCREMENTAL_PORT}/v1/prefill" \
  -H "Content-Type: application/json" \
  --data "{\"model\":\"${SERVED_MODEL_NAME}\",\"input_token_ids\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"cache_salt\":\"swe-chat-preflight\"}" \
  >/dev/null; then
  echo "Prefill endpoint is not ready on port 8000" >&2
  exit 1
fi

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

run_replay() {
  local algorithm="$1"
  local port="$2"
  local output="$3"
  local api_base="http://127.0.0.1:${port}/v1"

  PYTHONPATH="$ROOT_DIR/agent/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m minisweagent.run.extra.incremental_replay \
    "$TRACE_PATH" \
    --output "$output" \
    --algorithm "$algorithm" \
    "${limit_args[@]}" \
    -c "$ROOT_DIR/agent/src/minisweagent/config/benchmarks/swebench_replay_output_first.yaml" \
    -c "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}" \
    -c "model.model_kwargs.api_base=${api_base}" \
    -c "agent.tokenizer_path=${TOKENIZER_PATH}" \
    -c "agent.tokenizer_local_files_only=true" \
    -c "replay.served_model_name=${SERVED_MODEL_NAME}" \
    -c "replay.api_base=${api_base}" \
    -c "replay.prefill_url=${api_base}/prefill" \
    -c "replay.max_context_tokens=131072" \
    -c "replay.cache_block_tokens=16" \
    -c "replay.time_scale=1.0" \
    -c "replay.timeout=600"
}

mkdir -p "$OUTPUT_DIR"

incremental_output="$OUTPUT_DIR/incremental_${INCREMENTAL_GPU_LABEL}"
baseline_output="$OUTPUT_DIR/baseline_${BASELINE_GPU_LABEL}"

run_replay incremental "$INCREMENTAL_PORT" "$incremental_output" >"$incremental_output.log" 2>&1 &
incremental_pid=$!
run_replay baseline "$BASELINE_PORT" "$baseline_output" >"$baseline_output.log" 2>&1 &
baseline_pid=$!

status=0
wait "$incremental_pid" || status=1
wait "$baseline_pid" || status=1

echo "incremental log: $incremental_output.log"
echo "baseline log: $baseline_output.log"
echo "results: $OUTPUT_DIR"
exit "$status"
