#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${TRACE_PATH:-/home/pjw7200/traces/swe-chat}"
PYTHON="${PYTHON:-/data/pjw7200/src/mini-swe-agent/.venv/bin/python}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-small32-24b}"
RUN_NAME="${RUN_NAME:-swe_chat_codex_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/replay_runs/$RUN_NAME}"
LIMIT="${LIMIT:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

for port in 8000 8001; do
  if ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "vLLM is not healthy on port $port" >&2
    exit 1
  fi
done

if ! curl -fsS \
  -X POST http://127.0.0.1:8000/v1/prefill \
  -H "Content-Type: application/json" \
  --data "{\"model\":\"${SERVED_MODEL_NAME}\",\"input_token_ids\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"cache_salt\":\"swe-chat-codex-preflight\"}" \
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
    --swe-chat-format codex-jsonl \
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

run_replay incremental 8000 "$OUTPUT_DIR/incremental_gpu0" >"$OUTPUT_DIR/incremental_gpu0.log" 2>&1 &
incremental_pid=$!
run_replay baseline 8001 "$OUTPUT_DIR/baseline_gpu1" >"$OUTPUT_DIR/baseline_gpu1.log" 2>&1 &
baseline_pid=$!

status=0
wait "$incremental_pid" || status=1
wait "$baseline_pid" || status=1

echo "incremental log: $OUTPUT_DIR/incremental_gpu0.log"
echo "baseline log: $OUTPUT_DIR/baseline_gpu1.log"
echo "results: $OUTPUT_DIR"
exit "$status"
