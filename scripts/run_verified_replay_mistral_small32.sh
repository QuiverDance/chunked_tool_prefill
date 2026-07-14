#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${TRACE_PATH:-${1:-$ROOT_DIR/traces/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z}}"
ALGORITHM="${ALGORITHM:-${2:-baseline}}"
PORT="${PORT:-8000}"
RUN_NAME="${RUN_NAME:-replay_mistral_small32_${ALGORITHM}_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/$RUN_NAME}"
MINI_EXTRA="${MINI_EXTRA:-$ROOT_DIR/.conda/miniswe-py311/bin/mini-extra}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-small32-24b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506}"
API_BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-131072}"
CACHE_BLOCK_TOKENS="${CACHE_BLOCK_TOKENS:-16}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-4}"

if [[ ! -x "$MINI_EXTRA" ]]; then
  MINI_EXTRA="mini-extra"
fi

echo "trace_path=$TRACE_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "algorithm=$ALGORITHM"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "api_base=$API_BASE"
echo "max_context_tokens=$MAX_CONTEXT_TOKENS"
echo "cache_block_tokens=$CACHE_BLOCK_TOKENS"
echo "candidate_top_k=$CANDIDATE_TOP_K"

exec "$MINI_EXTRA" replay "$TRACE_PATH" \
  --output "$OUTPUT_DIR" \
  --algorithm "$ALGORITHM" \
  -c benchmarks/swebench_replay_output_first.yaml \
  -c "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}" \
  -c "model.model_kwargs.api_base=${API_BASE}" \
  -c "agent.tokenizer_path=${TOKENIZER_PATH}" \
  -c "agent.tokenizer_local_files_only=true" \
  -c "replay.served_model_name=${SERVED_MODEL_NAME}" \
  -c "replay.api_base=${API_BASE}" \
  -c "replay.prefill_url=${API_BASE}/prefill" \
  -c "replay.max_context_tokens=${MAX_CONTEXT_TOKENS}" \
  -c "replay.cache_block_tokens=${CACHE_BLOCK_TOKENS}" \
  -c "replay.candidate_prefill.top_k=${CANDIDATE_TOP_K}"
