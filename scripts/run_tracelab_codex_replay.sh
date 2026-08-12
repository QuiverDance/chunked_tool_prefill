#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_SOURCE="${TRACE_SOURCE:-/home/pjw7200/traces/tracelab/raw/release-v0.0.1/syfi_coding_trace.jsonl.gz}"
PYTHON="${PYTHON:-/data/pjw7200/src/mini-swe-agent/.venv/bin/python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistral-small32-24b}"
RUN_NAME="${RUN_NAME:-tracelab_codex_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/replay_runs/$RUN_NAME}"
MANIFEST="${MANIFEST:-$OUTPUT_DIR/prepared_workload.json}"
LIMIT="${LIMIT:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-tracelab-codex-v1}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-131072}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-}"
TIME_SCALE="${TIME_SCALE:-1.0}"
WARMUP_TRIALS="${WARMUP_TRIALS:-1}"
MAX_TRIAL_ATTEMPTS="${MAX_TRIAL_ATTEMPTS:-3}"
TIMEOUT="${TIMEOUT:-600}"
CROSSOVER="${CROSSOVER:-1}"
RESUME="${RESUME:-0}"
RESUME_LABEL="${RESUME_LABEL:-}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"
PRE_MEASUREMENT_IDLE_S="${PRE_MEASUREMENT_IDLE_S:-5}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TRACE_SOURCE" ]]; then
  echo "TraceLab source not found: $TRACE_SOURCE" >&2
  exit 1
fi
if [[ -n "$RESUME_LABEL" && ! "$RESUME_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "RESUME_LABEL must contain only letters, numbers, underscores, or hyphens" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$REBUILD_MANIFEST" == "1" || ! -f "$MANIFEST" ]]; then
  prepare_args=(
    "$TRACE_SOURCE"
    --output "$MANIFEST"
    --sample-seed "$SAMPLE_SEED"
    --max-context-tokens "$MAX_CONTEXT_TOKENS"
    --cache-block-tokens 16
  )
  if [[ "$LIMIT" == "all" ]]; then
    prepare_args+=(--all)
  else
    prepare_args+=(--limit "$LIMIT")
  fi
  if [[ -n "$MAX_COMPLETION_TOKENS" ]]; then
    prepare_args+=(--max-completion-tokens "$MAX_COMPLETION_TOKENS")
  fi
  PYTHONPATH="$ROOT_DIR/agent/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m minisweagent.run.extra.tracelab_replay prepare "${prepare_args[@]}"
fi

for port in 8000 8001; do
  if ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "vLLM is not healthy on port $port" >&2
    exit 1
  fi
  if ! curl -fsS -X POST "http://127.0.0.1:${port}/v1/prefill/reset" >/dev/null; then
    echo "Prefix-cache reset is not ready on port $port" >&2
    exit 1
  fi
done

# Trace scanning, hashing, sampling, and server preflight are complete before
# either measured process starts. This idle interval prevents preparation CPU
# and disk activity from overlapping the first trial.
sleep "$PRE_MEASUREMENT_IDLE_S"

run_replay() {
  local algorithm="$1"
  local port="$2"
  local output="$3"
  local sync_directory="$4"
  local participant="$5"
  local peer="$6"
  local api_base="http://127.0.0.1:${port}/v1"
  local run_args=(
    "$MANIFEST"
    --output "$output"
    --algorithm "$algorithm"
    --api-base "$api_base"
    --prefill-url "$api_base/prefill"
    --model-name "$SERVED_MODEL_NAME"
    --time-scale "$TIME_SCALE"
    --warmup-trials "$WARMUP_TRIALS"
    --max-trial-attempts "$MAX_TRIAL_ATTEMPTS"
    --sync-directory "$sync_directory"
    --participant "$participant"
    --peer "$peer"
    --timeout "$TIMEOUT"
  )
  if [[ "$RESUME" == "1" ]]; then
    run_args+=(--resume)
  fi

  PYTHONPATH="$ROOT_DIR/agent/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m minisweagent.run.extra.tracelab_replay run \
    "${run_args[@]}"
}

run_pair() {
  local phase="$1"
  local gpu0_algorithm="$2"
  local gpu1_algorithm="$3"
  local gpu0_output="$OUTPUT_DIR/${phase}_${gpu0_algorithm}_gpu0"
  local gpu1_output="$OUTPUT_DIR/${phase}_${gpu1_algorithm}_gpu1"
  local suffix="${RESUME_LABEL:+_$RESUME_LABEL}"
  local sync_directory="$OUTPUT_DIR/${phase}_measurement_sync${suffix}"
  local gpu0_log="$gpu0_output${suffix}.log"
  local gpu1_log="$gpu1_output${suffix}.log"

  if [[ -e "$sync_directory" ]]; then
    echo "Measurement sync directory already exists: $sync_directory" >&2
    return 1
  fi
  mkdir -p "$sync_directory"

  run_replay "$gpu0_algorithm" 8000 "$gpu0_output" "$sync_directory" gpu0 gpu1 >"$gpu0_log" 2>&1 &
  local gpu0_pid=$!
  run_replay "$gpu1_algorithm" 8001 "$gpu1_output" "$sync_directory" gpu1 gpu0 >"$gpu1_log" 2>&1 &
  local gpu1_pid=$!

  local phase_status=0
  wait "$gpu0_pid" || phase_status=1
  wait "$gpu1_pid" || phase_status=1
  if [[ "$phase_status" != "0" ]]; then
    echo "TraceLab replay failed in $phase; inspect $gpu0_log and $gpu1_log" >&2
    return "$phase_status"
  fi
}

run_pair phase_a incremental baseline

if [[ "$CROSSOVER" == "1" ]]; then
  for port in 8000 8001; do
    curl -fsS -X POST "http://127.0.0.1:${port}/v1/prefill/reset" >/dev/null
  done
  sleep "$PRE_MEASUREMENT_IDLE_S"
  run_pair phase_b baseline incremental
fi

echo "prepared workload: $MANIFEST"
echo "results: $OUTPUT_DIR"
