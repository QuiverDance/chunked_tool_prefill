#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${RUN_DIR:-}}"
SWEBENCH_PRO_HOME="${SWEBENCH_PRO_HOME:-$HOME/.cache/swebench-pro-os}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
RAW_SAMPLE_PATH="${RAW_SAMPLE_PATH:-}"
PATCH_PATH="${PATCH_PATH:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-jefzda}"
REPORT_DIR="${REPORT_DIR:-}"
SUMMARY_RUN_DIR="${SUMMARY_RUN_DIR:-}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: $0 RUN_DIR"
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -f "$SWEBENCH_PRO_HOME/swe_bench_pro_eval.py" ]]; then
  echo "SWE-bench Pro official repo not found at $SWEBENCH_PRO_HOME"
  echo "Run scripts/setup_swebench_pro.sh first, or set SWEBENCH_PRO_HOME."
  exit 2
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RAW_SAMPLE_PATH="${RAW_SAMPLE_PATH:-$RUN_DIR/swebench_pro_raw.jsonl}"
PATCH_PATH="${PATCH_PATH:-$RUN_DIR/swebench_pro_patches.json}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$RUN_DIR/swebench_pro_eval}"
SUMMARY_RUN_DIR="${SUMMARY_RUN_DIR:-$RUN_DIR}"
REPORT_DIR="${REPORT_DIR:-$SUMMARY_RUN_DIR/reports}"

if [[ ! -f "$RAW_SAMPLE_PATH" ]]; then
  echo "Raw sample file not found: $RAW_SAMPLE_PATH"
  exit 2
fi
if [[ ! -f "$PATCH_PATH" ]]; then
  echo "Patch JSON file not found: $PATCH_PATH"
  exit 2
fi

mkdir -p "$EVAL_OUTPUT_DIR" "$REPORT_DIR"

"$PYTHON_BIN" "$SWEBENCH_PRO_HOME/swe_bench_pro_eval.py" \
  --raw_sample_path "$RAW_SAMPLE_PATH" \
  --patch_path "$PATCH_PATH" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --scripts_dir "$SWEBENCH_PRO_HOME/run_scripts" \
  --num_workers "$EVAL_WORKERS" \
  --dockerhub_username "$DOCKERHUB_USERNAME" \
  --use_local_docker

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_swebench_pro.py" "$SUMMARY_RUN_DIR" --output-dir "$REPORT_DIR"
