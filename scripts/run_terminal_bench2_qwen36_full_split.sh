#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-terminal_bench2_qwen36_full_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT_DIR/runs/$RUN_NAME"
HARBOR_DIR="$RUN_DIR/harbor"
TRAJECTORIES_DIR="$RUN_DIR/trajectories"
REPORT_DIR="$ROOT_DIR/reports/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/harbor-py312/bin/python}"
HARBOR_BIN="${HARBOR_BIN:-$ROOT_DIR/.conda/harbor-py312/bin/harbor}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
TB2_MODEL_NAME="${TB2_MODEL_NAME:-hosted_vllm/${SERVED_MODEL_NAME}}"
TB2_TOKENIZER_PATH="${TB2_TOKENIZER_PATH:-/home/pjw7200/models/Qwen3.6-27B}"
TB2_STEP_LIMIT="${TB2_STEP_LIMIT:-0}"
TB2_WALL_TIME_LIMIT_SECONDS="${TB2_WALL_TIME_LIMIT_SECONDS:-10800}"
TB2_AGENT_TIMEOUT_SECONDS="${TB2_AGENT_TIMEOUT_SECONDS:-10800}"
TB2_MAX_TOKENS="${TB2_MAX_TOKENS:-81920}"
TB2_DATASET="${TB2_DATASET:-terminal-bench@2.0}"
TB2_AGENT_IMPORT_PATH="${TB2_AGENT_IMPORT_PATH:-minisweagent.run.benchmarks.terminal_bench2_harbor_agent:MiniSweTokenTimingHarborAgent}"
TB2_CONFIG_FILE="${TB2_CONFIG_FILE:-$ROOT_DIR/agent/src/minisweagent/config/benchmarks/terminal_bench2_token_timing.yaml}"
TB2_DISABLE_VERIFICATION="${TB2_DISABLE_VERIFICATION:-1}"
export DOCKER_HOST="${DOCKER_HOST:-tcp://127.0.0.1:2375}"

mkdir -p "$HARBOR_DIR" "$TRAJECTORIES_DIR" "$REPORT_DIR" "$LOG_DIR"

check_server() {
  local port="$1"
  local api_base="http://127.0.0.1:${port}/v1"
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
  "$PYTHON_BIN" - "$api_base" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys
import urllib.request

api_base, expected = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=5) as response:
    model = json.load(response)["data"][0]

if model["id"] != expected:
    raise SystemExit(f"expected model {expected!r} at {api_base!r}, got {model['id']!r}")
if model.get("max_model_len") != 262144:
    raise SystemExit(f"expected max_model_len 262144 at {api_base!r}, got {model.get('max_model_len')!r}")
PY
}

echo "run_dir=$RUN_DIR"
echo "harbor_dir=$HARBOR_DIR"
echo "trajectories_dir=$TRAJECTORIES_DIR"
echo "report_dir=$REPORT_DIR"
echo "dataset=$TB2_DATASET"
echo "model=$TB2_MODEL_NAME"
echo "tokenizer_path=$TB2_TOKENIZER_PATH"
echo "step_limit=$TB2_STEP_LIMIT"
echo "wall_time_limit_seconds=$TB2_WALL_TIME_LIMIT_SECONDS"
echo "agent_timeout_seconds=$TB2_AGENT_TIMEOUT_SECONDS"
echo "max_tokens=$TB2_MAX_TOKENS"
echo "disable_verification=$TB2_DISABLE_VERIFICATION"
echo "docker_host=$DOCKER_HOST"

check_server 8000
check_server 8001

export RUN_NAME HARBOR_DIR TRAJECTORIES_DIR TB2_DATASET TB2_AGENT_IMPORT_PATH TB2_MODEL_NAME
export TB2_TOKENIZER_PATH TB2_STEP_LIMIT TB2_WALL_TIME_LIMIT_SECONDS
export TB2_AGENT_TIMEOUT_SECONDS TB2_MAX_TOKENS TB2_CONFIG_FILE TB2_DISABLE_VERIFICATION
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import asyncio
import json
import os
import sys
from pathlib import Path

from harbor.models.job.config import DatasetConfig, JobConfig


async def task_names() -> list[str]:
    dataset = os.environ["TB2_DATASET"]
    if "@" in dataset:
        name, version = dataset.split("@", 1)
    else:
        name, version = dataset, None
    cfg = DatasetConfig(name=name, version=version)
    tasks = await cfg.get_task_configs(disable_verification=True)
    return [str(task.path) for task in tasks]


def job_config(label: str, port: int, names: list[str]) -> dict:
    dataset = os.environ["TB2_DATASET"]
    if "@" in dataset:
        dataset_name, dataset_version = dataset.split("@", 1)
    else:
        dataset_name, dataset_version = dataset, None

    disable_verification = os.environ["TB2_DISABLE_VERIFICATION"].lower() in {"1", "true", "yes"}
    run_dir = Path(sys.argv[1])
    return {
        "job_name": f"{os.environ['RUN_NAME']}_{label}",
        "jobs_dir": str(run_dir / "harbor" / label),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "datasets": [
            {
                "name": dataset_name,
                "version": dataset_version,
                "task_names": names,
            }
        ],
        "agents": [
            {
                "import_path": os.environ["TB2_AGENT_IMPORT_PATH"],
                "model_name": os.environ["TB2_MODEL_NAME"],
                "override_timeout_sec": float(os.environ["TB2_AGENT_TIMEOUT_SECONDS"]),
                "kwargs": {
                    "config_file": os.environ["TB2_CONFIG_FILE"],
                    "api_base": f"http://127.0.0.1:{port}/v1",
                    "tokenizer_path": os.environ["TB2_TOKENIZER_PATH"],
                    "trajectories_dir": str(run_dir / "trajectories" / label),
                    "step_limit": int(os.environ["TB2_STEP_LIMIT"]),
                    "wall_time_limit_seconds": int(os.environ["TB2_WALL_TIME_LIMIT_SECONDS"]),
                    "max_tokens": int(os.environ["TB2_MAX_TOKENS"]),
                },
            }
        ],
        "environment": {
            "type": "docker",
            "delete": True,
        },
        "verifier": {
            "disable": disable_verification,
        },
    }


async def main() -> None:
    run_dir = Path(sys.argv[1])
    names = await task_names()
    midpoint = (len(names) + 1) // 2
    splits = {
        "gpu0": names[:midpoint],
        "gpu1": names[midpoint:],
    }

    (run_dir / "task_split.json").write_text(json.dumps(splits, indent=2) + "\n")
    for label, port in [("gpu0", 8000), ("gpu1", 8001)]:
        config = job_config(label, port, splits[label])
        JobConfig.model_validate(config)
        (run_dir / f"harbor_job_config_{label}.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"{label}: {len(splits[label])} tasks, port {port}")


asyncio.run(main())
PY

run_half() {
  local label="$1"
  "$HARBOR_BIN" run \
    --config "$RUN_DIR/harbor_job_config_${label}.json" \
    --yes \
    > "$LOG_DIR/${label}.log" 2>&1
}

run_half gpu0 &
pid0="$!"
echo "$pid0" > "$RUN_DIR/gpu0.pid"

run_half gpu1 &
pid1="$!"
echo "$pid1" > "$RUN_DIR/gpu1.pid"

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_token_timing.py" "$TRAJECTORIES_DIR" --output-dir "$REPORT_DIR" \
  > "$LOG_DIR/summarize.log" 2>&1 || status=1

exit "$status"
