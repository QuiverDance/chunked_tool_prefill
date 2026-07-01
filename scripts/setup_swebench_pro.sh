#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEBENCH_PRO_HOME="${SWEBENCH_PRO_HOME:-$HOME/.cache/swebench-pro-os}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/python}"
PIP_BIN="${PIP_BIN:-$ROOT_DIR/.conda/miniswe-py311/bin/pip}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -x "$PIP_BIN" ]]; then
  PIP_BIN="pip"
fi

mkdir -p "$(dirname "$SWEBENCH_PRO_HOME")"
if [[ -d "$SWEBENCH_PRO_HOME/.git" ]]; then
  git -C "$SWEBENCH_PRO_HOME" pull --ff-only
else
  git clone https://github.com/scaleapi/SWE-bench_Pro-os "$SWEBENCH_PRO_HOME"
fi

"$PIP_BIN" install -r "$SWEBENCH_PRO_HOME/requirements.txt"

"$PYTHON_BIN" - <<'PY'
from datasets import load_dataset

dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
count = len(dataset)
if count != 731:
    raise SystemExit(f"expected 731 public SWE-bench Pro instances, got {count}")
print(f"SWE-bench Pro public test split: {count} instances")
PY

if command -v docker >/dev/null 2>&1; then
  docker info >/dev/null
  echo "Docker is available"
else
  echo "docker command not found; install or expose Docker before running the benchmark" >&2
fi

echo "SWEBENCH_PRO_HOME=$SWEBENCH_PRO_HOME"
