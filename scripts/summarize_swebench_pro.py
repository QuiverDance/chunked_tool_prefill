#!/usr/bin/env python3
"""Collect SWE-bench Pro generation and eval records into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(run_dir)
    results_path = output_dir / "swebench_pro_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    summary = build_summary(records, run_dir, output_dir)
    (output_dir / "summary.swebench_pro.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def collect_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in sorted(run_dir.glob("**/swebench_pro_results.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("instance_id"), record.get("patch_path"), record.get("trajectory_path"))
            if key in seen:
                continue
            seen.add(key)
            record.setdefault("source_results_file", str(path))
            records.append(record)
    return records


def build_summary(records: list[dict[str, Any]], run_dir: Path, output_dir: Path) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1

    official_eval = collect_eval_summary(run_dir)
    return {
        "total": len(records),
        "submitted": sum(1 for record in records if record.get("submitted")),
        "generated_patches": sum(1 for record in records if record.get("has_patch")),
        "no_patch": sum(1 for record in records if not record.get("has_patch")),
        "resolved": official_eval.get("resolved"),
        "pass_at_1": official_eval.get("pass_at_1"),
        "eval_failures": official_eval.get("eval_failures"),
        "exit_status_counts": statuses,
        "official_eval": official_eval,
        "timing": load_timing_summary(output_dir),
    }


def collect_eval_summary(run_dir: Path) -> dict[str, Any]:
    records = []
    paths = []
    for path in sorted(run_dir.glob("**/eval_results.json")):
        paths.append(str(path))
        records.extend(eval_records(load_json(path)))
    total = len(records)
    resolved = sum(1 for record in records if eval_record_resolved(record))
    failures = sum(1 for record in records if eval_record_failed(record))
    return {
        "paths": paths,
        "total": total,
        "resolved": resolved if paths else None,
        "pass_at_1": resolved / total if total else None,
        "eval_failures": failures if paths else None,
    }


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def eval_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("results", "instances", "eval_results"):
        value = data.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    if all(isinstance(value, bool) for value in data.values()):
        return [{"instance_id": key, "resolved": value} for key, value in data.items()]
    if all(isinstance(value, dict) for value in data.values()):
        return list(data.values())
    return [data]


def eval_record_resolved(record: dict[str, Any]) -> bool:
    for key in ("resolved", "success", "passed", "pass", "result"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"resolved", "pass", "passed", "success", "true"}:
            return True
    return False


def eval_record_failed(record: dict[str, Any]) -> bool:
    for key in ("error", "exception", "traceback", "failure"):
        if record.get(key):
            return True
    status = str(record.get("status") or "").lower()
    return status in {"error", "failed", "exception"}


def load_timing_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
