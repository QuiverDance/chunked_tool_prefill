#!/usr/bin/env python3
"""Collect LogDx-CI diagnosis records and write a compact summary."""

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
    results_path = output_dir / "logdx_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    summary = build_summary(records)
    (output_dir / "summary.logdx.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def collect_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in sorted(run_dir.glob("**/logdx_results.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("case_id"), record.get("prediction_path"), record.get("trajectory_path"))
            if key in seen:
                continue
            seen.add(key)
            record.setdefault("source_results_file", str(path))
            records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "total": len(records),
        "submitted": sum(1 for record in records if record.get("exit_status") == "Submitted"),
        "diagnosis_valid": sum(1 for record in records if record.get("diagnosis_valid")),
        "score": numeric_summary(numbers_for(records, "score")),
        "category_match": numeric_summary(numbers_for(records, "category_match")),
        "confident_error_rate": bool_rate(records, "confident_error"),
        "raw_log_chars": numeric_summary(numbers_for(records, "raw_log_chars")),
        "raw_log_tokens": numeric_summary(numbers_for(records, "raw_log_tokens")),
        "exit_status_counts": statuses,
    }


def numbers_for(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0, "mean": None, "max": None}
    return {"count": len(values), "sum": sum(values), "mean": sum(values) / len(values), "max": max(values)}


def bool_rate(records: list[dict[str, Any]], field: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if record.get(field)) / len(records)


if __name__ == "__main__":
    main()
