#!/usr/bin/env python3
"""Collect CI-Bench validation records and write a compact correctness summary."""

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
    results_path = output_dir / "cibench_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    summary = build_summary(records)
    (output_dir / "summary.cibench.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def collect_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in sorted(run_dir.glob("**/cibench_results.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("artifact_id"), record.get("patch_path"), record.get("validation_log_path"))
            if key in seen:
                continue
            seen.add(key)
            record.setdefault("source_results_file", str(path))
            records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    patch_applies = sum(1 for record in records if record.get("patch_applies"))
    plausible = sum(1 for record in records if record.get("plausible"))
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": total,
        "patch_applies": patch_applies,
        "plausible": plausible,
        "patch_apply_rate": patch_applies / total if total else 0,
        "plausibility_rate": plausible / total if total else 0,
        "exit_status_counts": statuses,
    }


if __name__ == "__main__":
    main()
