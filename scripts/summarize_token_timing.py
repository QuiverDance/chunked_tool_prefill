#!/usr/bin/env python3
"""Summarize token and tool timing metrics from mini SWE-bench trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

Record = dict[str, Any]

MODEL_FIELDS = [
    "instance_id",
    "trajectory",
    "message_index",
    "model_call_index",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "status",
    "incomplete_details",
    "stream",
    "request_start_s",
    "first_chunk_s",
    "ttft_s",
    "model_total_s",
    "decode_s",
    "stream_chunk_count",
]

TOOL_FIELDS = [
    "instance_id",
    "trajectory",
    "message_index",
    "tool_call_id",
    "sequence_index",
    "sequence_separator",
    "command_category",
    "command",
    "start_ts",
    "first_stdout_ts",
    "last_stdout_ts",
    "end_ts",
    "duration_s",
    "time_to_first_stdout_s",
    "returncode",
    "output_tokens",
    "stdout_tokens",
    "stderr_tokens",
    "exception_info",
]

PROBLEM_FIELDS = [
    "instance_id",
    "trajectory",
    "problem_e2e_s",
    "model_calls",
    "tool_calls",
    "sum_ttft_s",
    "sum_model_total_s",
    "sum_tool_duration_s",
    "ttft_share_of_e2e",
    "model_total_share_of_e2e",
    "tool_duration_share_of_e2e",
]

MODEL_SUMMARY_FIELDS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "first_chunk_s",
    "ttft_s",
    "model_total_s",
    "decode_s",
    "stream_chunk_count",
]
PROBLEM_SUMMARY_FIELDS = [
    "problem_e2e_s",
    "model_calls",
    "tool_calls",
    "sum_ttft_s",
    "sum_model_total_s",
    "sum_tool_duration_s",
    "ttft_share_of_e2e",
    "model_total_share_of_e2e",
    "tool_duration_share_of_e2e",
]
TOOL_SUMMARY_FIELDS = [
    "start_ts",
    "first_stdout_ts",
    "last_stdout_ts",
    "end_ts",
    "duration_s",
    "time_to_first_stdout_s",
    "output_tokens",
    "stdout_tokens",
    "stderr_tokens",
]
PERCENTILES = [50, 90, 95, 99]


@dataclass(frozen=True)
class FractionMetric:
    name: str
    fields: tuple[str, ...]
    matches: Callable[..., bool]


FRACTION_METRICS = [
    FractionMetric(
        "stdout_tokens_ge_512_and_duration_ge_1s",
        ("stdout_tokens", "duration_s"),
        lambda stdout_tokens, duration_s: stdout_tokens >= 512 and duration_s >= 1,
    ),
    FractionMetric(
        "first_stdout_before_half_duration",
        ("time_to_first_stdout_s", "duration_s"),
        lambda first_stdout_s, duration_s: first_stdout_s < 0.5 * duration_s,
    ),
]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_calls, tool_calls, problem_rows = collect_records(run_dir)
    write_csv(output_dir / "model_calls.csv", model_calls, MODEL_FIELDS)
    write_csv(output_dir / "tool_calls.csv", tool_calls, TOOL_FIELDS)
    write_csv(output_dir / "problem_timings.csv", problem_rows, PROBLEM_FIELDS)

    summary = build_summary(run_dir, model_calls, tool_calls, problem_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def collect_records(run_dir: Path) -> tuple[list[Record], list[Record], list[Record]]:
    model_calls: list[Record] = []
    tool_calls: list[Record] = []
    problem_rows: list[Record] = []

    for trajectory in sorted(run_dir.glob("**/*.traj.json")):
        data = load_json(trajectory)
        instance_id = data.get("instance_id") or trajectory.parent.name
        trajectory_model_calls: list[Record] = []
        trajectory_tool_calls: list[Record] = []

        for message_index, message in enumerate(data.get("messages", [])):
            if not isinstance(message, dict):
                continue
            if model_call := model_row_from_message(message, instance_id, trajectory, message_index):
                model_calls.append(model_call)
                trajectory_model_calls.append(model_call)
            rows = tool_rows_from_message(message, instance_id, trajectory, message_index)
            tool_calls.extend(rows)
            trajectory_tool_calls.extend(rows)
        problem_rows.append(problem_row_from_trajectory(data, instance_id, trajectory, trajectory_model_calls, trajectory_tool_calls))

    return model_calls, tool_calls, problem_rows


def problem_row_from_trajectory(
    data: Record,
    instance_id: str,
    trajectory: Path,
    model_calls: list[Record],
    tool_calls: list[Record],
) -> Record:
    timing = data.get("info", {}).get("token_timing", {}).get("problem", {})
    problem_e2e_s = number_from(timing, "e2e_s")
    sum_ttft_s = sum(number_from(record, "ttft_s") or 0.0 for record in model_calls)
    sum_model_total_s = sum(number_from(record, "model_total_s") or 0.0 for record in model_calls)
    sum_tool_duration_s = sum(number_from(record, "duration_s") or 0.0 for record in tool_calls)
    return {
        "instance_id": instance_id,
        "trajectory": str(trajectory),
        "problem_e2e_s": problem_e2e_s,
        "model_calls": len(model_calls),
        "tool_calls": len(tool_calls),
        "sum_ttft_s": sum_ttft_s,
        "sum_model_total_s": sum_model_total_s,
        "sum_tool_duration_s": sum_tool_duration_s,
        "ttft_share_of_e2e": safe_ratio(sum_ttft_s, problem_e2e_s),
        "model_total_share_of_e2e": safe_ratio(sum_model_total_s, problem_e2e_s),
        "tool_duration_share_of_e2e": safe_ratio(sum_tool_duration_s, problem_e2e_s),
    }


def model_row_from_message(message: Record, instance_id: str, trajectory: Path, message_index: int) -> Record:
    extra = extra_from_message(message)
    timing = timing_from_extra(extra)
    record = timing.get("model_call") or usage_from_message(message, extra)
    return record_row(record, MODEL_FIELDS, instance_id, trajectory, message_index) if record else {}


def tool_rows_from_message(message: Record, instance_id: str, trajectory: Path, message_index: int) -> list[Record]:
    timing = timing_from_extra(extra_from_message(message))
    return [
        record_row(tool_call, TOOL_FIELDS, instance_id, trajectory, message_index)
        for tool_call in timing.get("tool_calls", [])
    ]


def extra_from_message(message: Record) -> Record:
    extra = message.get("extra", {})
    return extra if isinstance(extra, dict) else {}


def timing_from_extra(extra: Record) -> Record:
    timing = extra.get("token_timing", {})
    return timing if isinstance(timing, dict) else {}


def usage_from_message(message: Record, extra: Record) -> Record:
    response = extra.get("response") or (message if "usage" in message else {})
    usage = response.get("usage") or {}
    if not usage:
        return {}
    choices = response.get("choices") or []
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "status": response.get("status"),
        "incomplete_details": response.get("incomplete_details"),
    }


def record_row(record: Record, fields: list[str], instance_id: str, trajectory: Path, message_index: int) -> Record:
    row = {field: record.get(field, "") for field in fields}
    row["instance_id"] = instance_id
    row["trajectory"] = str(trajectory)
    row["message_index"] = message_index
    if row.get("incomplete_details"):
        row["incomplete_details"] = json.dumps(row["incomplete_details"], sort_keys=True)
    return row


def load_json(path: Path) -> Record:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_csv(path: Path, rows: list[Record], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(run_dir: Path, model_calls: list[Record], tool_calls: list[Record], problem_rows: list[Record]) -> Record:
    return {
        "run_dir": str(run_dir),
        "trajectory_count": len({record["trajectory"] for record in model_calls + tool_calls + problem_rows}),
        "problems": problem_summary(problem_rows),
        "model_calls": model_summary(model_calls),
        "tool_calls": tool_summary(tool_calls),
        "commands": command_summaries(tool_calls),
    }


def problem_summary(records: list[Record]) -> Record:
    return {
        "count": len(records),
        **numeric_summaries(records, PROBLEM_SUMMARY_FIELDS),
    }


def model_summary(records: list[Record]) -> Record:
    return {
        "count": len(records),
        **numeric_summaries(records, MODEL_SUMMARY_FIELDS),
        "finish_reason_counts": value_counts(records, "finish_reason"),
    }


def tool_summary(records: list[Record]) -> Record:
    return {
        "count": len(records),
        **numeric_summaries(records, TOOL_SUMMARY_FIELDS),
        "fractions": fraction_summaries(records),
        "returncode_counts": value_counts(records, "returncode"),
    }


def command_summaries(tool_calls: list[Record]) -> Record:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in tool_calls:
        groups[record.get("command_category") or "unknown"].append(record)
    return {
        command: tool_summary(records)
        for command, records in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def fraction_summaries(records: list[Record]) -> Record:
    return {metric.name: fraction(records, metric) for metric in FRACTION_METRICS}


def fraction(records: list[Record], metric: FractionMetric) -> Record:
    total = 0
    count = 0
    for record in records:
        values = [number_from(record, field) for field in metric.fields]
        if any(value is None for value in values):
            continue
        total += 1
        count += int(metric.matches(*values))
    return {"count": count, "total": total, "fraction": count / total if total else None}


def numeric_summaries(records: list[Record], fields: list[str]) -> Record:
    return {field: numeric_summary(numbers_for(records, field)) for field in fields}


def value_counts(records: list[Record], field: str) -> dict[str, int]:
    return dict(Counter("" if record.get(field) is None else str(record.get(field)) for record in records))


def numbers_for(records: list[Record], field: str) -> list[float]:
    return [value for record in records if (value := number_from(record, field)) is not None]


def number_from(record: Record, field: str) -> float | None:
    value = record.get(field)
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return numerator / denominator


def numeric_summary(values: list[float]) -> Record:
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": None,
            **{f"p{percentile_value}": None for percentile_value in PERCENTILES},
            "max": None,
        }

    ordered = sorted(values)
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "mean": statistics.fmean(ordered),
        **{f"p{percentile_value}": percentile(ordered, percentile_value) for percentile_value in PERCENTILES},
        "max": ordered[-1],
    }


def percentile(values: list[float], percentile_value: int) -> float:
    return values[round((len(values) - 1) * percentile_value / 100)]


if __name__ == "__main__":
    main()
