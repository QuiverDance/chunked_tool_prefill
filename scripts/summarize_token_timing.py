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
STREAM_SAMPLE_INTERVAL_S = 0.05
TOKENIZER_CACHE: dict[tuple[str, bool], Any] = {}

MODEL_FIELDS = [
    "instance_id",
    "trajectory",
    "message_index",
    "model_call_index",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "ttft_s",
    "model_total_s",
    "decode_s",
]

TOOL_FIELDS = [
    "instance_id",
    "trajectory",
    "message_index",
    "tool_call_id",
    "command_category",
    "duration_s",
    "time_to_first_output_s",
    "returncode",
    "raw_output_tokens",
    "rendered_observation_tokens",
    "was_truncated",
    "raw_output_chars",
    "stream_max_tokens_per_sample",
    "stream_mean_tokens_per_sample",
]

PROBLEM_FIELDS = [
    "instance_id",
    "trajectory",
    "problem_e2e_s",
    "serving_relevant_e2e_s",
    "agent_overhead_s",
    "model_calls",
    "tool_calls",
    "sum_ttft_s",
    "sum_model_total_s",
    "sum_tool_duration_s",
    "ttft_share_of_e2e",
    "model_total_share_of_e2e",
    "tool_duration_share_of_e2e",
    "ttft_share_of_serving_relevant_e2e",
]

MODEL_SUMMARY_FIELDS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "ttft_s",
    "model_total_s",
    "decode_s",
]
PROBLEM_SUMMARY_FIELDS = [
    "problem_e2e_s",
    "serving_relevant_e2e_s",
    "agent_overhead_s",
    "model_calls",
    "tool_calls",
    "sum_ttft_s",
    "sum_model_total_s",
    "sum_tool_duration_s",
    "ttft_share_of_e2e",
    "model_total_share_of_e2e",
    "tool_duration_share_of_e2e",
    "ttft_share_of_serving_relevant_e2e",
]
TOOL_SUMMARY_FIELDS = [
    "duration_s",
    "time_to_first_output_s",
    "raw_output_tokens",
    "rendered_observation_tokens",
    "was_truncated",
    "raw_output_chars",
    "stream_max_tokens_per_sample",
    "stream_mean_tokens_per_sample",
]
PERCENTILES = [50, 90, 95, 99]


@dataclass(frozen=True)
class FractionMetric:
    name: str
    fields: tuple[str, ...]
    matches: Callable[..., bool]


FRACTION_METRICS = [
    FractionMetric(
        "rendered_observation_was_truncated",
        ("was_truncated",),
        lambda was_truncated: bool(was_truncated),
    ),
]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_calls, tool_calls, problem_rows = collect_records(
        run_dir,
        tokenizer_path=args.tokenizer_path,
        tokenizer_local_files_only=args.tokenizer_local_files_only,
        stream_sample_interval_s=args.stream_sample_interval_s,
    )
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
    parser.add_argument(
        "--tokenizer-path",
        help="Tokenizer used for offline tool-output metrics. Defaults to each trajectory's agent tokenizer_path.",
    )
    parser.add_argument(
        "--tokenizer-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load tokenizer from local files only. Use --no-tokenizer-local-files-only to allow downloads.",
    )
    parser.add_argument(
        "--stream-sample-interval-s",
        type=float,
        default=STREAM_SAMPLE_INTERVAL_S,
        help="Offline stream token sample interval in seconds.",
    )
    return parser.parse_args()


def collect_records(
    run_dir: Path,
    *,
    tokenizer_path: str | None = None,
    tokenizer_local_files_only: bool = True,
    stream_sample_interval_s: float = STREAM_SAMPLE_INTERVAL_S,
) -> tuple[list[Record], list[Record], list[Record]]:
    model_calls: list[Record] = []
    tool_calls: list[Record] = []
    problem_rows: list[Record] = []

    for trajectory in sorted(run_dir.glob("**/*.traj.json")):
        data = load_json(trajectory)
        tokenizer = load_tokenizer(
            tokenizer_path or tokenizer_path_from_trajectory(data),
            local_files_only=tokenizer_local_files_only,
            required=bool(tokenizer_path),
        )
        instance_id = data.get("instance_id") or trajectory.parent.name
        trajectory_model_calls: list[Record] = []
        trajectory_tool_calls: list[Record] = []

        for message_index, message in enumerate(data.get("messages", [])):
            if not isinstance(message, dict):
                continue
            if model_call := model_row_from_message(message, instance_id, trajectory, message_index):
                model_calls.append(model_call)
                trajectory_model_calls.append(model_call)
            rows = tool_rows_from_message(
                message,
                instance_id,
                trajectory,
                message_index,
                tokenizer=tokenizer,
                stream_sample_interval_s=stream_sample_interval_s,
            )
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
    serving_relevant_e2e_s = sum_model_total_s + sum_tool_duration_s
    agent_overhead_s = problem_e2e_s - serving_relevant_e2e_s if problem_e2e_s is not None else None
    return {
        "instance_id": instance_id,
        "trajectory": str(trajectory),
        "problem_e2e_s": problem_e2e_s,
        "serving_relevant_e2e_s": serving_relevant_e2e_s,
        "agent_overhead_s": agent_overhead_s,
        "model_calls": len(model_calls),
        "tool_calls": len(tool_calls),
        "sum_ttft_s": sum_ttft_s,
        "sum_model_total_s": sum_model_total_s,
        "sum_tool_duration_s": sum_tool_duration_s,
        "ttft_share_of_e2e": safe_ratio(sum_ttft_s, problem_e2e_s),
        "model_total_share_of_e2e": safe_ratio(sum_model_total_s, problem_e2e_s),
        "tool_duration_share_of_e2e": safe_ratio(sum_tool_duration_s, problem_e2e_s),
        "ttft_share_of_serving_relevant_e2e": safe_ratio(sum_ttft_s, serving_relevant_e2e_s),
    }


def model_row_from_message(message: Record, instance_id: str, trajectory: Path, message_index: int) -> Record:
    extra = extra_from_message(message)
    timing = timing_from_extra(extra)
    record = timing.get("model_call") or usage_from_message(message, extra)
    return record_row(record, MODEL_FIELDS, instance_id, trajectory, message_index) if record else {}


def tool_rows_from_message(
    message: Record,
    instance_id: str,
    trajectory: Path,
    message_index: int,
    *,
    tokenizer,
    stream_sample_interval_s: float,
) -> list[Record]:
    timing = timing_from_extra(extra_from_message(message))
    return [
        record_row(
            enriched_tool_call(
                tool_call,
                message,
                tokenizer,
                stream_sample_interval_s,
                rendered_observation_owner=index == 0,
            ),
            TOOL_FIELDS,
            instance_id,
            trajectory,
            message_index,
        )
        for index, tool_call in enumerate(timing.get("tool_calls", []))
    ]


def extra_from_message(message: Record) -> Record:
    extra = message.get("extra", {})
    return extra if isinstance(extra, dict) else {}


def timing_from_extra(extra: Record) -> Record:
    timing = extra.get("token_timing", {})
    return timing if isinstance(timing, dict) else {}


def enriched_tool_call(
    tool_call: Record,
    message: Record,
    tokenizer,
    stream_sample_interval_s: float,
    rendered_observation_owner: bool = False,
) -> Record:
    record = dict(tool_call)
    raw_output = raw_output_from_message(message, record)
    stdout = str(record.get("stdout") if record.get("stdout") not in (None, "") else raw_output)
    stderr = str(record.get("stderr") or "")
    stream_record = {
        **record,
        "output": raw_output,
        "stdout": stdout,
        "stderr": stderr,
    }

    record.setdefault("raw_output_chars", len(raw_output))
    record.setdefault("raw_output_bytes", len(raw_output.encode("utf-8")))

    if tokenizer is not None:
        record["raw_output_tokens"] = count_tokens(tokenizer, raw_output)

    record.update(stream_timing_summary(tokenizer, stream_record, stream_sample_interval_s))

    if rendered_observation_owner or record.get("rendered_observation_owner"):
        rendered = rendered_observation_text(message)
        record.update(rendered_observation_summary(tokenizer, raw_output, rendered))

    return record


def raw_output_from_message(message: Record, tool_call: Record) -> str:
    extra = extra_from_message(message)
    value = extra.get("raw_output")
    if value is None:
        value = tool_call.get("output") or tool_call.get("stdout") or ""
    return str(value or "")


def tokenizer_path_from_trajectory(data: Record) -> str:
    value = data.get("info", {}).get("config", {}).get("agent", {}).get("tokenizer_path")
    return str(value or "")


def load_tokenizer(tokenizer_path: str | None, *, local_files_only: bool, required: bool = False):
    if not tokenizer_path:
        return None
    cache_key = (str(tokenizer_path), local_files_only)
    if cache_key not in TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError:
            if required:
                raise
            TOKENIZER_CACHE[cache_key] = None
            return None

        TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    return TOKENIZER_CACHE[cache_key]


def count_tokens(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def rendered_observation_text(message: Record) -> str:
    if "content" in message:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(text_part(item) for item in content if text_part(item))
    if "output" in message:
        return str(message.get("output") or "")
    return ""


def text_part(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    value = item.get("text") or item.get("content")
    return value if isinstance(value, str) else ""


def rendered_observation_summary(tokenizer, raw_output: str, rendered: str) -> Record:
    elided_chars = rendered_elided_chars(rendered)
    summary: Record = {
        "was_truncated": elided_chars is not None,
        "raw_output_chars": len(raw_output),
    }
    if tokenizer is not None:
        summary["rendered_observation_tokens"] = count_tokens(tokenizer, rendered)
    return summary


def rendered_elided_chars(rendered: str) -> int | None:
    import re

    match = re.search(r"<elided_chars>\s*(\d+)\s*(?:</elided_chars>)?", rendered)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+characters\s+elided\b", rendered)
    return int(match.group(1)) if match else None


def stream_timing_summary(tokenizer, record: Record, stream_sample_interval_s: float) -> Record:
    if tokenizer is None:
        return {}

    token_samples = stream_token_samples(tokenizer, record, stream_sample_interval_s)
    return stream_token_sample_summary(token_samples)


def stream_token_samples(tokenizer, record: Record, stream_sample_interval_s: float) -> list[Record]:
    samples = output_samples_for_record(record, stream_sample_interval_s)
    if tokenizer is None or not samples:
        return []

    total = 0
    rows = []
    previous_output_chars = 0
    previous_stdout_bytes = 0
    previous_stderr_bytes = 0
    stdout = record.get("stdout", "") or ""
    stderr = record.get("stderr", "") or ""

    for sample in samples:
        output_chars = sample.get("output_chars")
        if output_chars is not None:
            output_chars = int(output_chars)
            text = (record.get("output", "") or "")[previous_output_chars:output_chars]
            previous_output_chars = output_chars
            stdout_bytes = 0
            stderr_bytes = 0
        else:
            stdout_bytes = int(sample.get("stdout_bytes") or 0)
            stderr_bytes = int(sample.get("stderr_bytes") or 0)
            text = text_byte_slice(stdout, previous_stdout_bytes, stdout_bytes)
            text += text_byte_slice(stderr, previous_stderr_bytes, stderr_bytes)
        tokens = count_tokens(tokenizer, text) or 0
        total += tokens
        row: Record = {
            "index": sample.get("index"),
            "t": sample.get("t"),
            "tokens": tokens,
            "cumulative_tokens": total,
        }
        if output_chars is not None:
            row["output_chars"] = output_chars
        else:
            row["stdout_bytes"] = stdout_bytes
            row["stderr_bytes"] = stderr_bytes
            row["output_bytes"] = stdout_bytes + stderr_bytes
        rows.append(row)
        if output_chars is None:
            previous_stdout_bytes = stdout_bytes
            previous_stderr_bytes = stderr_bytes

    return rows


def output_samples_for_record(record: Record, stream_sample_interval_s: float) -> list[Record]:
    output_events = record.get("output_events") or []
    duration = number_from(record, "duration_s")
    if output_events and duration is not None:
        return output_samples_from_events(
            output_events,
            duration,
            len(record.get("output", "") or ""),
            final_output_bytes=int(record.get("raw_output_bytes") or len((record.get("output", "") or "").encode("utf-8"))),
            stream_sample_interval_s=stream_sample_interval_s,
        )
    samples = record.get("output_samples") or []
    return samples if isinstance(samples, list) else []


def output_samples_from_events(
    events: list[Record],
    duration_s: float,
    final_output_chars: int,
    *,
    final_output_bytes: int | None,
    stream_sample_interval_s: float,
) -> list[Record]:
    first_sample: Record = {"index": 0, "t": 0.0, "output_chars": 0}
    if final_output_bytes is not None:
        first_sample["output_bytes"] = 0
    samples = [first_sample]
    event_index = 0
    current_chars = 0
    current_bytes: int | None = 0 if final_output_bytes is not None else None
    index = 1
    while index * stream_sample_interval_s < duration_s:
        sample_time = index * stream_sample_interval_s
        while event_index < len(events) and float(events[event_index]["t"]) <= sample_time:
            current_chars = int(events[event_index]["output_chars"])
            if current_bytes is not None:
                current_bytes = int(events[event_index].get("output_bytes") or current_bytes)
            event_index += 1
        sample: Record = {"index": index, "t": sample_time, "output_chars": current_chars}
        if current_bytes is not None:
            sample["output_bytes"] = current_bytes
        samples.append(sample)
        index += 1
    final_sample: Record = {"index": "final", "t": duration_s, "output_chars": final_output_chars}
    if final_output_bytes is not None:
        final_sample["output_bytes"] = final_output_bytes
    samples.append(final_sample)
    return samples


def stream_token_sample_summary(samples: list[Record]) -> Record:
    tokens = [int(sample["tokens"]) for sample in samples]
    return {
        "stream_max_tokens_per_sample": max(tokens) if tokens else 0,
        "stream_mean_tokens_per_sample": sum(tokens) / len(tokens) if tokens else None,
    }


def text_byte_slice(text: str, start: int, end: int) -> str:
    if end <= start:
        return ""
    data = (text or "").encode("utf-8")
    return data[max(0, start) : max(0, end)].decode("utf-8", errors="ignore")


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
    }


def record_row(record: Record, fields: list[str], instance_id: str, trajectory: Path, message_index: int) -> Record:
    row = {field: record.get(field, "") for field in fields}
    row["instance_id"] = instance_id
    row["trajectory"] = str(trajectory)
    row["message_index"] = message_index
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
