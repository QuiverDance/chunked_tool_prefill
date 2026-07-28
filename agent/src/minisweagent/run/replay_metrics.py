"""Small metric helpers for trace replay results."""

from __future__ import annotations

import statistics
from typing import Any

RESULT_KEYS = (
    "trajectory_path",
    "instance_id",
    "algorithm",
    "step_index",
    "valid",
    "skip_reason",
    "prompt_tokens",
    "trace_prompt_tokens",
    "trace_completion_tokens",
    "requested_completion_tokens",
    "replay_completion_tokens",
    "trace_finish_reason",
    "problem_e2e_s",
    "trace_ttft_s",
    "replay_ttft_s",
    "trace_model_total_s",
    "replay_model_total_s",
    "trace_decode_s",
    "replay_decode_s",
    "cached_tokens",
    "tool_call_count",
    "simulated_tool_duration_s",
    "tool_output_chars",
    "tool_output_events",
    "missing_tool_timing_count",
    "prefill_count",
    "prefill_submitted_count",
    "prefill_coalesced_count",
    "prefill_started_count",
    "prefill_completed_count",
    "prefill_completed_prompt_tokens",
    "prefilled_prompt_suffix_tokens",
    "prefilled_tool_output_tokens",
    "unprefilled_prompt_suffix_tokens",
    "unprefilled_tool_output_tokens",
    "prefill_active_at_tool_end",
    "prefill_pending_at_tool_end",
    "active_prefill_prefix_len_at_tool_end",
    "pending_prefill_prefix_len_at_tool_end",
    "active_prefill_cancel_requested_at_tool_end",
    "active_prefill_cancel_latency_s",
    "active_prefill_cancel_error",
    "candidate_selected_count",
    "candidate_skipped_capacity_count",
    "candidate_submitted_count",
    "candidate_completed_count",
    "candidate_shared_prefix_tokens",
    "candidate_verified_prefix_tokens",
    "candidate_verified_tool_output_tokens",
    "candidate_pruned_count",
    "candidate_surviving_count",
    "candidate_fallback_to_chunked",
    "candidate_cancelled_count",
)

SUMMARY_METRIC_KEYS = (
    "prompt_tokens",
    "trace_prompt_tokens",
    "trace_completion_tokens",
    "replay_completion_tokens",
    "problem_e2e_s",
    "trace_ttft_s",
    "replay_ttft_s",
    "trace_model_total_s",
    "replay_model_total_s",
    "trace_decode_s",
    "replay_decode_s",
    "cached_tokens",
    "tool_call_count",
    "simulated_tool_duration_s",
    "tool_output_chars",
    "tool_output_events",
    "missing_tool_timing_count",
    "prefill_count",
    "prefill_submitted_count",
    "prefill_coalesced_count",
    "prefill_started_count",
    "prefill_completed_count",
    "prefill_completed_prompt_tokens",
    "prefilled_prompt_suffix_tokens",
    "prefilled_tool_output_tokens",
    "unprefilled_prompt_suffix_tokens",
    "unprefilled_tool_output_tokens",
    "prefill_active_at_tool_end",
    "prefill_pending_at_tool_end",
    "active_prefill_prefix_len_at_tool_end",
    "pending_prefill_prefix_len_at_tool_end",
    "active_prefill_cancel_requested_at_tool_end",
    "active_prefill_cancel_latency_s",
    "candidate_selected_count",
    "candidate_skipped_capacity_count",
    "candidate_submitted_count",
    "candidate_completed_count",
    "candidate_shared_prefix_tokens",
    "candidate_verified_prefix_tokens",
    "candidate_verified_tool_output_tokens",
    "candidate_pruned_count",
    "candidate_surviving_count",
    "candidate_fallback_to_chunked",
    "candidate_cancelled_count",
)


def record_for_output(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in RESULT_KEYS if key in record}


def summarize(records: list[dict[str, Any]], invalid_records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("valid")]
    skipped = [record for record in records if not record.get("valid")]
    summary = {
        "total": len(records) + len(invalid_records),
        "measured": len(records),
        "valid": len(valid),
        "skipped": len(skipped) + len(invalid_records),
        "capacity_skips": sum(
            1 for record in records + invalid_records if record.get("skip_reason") == "skipped_capacity"
        ),
        "skip_reasons": reason_counts(records + invalid_records),
        "algorithms": value_counts(record.get("algorithm") for record in valid),
    }
    for key in SUMMARY_METRIC_KEYS:
        summary[key] = stats([record.get(key) for record in valid])
    return summary


def stats(values: list[Any]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": statistics.median(numbers),
        "p95": percentile(numbers, 0.95),
    }


def percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]


def reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = record.get("skip_reason") or ""
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def value_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts
