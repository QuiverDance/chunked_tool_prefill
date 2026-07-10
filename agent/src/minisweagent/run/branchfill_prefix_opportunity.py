"""Measure exact tool-output prefix reuse opportunities in saved trajectories."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from jinja2 import Template

from minisweagent.run.benchmarks.utils.token_timing import pipeline_category

Record = dict[str, Any]
PAYLOAD_BOUNDARY = "<output>\n"
TRUNCATED_PAYLOAD_BOUNDARY = "<output_head>\n"
TRUNCATED_TAIL_BOUNDARY = "<output_tail>\n"
app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ToolOutput:
    call_index: int
    tool_call_id: str
    tool_name: str
    command: str
    command_category: str
    exact_args_key: str
    raw_output: str
    raw_tokens: list[int]
    rendered_prefix_tokens: list[int]
    rendered_payload_tokens: int
    render_kind: str
    returncode: int | None
    message_index: int


@app.command()
def main(
    run_dir: Path,
    output_dir: Path = typer.Option(..., "--output-dir"),
    tokenizer_path: str | None = typer.Option(None, "--tokenizer-path"),
    allow_tokenizer_download: bool = typer.Option(False, "--allow-tokenizer-download"),
) -> None:
    """Analyze exact causal tool-output prefix reuse in saved trajectories."""
    trajectories = sorted(run_dir.glob("**/*.traj.json"))
    if not trajectories:
        raise typer.BadParameter(f"No trajectory files found below {run_dir}")

    tokenizer_path = tokenizer_path or tokenizer_path_from_trajectory(trajectories[0])
    if not tokenizer_path:
        raise typer.BadParameter("No tokenizer path was supplied or recorded in the trajectories")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=not allow_tokenizer_download)

    summary = run_analysis(run_dir, output_dir, tokenizer)
    typer.echo(
        json.dumps(
            {
                "trajectory_count": summary["trajectory_count"],
                "tool_call_count": summary["tool_call_count"],
                "model_visible": {
                    pool: {
                        "reuse_ratio": summary["model_visible"][pool]["reuse_ratio"],
                        "reusable_tokens": summary["model_visible"][pool]["reusable_tokens"],
                    }
                    for pool in ("any_prior", "same_category", "exact_args")
                },
                "report": str(output_dir / "report.md"),
            },
            indent=2,
        )
    )


def run_analysis(run_dir: Path, output_dir: Path, tokenizer: Any) -> Record:
    """Analyze every trajectory below ``run_dir`` and write inspectable artifacts."""
    rows: list[Record] = []
    trajectories = sorted(run_dir.glob("**/*.traj.json"))
    for trajectory_path in trajectories:
        trajectory = json.loads(trajectory_path.read_text())
        rows.extend(
            analyze_trajectory(
                trajectory,
                tokenizer,
                trajectory_path=str(trajectory_path),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_call.jsonl", rows)
    largest_matches = sorted(
        rows,
        key=lambda row: int(row.get("model_visible_any_prior_lcp_tokens") or 0),
        reverse=True,
    )[:50]
    rows_by_call = {(row["trajectory"], row["call_index"]): row for row in rows}
    top_matches = []
    for row in largest_matches:
        match = rows_by_call.get((row["trajectory"], row.get("model_visible_any_prior_match_call_index")))
        top_matches.append(
            {
                **row,
                "model_visible_any_prior_match_command": match.get("command") if match else None,
            }
        )
    (output_dir / "top_matches.json").write_text(json.dumps(top_matches, indent=2) + "\n")
    summary = build_summary(run_dir, trajectories, rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(markdown_report(summary))
    return summary


def tokenizer_path_from_trajectory(path: Path) -> str:
    trajectory = json.loads(path.read_text())
    return str(trajectory.get("info", {}).get("config", {}).get("agent", {}).get("tokenizer_path") or "")


def load_tokenizer(path: str, *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        use_fast=True,
    )
    tokenizer.model_max_length = max(tokenizer.model_max_length, 10**12)
    return tokenizer


def analyze_trajectory(
    trajectory: Record,
    tokenizer: Any,
    *,
    trajectory_path: str = "",
) -> list[Record]:
    """Return one causal prefix-opportunity record for each tool output."""
    instance_id = str(trajectory.get("instance_id") or "")
    actions: dict[str, Record] = {}
    history: list[ToolOutput] = []
    rows: list[Record] = []

    for message_index, message in enumerate(trajectory.get("messages", [])):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            actions.update(actions_from_message(message))
            continue
        if message.get("role") not in {"tool", "user"}:
            continue

        call = tool_output_from_message(
            message,
            actions,
            tokenizer,
            call_index=len(history),
            message_index=message_index,
        )
        if call is None:
            continue

        pools = candidate_pools(call, history)
        row: Record = {
            "instance_id": instance_id,
            "trajectory": trajectory_path,
            "call_index": call.call_index,
            "message_index": call.message_index,
            "tool_call_id": call.tool_call_id,
            "tool_name": call.tool_name,
            "command": call.command,
            "command_category": call.command_category,
            "returncode": call.returncode,
            "render_kind": call.render_kind,
            "raw_output_chars": len(call.raw_output),
            "raw_output_tokens": len(call.raw_tokens),
            "model_visible_output_tokens": call.rendered_payload_tokens,
        }
        for pool_name, candidates in pools.items():
            row[f"{pool_name}_candidate_count"] = len(candidates)
            raw_lcp, raw_match = best_match(call, candidates, token_field="raw_tokens")
            visible_candidates = [candidate for candidate in candidates if candidate.render_kind == call.render_kind]
            visible_lcp, visible_match = best_match(
                call,
                visible_candidates,
                token_field="rendered_prefix_tokens",
            )
            row[f"raw_{pool_name}_lcp_tokens"] = raw_lcp
            add_match_fields(row, tokenizer, "raw", pool_name, raw_lcp, raw_match, call.raw_tokens)
            row[f"model_visible_{pool_name}_candidate_count"] = len(visible_candidates)
            row[f"model_visible_{pool_name}_lcp_tokens"] = visible_lcp
            add_match_fields(
                row,
                tokenizer,
                "model_visible",
                pool_name,
                visible_lcp,
                visible_match,
                call.rendered_prefix_tokens,
            )

        rows.append(row)
        history.append(call)

    return rows


def actions_from_message(message: Record) -> dict[str, Record]:
    tool_names = {
        str(tool_call.get("id") or ""): str((tool_call.get("function") or {}).get("name") or "")
        for tool_call in message.get("tool_calls") or []
        if isinstance(tool_call, dict)
    }
    actions: dict[str, Record] = {}
    for action in (message.get("extra") or {}).get("actions") or []:
        if not isinstance(action, dict):
            continue
        call_id = str(action.get("tool_call_id") or "")
        if call_id:
            actions[call_id] = {**action, "tool_name": tool_names.get(call_id, "")}
    return actions


def tool_output_from_message(
    message: Record,
    actions: dict[str, Record],
    tokenizer: Any,
    *,
    call_index: int,
    message_index: int,
) -> ToolOutput | None:
    call_id = str(message.get("tool_call_id") or "")
    action = actions.get(call_id)
    if action is None:
        return None

    extra = message.get("extra") or {}
    raw_output = str(extra.get("raw_output") or "")
    metric = metric_from_message(extra, call_id)
    rendered_prefix, render_kind, rendered_payload_tokens = rendered_payload(message, raw_output, tokenizer)
    command = str(action.get("command") or "")
    tool_name = str(action.get("tool_name") or "")
    exact_args = {key: value for key, value in action.items() if key not in {"tool_call_id", "tool_name"}}

    return ToolOutput(
        call_index=call_index,
        tool_call_id=call_id,
        tool_name=tool_name,
        command=command,
        command_category=str(metric.get("command_category") or pipeline_category(command) or tool_name),
        exact_args_key=f"{tool_name}:{json.dumps(exact_args, sort_keys=True, separators=(',', ':'))}",
        raw_output=raw_output,
        raw_tokens=encode(tokenizer, raw_output),
        rendered_prefix_tokens=encode_payload(tokenizer, rendered_prefix, render_kind),
        rendered_payload_tokens=rendered_payload_tokens,
        render_kind=render_kind,
        returncode=integer_or_none(extra.get("returncode")),
        message_index=message_index,
    )


def metric_from_message(extra: Record, call_id: str) -> Record:
    metrics = ((extra.get("token_timing") or {}).get("tool_calls") or [])
    for metric in metrics:
        if isinstance(metric, dict) and str(metric.get("tool_call_id") or call_id) == call_id:
            return metric
    return {}


def rendered_payload(message: Record, raw_output: str, tokenizer: Any) -> tuple[str, str, int]:
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    if "<output_head>" not in text:
        return raw_output, "full", len(encode_payload(tokenizer, raw_output, "full"))

    head = raw_output[:5000]
    tail = raw_output[-5000:]
    visible_tokens = len(encode_payload(tokenizer, head, "truncated")) + len(
        encode_payload(tokenizer, tail, "truncated_tail")
    )
    return head, "truncated", visible_tokens


def encode_payload(tokenizer: Any, payload: str, render_kind: str) -> list[int]:
    boundaries = {
        "full": PAYLOAD_BOUNDARY,
        "truncated": TRUNCATED_PAYLOAD_BOUNDARY,
        "truncated_tail": TRUNCATED_TAIL_BOUNDARY,
    }
    boundary = boundaries[render_kind]
    text = boundary + payload
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [
        token_id
        for token_id, (_, end) in zip(encoded["input_ids"], encoded["offset_mapping"])
        if end > len(boundary)
    ]


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text or "", add_special_tokens=False))


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=False))


def candidate_pools(call: ToolOutput, history: list[ToolOutput]) -> dict[str, list[ToolOutput]]:
    return {
        "any_prior": list(history),
        "same_category": [candidate for candidate in history if candidate.command_category == call.command_category],
        "exact_args": [candidate for candidate in history if candidate.exact_args_key == call.exact_args_key],
    }


def best_match(
    call: ToolOutput,
    candidates: list[ToolOutput],
    *,
    token_field: str,
) -> tuple[int, ToolOutput | None]:
    best_lcp = 0
    best_candidate = None
    current_tokens = getattr(call, token_field)
    for candidate in candidates:
        lcp = longest_common_prefix(current_tokens, getattr(candidate, token_field))
        if lcp > best_lcp:
            best_lcp = lcp
            best_candidate = candidate
    return best_lcp, best_candidate


def add_match_fields(
    row: Record,
    tokenizer: Any,
    view: str,
    pool_name: str,
    lcp: int,
    match: ToolOutput | None,
    current_tokens: list[int],
) -> None:
    prefix = f"{view}_{pool_name}"
    row[f"{prefix}_match_call_index"] = match.call_index if match else None
    row[f"{prefix}_match_tool_call_id"] = match.tool_call_id if match else None
    if view == "model_visible" and pool_name == "any_prior":
        row[f"{prefix}_prefix_preview"] = decode(tokenizer, current_tokens[:lcp])[:500] if lcp else ""


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def integer_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def write_jsonl(path: Path, rows: list[Record]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_summary(run_dir: Path, trajectories: list[Path], rows: list[Record]) -> Record:
    summary: Record = {
        "run_dir": str(run_dir),
        "trajectory_count": len(trajectories),
        "tool_call_count": len(rows),
    }
    for view, output_field in (
        ("model_visible", "model_visible_output_tokens"),
        ("raw", "raw_output_tokens"),
    ):
        summary[view] = {
            pool: summarize_metric(
                rows,
                output_field,
                f"{view}_{pool}_lcp_tokens",
                f"{view}_{pool}_candidate_count" if view == "model_visible" else f"{pool}_candidate_count",
            )
            for pool in ("any_prior", "same_category", "exact_args")
        }
    summary["rendering"] = {
        "full_calls": sum(row.get("render_kind") == "full" for row in rows),
        "truncated_calls": sum(row.get("render_kind") == "truncated" for row in rows),
        "full_output_tokens": sum(
            int(row.get("model_visible_output_tokens") or 0) for row in rows if row.get("render_kind") == "full"
        ),
        "truncated_output_tokens": sum(
            int(row.get("model_visible_output_tokens") or 0)
            for row in rows
            if row.get("render_kind") == "truncated"
        ),
    }
    summary["command_categories"] = category_breakdown(rows)
    summary["output_length_buckets"] = output_length_breakdown(rows)
    return summary


def summarize_metric(
    rows: list[Record],
    output_field: str,
    lcp_field: str,
    candidate_count_field: str,
    *,
    include_ci: bool = True,
) -> Record:
    output_tokens = sum(int(row.get(output_field) or 0) for row in rows)
    reusable_tokens = sum(int(row.get(lcp_field) or 0) for row in rows)
    eligible_rows = [row for row in rows if int(row.get(candidate_count_field) or 0) > 0]
    eligible_output_tokens = sum(int(row.get(output_field) or 0) for row in eligible_rows)
    lcp_values = [int(row.get(lcp_field) or 0) for row in rows]
    by_trajectory: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        by_trajectory[str(row.get("trajectory") or "")].append(row)
    trajectory_ratios = []
    for trajectory_rows in by_trajectory.values():
        denominator = sum(int(row.get(output_field) or 0) for row in trajectory_rows)
        numerator = sum(int(row.get(lcp_field) or 0) for row in trajectory_rows)
        if denominator:
            trajectory_ratios.append(numerator / denominator)

    summary = {
        "output_tokens": output_tokens,
        "reusable_tokens": reusable_tokens,
        "reuse_ratio": reusable_tokens / output_tokens if output_tokens else None,
        "eligible_calls": len(eligible_rows),
        "eligible_output_tokens": eligible_output_tokens,
        "eligible_reuse_ratio": reusable_tokens / eligible_output_tokens if eligible_output_tokens else None,
        "positive_lcp_calls": sum(1 for row in rows if int(row.get(lcp_field) or 0) > 0),
        "lcp_tokens": numeric_summary(lcp_values),
        "trajectory_reuse_ratio": numeric_summary(trajectory_ratios),
        "thresholds": {
            str(threshold): threshold_summary(lcp_values, len(eligible_rows), threshold)
            for threshold in (1, 8, 32, 64, 128, 256, 512)
        },
    }
    if include_ci:
        summary["reuse_ratio_ci95"] = bootstrap_reuse_ratio(rows, output_field, lcp_field)
    return summary


def bootstrap_reuse_ratio(
    rows: list[Record],
    output_field: str,
    lcp_field: str,
    *,
    samples: int = 1000,
) -> Record:
    by_trajectory: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        totals = by_trajectory[str(row.get("trajectory") or "")]
        totals[0] += int(row.get(output_field) or 0)
        totals[1] += int(row.get(lcp_field) or 0)
    groups = list(by_trajectory.values())
    if not groups:
        return {"low": None, "high": None}
    if len(groups) == 1:
        denominator, numerator = groups[0]
        ratio = numerator / denominator if denominator else None
        return {"low": ratio, "high": ratio}

    rng = random.Random(0)
    ratios = []
    for _ in range(samples):
        sampled = rng.choices(groups, k=len(groups))
        denominator = sum(group[0] for group in sampled)
        numerator = sum(group[1] for group in sampled)
        if denominator:
            ratios.append(numerator / denominator)
    ratios.sort()
    return {
        "low": percentile(ratios, 2.5) if ratios else None,
        "high": percentile(ratios, 97.5) if ratios else None,
    }


def category_breakdown(rows: list[Record]) -> Record:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("command_category") or "unknown")].append(row)
    return {
        category: {
            "tool_call_count": len(category_rows),
            "model_visible": {
                pool: compact_metric(
                    summarize_metric(
                        category_rows,
                        "model_visible_output_tokens",
                        f"model_visible_{pool}_lcp_tokens",
                        f"model_visible_{pool}_candidate_count",
                        include_ci=False,
                    )
                )
                for pool in ("any_prior", "same_category", "exact_args")
            },
        }
        for category, category_rows in sorted(grouped.items())
    }


def compact_metric(metric: Record) -> Record:
    return {
        key: metric[key]
        for key in (
            "output_tokens",
            "reusable_tokens",
            "reuse_ratio",
            "eligible_calls",
            "eligible_output_tokens",
            "eligible_reuse_ratio",
            "positive_lcp_calls",
        )
    }


def output_length_breakdown(rows: list[Record]) -> Record:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        grouped[output_length_bucket(int(row.get("model_visible_output_tokens") or 0))].append(row)
    breakdown = {}
    for label in ("0", "1-31", "32-127", "128-511", "512-2047", "2048+"):
        bucket_rows = grouped.get(label, [])
        metric = summarize_metric(
            bucket_rows,
            "model_visible_output_tokens",
            "model_visible_any_prior_lcp_tokens",
            "model_visible_any_prior_candidate_count",
            include_ci=False,
        )
        breakdown[label] = {"tool_call_count": len(bucket_rows), **metric}
    return breakdown


def output_length_bucket(tokens: int) -> str:
    if tokens == 0:
        return "0"
    if tokens < 32:
        return "1-31"
    if tokens < 128:
        return "32-127"
    if tokens < 512:
        return "128-511"
    if tokens < 2048:
        return "512-2047"
    return "2048+"


def threshold_summary(values: list[int], eligible_calls: int, threshold: int) -> Record:
    calls = sum(value >= threshold for value in values)
    return {
        "calls": calls,
        "all_call_rate": calls / len(values) if values else None,
        "eligible_call_rate": calls / eligible_calls if eligible_calls else None,
    }


def numeric_summary(values: list[float | int]) -> Record:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "p25": percentile(ordered, 25),
        "median": statistics.median(ordered),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
    }


def percentile(ordered: list[float | int], percentage: float) -> float | int:
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


REPORT_TEMPLATE = """# BranchFill Prefix Opportunity

- Trajectories: {{ summary.trajectory_count }}
- Tool calls: {{ summary.tool_call_count }}

## Findings

- The causal any-prior oracle reuses {{ ratio(any_prior.reuse_ratio) }} of model-visible payload tokens (trajectory-bootstrap 95% CI {{ ratio(any_prior.reuse_ratio_ci95.low) }}–{{ ratio(any_prior.reuse_ratio_ci95.high) }}).
- Restricting candidates to the same command category retains {{ ratio(divide(same_category.reusable_tokens, any_prior.reusable_tokens)) }} of those oracle-reusable tokens.
- Exact-argument history exists for {{ exact_args.eligible_calls }} calls ({{ ratio(divide(exact_args.eligible_calls, summary.tool_call_count)) }} of all calls). Within those eligible calls it reuses {{ ratio(exact_args.eligible_reuse_ratio) }} of payload tokens, or {{ ratio(exact_args.reuse_ratio) }} globally.
- Among exact-argument-eligible calls, {{ exact_32.calls }} ({{ ratio(exact_32.eligible_call_rate) }}) reach at least 32 exact prefix tokens.

## Oracle reuse

| View | Candidate pool | Output tokens | Reusable tokens | Reuse ratio |
|---|---:|---:|---:|---:|
{% for view in views %}{% for pool in pools %}{% set metric = summary[view][pool] %}| {{ view }} | {{ pool }} | {{ metric.output_tokens }} | {{ metric.reusable_tokens }} | {{ ratio(metric.reuse_ratio) }} |
{% endfor %}{% endfor %}
## Model-visible LCP coverage

| Candidate pool | Eligible calls | Positive LCP calls | Median LCP | P90 LCP | P99 LCP |
|---|---:|---:|---:|---:|---:|
{% for pool in pools %}{% set metric = summary.model_visible[pool] %}| {{ pool }} | {{ metric.eligible_calls }} | {{ metric.positive_lcp_calls }} | {{ number(metric.lcp_tokens.median) }} | {{ number(metric.lcp_tokens.p90) }} | {{ number(metric.lcp_tokens.p99) }} |
{% endfor %}
## Per-trajectory reuse ratio

| Candidate pool | P25 | Median | P75 | P90 |
|---|---:|---:|---:|---:|
{% for pool in pools %}{% set distribution = summary.model_visible[pool].trajectory_reuse_ratio %}| {{ pool }} | {{ ratio(distribution.p25) }} | {{ ratio(distribution.median) }} | {{ ratio(distribution.p75) }} | {{ ratio(distribution.p90) }} |
{% endfor %}
## Output-length breakdown

| Output tokens | Calls | Output tokens | Any-prior reusable tokens | Reuse ratio |
|---|---:|---:|---:|---:|
{% for label, metric in summary.output_length_buckets.items() %}| {{ label }} | {{ metric.tool_call_count }} | {{ metric.output_tokens }} | {{ metric.reusable_tokens }} | {{ ratio(metric.reuse_ratio) }} |
{% endfor %}
## Rendering

- Full outputs: {{ summary.rendering.full_calls }} calls / {{ summary.rendering.full_output_tokens }} payload tokens
- Truncated outputs: {{ summary.rendering.truncated_calls }} calls / {{ summary.rendering.truncated_output_tokens }} visible head-and-tail payload tokens
- Truncated-output reuse is conservatively capped at the visible 5,000-character head.

## Largest command categories

| Category | Calls | Output tokens | Any-prior reuse | Exact-args reuse |
|---|---:|---:|---:|---:|
{% for category, category_summary in categories %}{% set category_any = category_summary.model_visible.any_prior %}{% set category_exact = category_summary.model_visible.exact_args %}| `{{ category }}` | {{ category_summary.tool_call_count }} | {{ category_any.output_tokens }} | {{ ratio(category_any.reuse_ratio) }} | {{ ratio(category_exact.reuse_ratio) }} |
{% endfor %}
## Method

For each tool call, candidates are restricted to completed earlier calls in the same trajectory. The oracle chooses the candidate with the longest exact target-tokenizer prefix. No text normalization or future/cross-trajectory output is used. Candidate pools are all prior calls, the same recorded command category, and exact tool arguments. Calls without a candidate remain in the denominator.

The model-visible metric requires the candidate and current output to use the same full/truncated rendering form. This preserves the exact formatter boundary and token positions needed for KV reuse. The raw metric compares every causal candidate regardless of rendering form.

`per_call.jsonl` contains every comparison result. `top_matches.json` contains the 50 largest model-visible any-prior matches for manual inspection.
"""


def markdown_report(summary: Record) -> str:
    categories = sorted(
        summary["command_categories"].items(),
        key=lambda item: item[1]["model_visible"]["any_prior"]["output_tokens"],
        reverse=True,
    )[:20]
    return Template(REPORT_TEMPLATE).render(
        summary=summary,
        any_prior=summary["model_visible"]["any_prior"],
        same_category=summary["model_visible"]["same_category"],
        exact_args=summary["model_visible"]["exact_args"],
        exact_32=summary["model_visible"]["exact_args"]["thresholds"]["32"],
        views=("model_visible", "raw"),
        pools=("any_prior", "same_category", "exact_args"),
        categories=categories,
        ratio=format_ratio,
        divide=safe_ratio,
        number=format_number,
    )


def format_ratio(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if isinstance(value, float) and not value.is_integer() else str(int(value))


if __name__ == "__main__":
    app()
