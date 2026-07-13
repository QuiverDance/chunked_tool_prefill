"""Measure causal cross-turn response and next-tool reuse opportunities."""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.run.extra.branchfill_prefix_opportunity import (
    TOP_K,
    ToolOutput,
    actions_from_message,
    load_tokenizer,
    longest_common_prefix,
    rank_candidates,
    tokenizer_path_from_trajectory,
    tool_output_from_message,
)

Record = dict[str, Any]
POLICIES = ("recent", "command_similarity")
app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ResponseDraft:
    message_index: int
    tokens: list[int]
    reasoning_tokens: list[int]
    visible_tokens: list[int]
    action_key: str
    tool_call_ids: tuple[str, ...]
    prompt_tokens: int | None
    ttft_s: float | None
    decode_s: float | None


@dataclass(frozen=True)
class Interaction:
    output: ToolOutput
    action_message_index: int
    response: ResponseDraft
    current_tool_duration_s: float | None
    next_tool_duration_s: float | None


@app.command()
def main(
    run_dir: Path,
    output_dir: Path = typer.Option(..., "--output-dir"),
    tokenizer_path: str | None = typer.Option(None, "--tokenizer-path"),
    allow_tokenizer_download: bool = typer.Option(False, "--allow-tokenizer-download"),
    use_recorded_completion_length: bool = typer.Option(
        True,
        "--use-recorded-completion-length/--ignore-recorded-completion-length",
    ),
) -> None:
    """Analyze historical response drafts and next-tool predictions."""
    trajectories = sorted(run_dir.glob("**/*.traj.json"))
    if not trajectories:
        raise typer.BadParameter(f"No trajectory files found below {run_dir}")

    tokenizer_path = tokenizer_path or tokenizer_path_from_trajectory(trajectories[0])
    if not tokenizer_path:
        raise typer.BadParameter("No tokenizer path was supplied or recorded in the trajectories")
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=not allow_tokenizer_download)
    summary = run_analysis(
        run_dir,
        output_dir,
        tokenizer,
        tokenizer_path=tokenizer_path,
        use_recorded_completion_length=use_recorded_completion_length,
    )
    typer.echo(json.dumps({"trajectory_count": summary["trajectory_count"], "report": str(output_dir / "report.md")}, indent=2))


def run_analysis(
    run_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    *,
    tokenizer_path: str = "",
    use_recorded_completion_length: bool = True,
) -> Record:
    """Analyze all trajectories and write per-call evidence plus aggregate metrics."""
    rows = []
    trajectories = sorted(run_dir.glob("**/*.traj.json"))
    for path in trajectories:
        trajectory = json.loads(path.read_text())
        rows.extend(
            analyze_trajectory(
                trajectory,
                tokenizer,
                trajectory_path=str(path),
                use_recorded_completion_length=use_recorded_completion_length,
            )
        )

    summary = build_summary(
        run_dir,
        trajectories,
        rows,
        tokenizer_path=tokenizer_path,
        use_recorded_completion_length=use_recorded_completion_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "per_call.jsonl.gz", "wt", compresslevel=6) as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    top_response_matches = sorted(
        rows,
        key=lambda row: int(row["policy_command_similarity_k4_response_lcp_tokens"]),
        reverse=True,
    )[:50]
    top_tool_time_hits = sorted(
        (
            row
            for row in rows
            if row["policy_command_similarity_k4_next_tool_exact"] and row.get("next_tool_duration_s") is not None
        ),
        key=lambda row: float(row["next_tool_duration_s"]),
        reverse=True,
    )[:50]
    (output_dir / "top_response_matches.json").write_text(json.dumps(top_response_matches, indent=2) + "\n")
    (output_dir / "top_tool_time_hits.json").write_text(json.dumps(top_tool_time_hits, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(markdown_report(summary))
    return summary


def analyze_trajectory(
    trajectory: Record,
    tokenizer: Any,
    *,
    trajectory_path: str = "",
    use_recorded_completion_length: bool = True,
) -> list[Record]:
    """Return one response-opportunity row for each model re-entry after tools."""
    messages = trajectory.get("messages") or []
    actions: dict[str, Record] = {}
    action_message_indices: dict[str, int] = {}
    outputs: list[ToolOutput] = []
    durations: dict[str, float] = {}

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            message_actions = actions_from_message(message)
            actions.update(message_actions)
            action_message_indices.update({call_id: message_index for call_id in message_actions})
            continue
        if message.get("role") not in {"tool", "user"}:
            continue
        output = tool_output_from_message(
            message,
            actions,
            tokenizer,
            call_index=len(outputs),
            message_index=message_index,
        )
        if output is None:
            continue
        outputs.append(output)
        duration = tool_duration(message, output.tool_call_id)
        if duration is not None:
            durations[output.tool_call_id] = duration

    next_assistant = next_assistant_indices(messages)
    response_owner = response_owner_indices(outputs, next_assistant)
    interactions = []
    for output in outputs:
        response_index = next_assistant.get(output.message_index)
        if response_index is None or response_owner.get(response_index) != output.message_index:
            continue
        response = response_draft(
            messages[response_index],
            response_index,
            tokenizer,
            use_recorded_completion_length=use_recorded_completion_length,
        )
        if not response.tokens:
            continue
        interactions.append(
            Interaction(
                output=output,
                action_message_index=action_message_indices.get(output.tool_call_id, -1),
                response=response,
                current_tool_duration_s=durations.get(output.tool_call_id),
                next_tool_duration_s=sum_tool_durations(response.tool_call_ids, durations),
            )
        )

    rows = []
    for current in interactions:
        candidates = [
            candidate
            for candidate in interactions
            if candidate.response.message_index <= current.action_message_index
        ]
        row: Record = {
            "instance_id": str(trajectory.get("instance_id") or ""),
            "trajectory": trajectory_path,
            "call_index": current.output.call_index,
            "message_index": current.output.message_index,
            "command": current.output.command,
            "response_tokens": len(current.response.tokens),
            "reasoning_tokens": len(current.response.reasoning_tokens),
            "visible_response_tokens": len(current.response.visible_tokens),
            "response_decode_s": current.response.decode_s,
            "has_next_tool": bool(current.response.action_key),
            "next_tool_action_key": current.response.action_key,
            "current_tool_duration_s": current.current_tool_duration_s,
            "next_tool_duration_s": current.next_tool_duration_s,
        }
        for policy in POLICIES:
            ranked = rank_interactions(policy, current, candidates)
            for k in TOP_K:
                add_policy_result(row, policy, k, current, distinct_branches(ranked, k))
        add_policy_result(row, "any_prior", 0, current, candidates)
        rows.append(row)
    return rows


def next_assistant_indices(messages: list[Any]) -> dict[int, int]:
    result = {}
    next_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            next_index = index
        elif next_index is not None:
            result[index] = next_index
    return result


def response_owner_indices(outputs: list[ToolOutput], next_assistant: dict[int, int]) -> dict[int, int]:
    owners = {}
    for output in outputs:
        response_index = next_assistant.get(output.message_index)
        if response_index is not None:
            owners[response_index] = max(owners.get(response_index, -1), output.message_index)
    return owners


def response_draft(
    message: Record,
    message_index: int,
    tokenizer: Any,
    *,
    use_recorded_completion_length: bool,
) -> ResponseDraft:
    tokens = assistant_completion_tokens(
        message,
        tokenizer,
        use_recorded_completion_length=use_recorded_completion_length,
    )
    reasoning_length = reasoning_token_count(
        message,
        tokenizer,
        use_recorded_completion_length=use_recorded_completion_length,
    )
    reasoning_length = min(reasoning_length, len(tokens))
    tool_calls = message.get("tool_calls") or []
    call_ids = tuple(str(call.get("id") or "") for call in tool_calls if isinstance(call, dict))
    timing = ((message.get("extra") or {}).get("token_timing") or {}).get("model_call") or {}
    prompt_tokens = timing.get("prompt_tokens")
    ttft_s = timing.get("ttft_s")
    decode_s = timing.get("decode_s")
    return ResponseDraft(
        message_index=message_index,
        tokens=tokens,
        reasoning_tokens=tokens[:reasoning_length],
        visible_tokens=tokens[reasoning_length:],
        action_key=action_key(tool_calls),
        tool_call_ids=call_ids,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        ttft_s=float(ttft_s) if ttft_s is not None else None,
        decode_s=float(decode_s) if decode_s is not None else None,
    )


def assistant_completion_tokens(
    message: Record,
    tokenizer: Any,
    *,
    use_recorded_completion_length: bool,
) -> list[int]:
    assistant = {key: copy.deepcopy(value) for key, value in message.items() if key in {"role", "content", "tool_calls"}}
    prompt_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "response draft"},
    ]
    prompt = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full = render_chat(tokenizer, [*prompt_messages, assistant], add_generation_prompt=False)
    if not full.startswith(prompt):
        return fallback_completion_tokens(
            message,
            tokenizer,
            use_recorded_completion_length=use_recorded_completion_length,
        )

    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    full = prompt + reasoning + full[len(prompt) :]
    prompt_tokens = encode(tokenizer, prompt)
    full_tokens = encode(tokenizer, full)
    prefix = longest_common_prefix(prompt_tokens, full_tokens)
    completion = full_tokens[prefix:]
    recorded_length = completion_token_count(message) if use_recorded_completion_length else None
    return completion[:recorded_length] if recorded_length is not None else completion


def render_chat(tokenizer: Any, messages: list[Record], *, add_generation_prompt: bool) -> str:
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tools=[BASH_TOOL],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )
    except (TypeError, ValueError):
        safe_messages = copy.deepcopy(messages)
        for message in safe_messages:
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, str):
                    try:
                        call["function"]["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
        return str(
            tokenizer.apply_chat_template(
                safe_messages,
                tools=[BASH_TOOL],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )


def fallback_completion_tokens(
    message: Record,
    tokenizer: Any,
    *,
    use_recorded_completion_length: bool,
) -> list[int]:
    parts = [str(message.get("reasoning_content") or message.get("reasoning") or ""), str(message.get("content") or "")]
    parts.extend(
        json.dumps(canonical_tool_call(call), sort_keys=True)
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    )
    tokens = encode(tokenizer, "\n".join(parts))
    recorded_length = completion_token_count(message) if use_recorded_completion_length else None
    return tokens[:recorded_length] if recorded_length is not None else tokens


def completion_token_count(message: Record) -> int | None:
    extra = message.get("extra") or {}
    model_call = ((extra.get("token_timing") or {}).get("model_call") or {})
    value = model_call.get("completion_tokens")
    if value is None:
        value = ((extra.get("response") or {}).get("usage") or {}).get("completion_tokens")
    return int(value) if value is not None else None


def reasoning_token_count(
    message: Record,
    tokenizer: Any,
    *,
    use_recorded_completion_length: bool,
) -> int:
    usage = (((message.get("extra") or {}).get("response") or {}).get("usage") or {})
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    if use_recorded_completion_length and value is not None:
        return int(value)
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    return len(encode(tokenizer, reasoning))


def action_key(tool_calls: list[Any]) -> str:
    calls = [canonical_tool_call(call) for call in tool_calls if isinstance(call, dict)]
    return json.dumps(calls, separators=(",", ":")) if calls else ""


def action_commands(key: str) -> list[str]:
    commands = []
    for call in json.loads(key or "[]"):
        arguments = call.get("arguments") or {}
        if call.get("name") == "bash" and isinstance(arguments, dict) and arguments.get("command"):
            commands.append(str(arguments["command"]))
    return commands


def canonical_tool_call(call: Record) -> Record:
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {"name": str(function.get("name") or ""), "arguments": arguments}


def tool_duration(message: Record, call_id: str) -> float | None:
    metrics = (((message.get("extra") or {}).get("token_timing") or {}).get("tool_calls") or [])
    matching = [
        metric
        for metric in metrics
        if isinstance(metric, dict) and str(metric.get("tool_call_id") or "") == call_id
    ]
    if not matching and len(metrics) == 1 and isinstance(metrics[0], dict) and not metrics[0].get("tool_call_id"):
        matching = [metrics[0]]
    for metric in matching:
        if not isinstance(metric, dict):
            continue
        if metric.get("duration_s") is not None:
            return float(metric["duration_s"])
    return None


def sum_tool_durations(call_ids: tuple[str, ...], durations: dict[str, float]) -> float | None:
    if not call_ids or any(call_id not in durations for call_id in call_ids):
        return None
    return sum(durations[call_id] for call_id in call_ids)


def rank_interactions(policy: str, current: Interaction, candidates: list[Interaction]) -> list[Interaction]:
    if policy == "recent":
        return list(reversed(candidates))
    ranked_outputs = rank_candidates("command_similarity", current.output, [candidate.output for candidate in candidates])
    by_index = {candidate.output.call_index: candidate for candidate in candidates}
    return [by_index[output.call_index] for output in ranked_outputs]


def distinct_branches(candidates: list[Interaction], k: int) -> list[Interaction]:
    selected = []
    seen = set()
    for candidate in candidates:
        key = (candidate.output.raw_output, tuple(candidate.response.tokens), candidate.response.action_key)
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) == k:
            break
    return selected


def add_policy_result(row: Record, policy: str, k: int, current: Interaction, candidates: list[Interaction]) -> None:
    prefix = f"policy_{policy}_k{k}"
    best = max(
        candidates,
        key=lambda candidate: (
            longest_common_prefix(current.response.tokens, candidate.response.tokens),
            observation_lcp(current, candidate),
        ),
        default=None,
    )
    response_lcp = longest_common_prefix(current.response.tokens, best.response.tokens) if best else 0
    reasoning_lcp = max(
        (
            longest_common_prefix(current.response.reasoning_tokens, candidate.response.reasoning_tokens)
            for candidate in candidates
        ),
        default=0,
    )
    visible_lcp = max(
        (
            longest_common_prefix(current.response.visible_tokens, candidate.response.visible_tokens)
            for candidate in candidates
        ),
        default=0,
    )
    output_lcp = observation_lcp(current, best) if best else 0
    full_output_match = bool(best and model_visible_output_equal(current, best))
    any_full_output_match = any(model_visible_output_equal(current, candidate) for candidate in candidates)
    tool_match = next(
        (
            candidate
            for candidate in candidates
            if current.response.action_key and candidate.response.action_key == current.response.action_key
        ),
        None,
    )
    row[f"{prefix}_selected_count"] = len(candidates)
    row[f"{prefix}_response_lcp_tokens"] = response_lcp
    row[f"{prefix}_reasoning_lcp_tokens"] = reasoning_lcp
    row[f"{prefix}_visible_response_lcp_tokens"] = visible_lcp
    row[f"{prefix}_observation_lcp_tokens"] = output_lcp
    row[f"{prefix}_full_observation_match"] = any_full_output_match
    row[f"{prefix}_salvage_group"] = salvage_group(full_output_match, output_lcp, response_lcp)
    row[f"{prefix}_next_tool_exact"] = tool_match is not None
    row[f"{prefix}_next_tool_match_call_index"] = tool_match.output.call_index if tool_match else None
    row[f"{prefix}_next_tool_match_commands"] = action_commands(tool_match.response.action_key) if tool_match else []
    row[f"{prefix}_match_call_index"] = best.output.call_index if best else None
    row[f"{prefix}_match_command"] = best.output.command if best else None
    joint_match = max(candidates, key=lambda candidate: joint_opportunity_s(current, candidate)[0], default=None)
    joint_s, prefill_s, decode_s, tool_s = joint_opportunity_s(current, joint_match)
    row[f"{prefix}_joint_opportunity_s"] = joint_s
    row[f"{prefix}_joint_prefill_s"] = prefill_s
    row[f"{prefix}_joint_decode_s"] = decode_s
    row[f"{prefix}_joint_tool_s"] = tool_s
    row[f"{prefix}_joint_match_call_index"] = joint_match.output.call_index if joint_match else None


def observation_lcp(current: Interaction, candidate: Interaction) -> int:
    if current.output.render_kind != candidate.output.render_kind:
        return 0
    return longest_common_prefix(current.output.rendered_prefix_tokens, candidate.output.rendered_prefix_tokens)


def model_visible_output_equal(current: Interaction, candidate: Interaction) -> bool:
    if current.output.render_kind != candidate.output.render_kind:
        return False
    if current.output.render_kind == "full":
        return current.output.raw_output == candidate.output.raw_output
    return (
        len(current.output.raw_output) == len(candidate.output.raw_output)
        and current.output.raw_output[:5000] == candidate.output.raw_output[:5000]
        and current.output.raw_output[-5000:] == candidate.output.raw_output[-5000:]
    )


def joint_opportunity_s(current: Interaction, candidate: Interaction | None) -> tuple[float, float, float, float]:
    if candidate is None:
        return 0.0, 0.0, 0.0, 0.0

    response = current.response
    output_lcp = observation_lcp(current, candidate)
    response_lcp = longest_common_prefix(response.tokens, candidate.response.tokens)
    prefill_s = proportional_time(response.ttft_s, output_lcp, response.prompt_tokens)
    decode_s = proportional_time(response.decode_s, response_lcp, len(response.tokens))
    tool_s = 0.0
    if response.action_key and response.action_key == candidate.response.action_key:
        if current.current_tool_duration_s is not None and current.next_tool_duration_s is not None:
            tool_s = min(current.current_tool_duration_s, current.next_tool_duration_s)
    return prefill_s + decode_s + tool_s, prefill_s, decode_s, tool_s


def proportional_time(duration_s: float | None, covered_tokens: int, total_tokens: int | None) -> float:
    if duration_s is None or not total_tokens:
        return 0.0
    return duration_s * min(covered_tokens, total_tokens) / total_tokens


def salvage_group(full_match: bool, observation_lcp_tokens: int, response_lcp_tokens: int) -> str:
    if not response_lcp_tokens:
        return "no_response_hit"
    if full_match:
        return "full_observation_response_hit"
    if observation_lcp_tokens:
        return "partial_observation_response_hit"
    return "zero_observation_response_hit"


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text or "", add_special_tokens=False))


def build_summary(
    run_dir: Path,
    trajectories: list[Path],
    rows: list[Record],
    *,
    tokenizer_path: str,
    use_recorded_completion_length: bool,
) -> Record:
    trajectory_timings = {str(path): trajectory_e2e_s(path) for path in trajectories}
    measured_trajectories = {path for path, e2e_s in trajectory_timings.items() if e2e_s is not None}
    trace_e2e_s = sum(e2e_s for e2e_s in trajectory_timings.values() if e2e_s is not None)
    summary: Record = {
        "run_dir": str(run_dir),
        "tokenizer_path": tokenizer_path,
        "trace_models": sorted(set(filter(None, (trajectory_model(path) for path in trajectories)))),
        "trajectory_count": len(trajectories),
        "response_boundary_count": len(rows),
        "response_tokens": sum(int(row["response_tokens"]) for row in rows),
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
        "visible_response_tokens": sum(int(row["visible_response_tokens"]) for row in rows),
        "used_recorded_completion_length": use_recorded_completion_length,
        "trace_e2e_s": trace_e2e_s,
        "e2e_measured_trajectory_count": len(measured_trajectories),
        "policy_frontier": {},
    }
    for policy in POLICIES:
        summary["policy_frontier"][policy] = {}
        for k in TOP_K:
            prefix = f"policy_{policy}_k{k}"
            summary["policy_frontier"][policy][str(k)] = summarize_policy(
                rows, prefix, trace_e2e_s, measured_trajectories
            )
    summary["policy_frontier"]["any_prior"] = {
        "all": summarize_policy(rows, "policy_any_prior_k0", trace_e2e_s, measured_trajectories)
    }
    return summary


def trajectory_e2e_s(path: Path) -> float | None:
    trajectory = json.loads(path.read_text())
    value = (((trajectory.get("info") or {}).get("token_timing") or {}).get("problem") or {}).get("e2e_s")
    return float(value) if value is not None else None


def trajectory_model(path: Path) -> str:
    trajectory = json.loads(path.read_text())
    model = (((trajectory.get("info") or {}).get("config") or {}).get("model") or {}).get("model_name")
    return str(model or "")


def summarize_policy(
    rows: list[Record],
    prefix: str,
    trace_e2e_s: float,
    measured_trajectories: set[str],
) -> Record:
    response_lcp = sum(int(row[f"{prefix}_response_lcp_tokens"]) for row in rows)
    reasoning_lcp = sum(int(row[f"{prefix}_reasoning_lcp_tokens"]) for row in rows)
    visible_lcp = sum(int(row[f"{prefix}_visible_response_lcp_tokens"]) for row in rows)
    response_tokens = sum(int(row["response_tokens"]) for row in rows)
    reasoning_tokens = sum(int(row["reasoning_tokens"]) for row in rows)
    visible_tokens = sum(int(row["visible_response_tokens"]) for row in rows)
    next_tool_rows = [row for row in rows if row["has_next_tool"] and row.get("next_tool_duration_s") is not None]
    covered_tool_time = sum(
        float(row["next_tool_duration_s"])
        for row in next_tool_rows
        if row[f"{prefix}_next_tool_exact"]
    )
    total_tool_time = sum(float(row["next_tool_duration_s"]) for row in next_tool_rows)
    lcp_values = [int(row[f"{prefix}_response_lcp_tokens"]) for row in rows]
    positive_lcp_values = sorted(value for value in lcp_values if value)
    measured_decode_rows = [row for row in rows if row.get("response_decode_s") is not None and row["response_tokens"]]
    saved_decode_s = sum(
        float(row["response_decode_s"]) * int(row[f"{prefix}_response_lcp_tokens"]) / int(row["response_tokens"])
        for row in measured_decode_rows
    )
    decode_s = sum(float(row["response_decode_s"]) for row in measured_decode_rows)
    measured_rows = [row for row in rows if row["trajectory"] in measured_trajectories]
    joint_prefill_s = sum(float(row[f"{prefix}_joint_prefill_s"]) for row in measured_rows)
    joint_decode_s = sum(float(row[f"{prefix}_joint_decode_s"]) for row in measured_rows)
    joint_tool_s = sum(float(row[f"{prefix}_joint_tool_s"]) for row in measured_rows)
    joint_s = joint_prefill_s + joint_decode_s + joint_tool_s
    return {
        "eligible_calls": sum(bool(row[f"{prefix}_selected_count"]) for row in rows),
        "response_lcp_tokens": response_lcp,
        "response_draft_coverage": response_lcp / response_tokens if response_tokens else 0.0,
        "reasoning_lcp_tokens": reasoning_lcp,
        "reasoning_draft_coverage": reasoning_lcp / reasoning_tokens if reasoning_tokens else 0.0,
        "visible_response_lcp_tokens": visible_lcp,
        "visible_response_draft_coverage": visible_lcp / visible_tokens if visible_tokens else 0.0,
        "decode_time_weighted_coverage": saved_decode_s / decode_s if decode_s else None,
        "positive_response_lcp_calls": sum(value > 0 for value in lcp_values),
        "accepted_run_length": numeric_distribution(positive_lcp_values),
        "response_lcp_thresholds": {
            str(threshold): sum(value >= threshold for value in lcp_values)
            for threshold in (4, 8, 16, 32, 64)
        },
        "salvage_groups": summarize_salvage_groups(rows, prefix),
        "next_tool_measured_calls": len(next_tool_rows),
        "next_tool_exact_calls": sum(bool(row[f"{prefix}_next_tool_exact"]) for row in next_tool_rows),
        "next_tool_time_s": total_tool_time,
        "next_tool_time_covered_s": covered_tool_time,
        "next_tool_time_coverage": covered_tool_time / total_tool_time if total_tool_time else 0.0,
        "joint_opportunity_s": joint_s,
        "joint_prefill_s": joint_prefill_s,
        "joint_decode_s": joint_decode_s,
        "joint_tool_s": joint_tool_s,
        "estimated_e2e_gain": joint_s / trace_e2e_s if trace_e2e_s else None,
    }


def numeric_distribution(values: list[int]) -> Record:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": values[(len(values) - 1) // 2],
        "p90": values[int((len(values) - 1) * 0.9)],
        "max": values[-1],
    }


def summarize_salvage_groups(rows: list[Record], prefix: str) -> Record:
    groups = {}
    for group in sorted({str(row[f"{prefix}_salvage_group"]) for row in rows}):
        group_rows = [row for row in rows if row[f"{prefix}_salvage_group"] == group]
        values = [int(row[f"{prefix}_response_lcp_tokens"]) for row in group_rows]
        groups[group] = {
            "calls": len(group_rows),
            "response_lcp_tokens": sum(values),
            "lcp_at_least_8_calls": sum(value >= 8 for value in values),
            "lcp_at_least_16_calls": sum(value >= 16 for value in values),
        }
    return groups


def markdown_report(summary: Record) -> str:
    lines = [
        "# BranchAhead Offline Opportunity",
        "",
        f"- Trajectories: {summary['trajectory_count']}",
        f"- Response boundaries: {summary['response_boundary_count']}",
        f"- Response tokens: {summary['response_tokens']}",
        f"- Reasoning tokens: {summary['reasoning_tokens']}",
        f"- Visible response tokens: {summary['visible_response_tokens']}",
        f"- Trace E2E seconds: {summary['trace_e2e_s']:.2f}",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Tokenizer: `{summary['tokenizer_path'] or 'caller-provided object'}`",
        f"- Trace models: `{', '.join(sorted(set(summary['trace_models']))) or 'unknown'}`",
        f"- Recorded completion length used: {summary['used_recorded_completion_length']}",
        "",
        "## Policy frontier",
        "",
        "| Policy | k | Full response | Reasoning | Visible | Decode-weighted | LCP >=16 | Next-tool hit | Tool-time | Joint E2E upper bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        for k in TOP_K:
            metric = summary["policy_frontier"][policy][str(k)]
            lines.append(
                f"| {policy} | {k} | {metric['response_draft_coverage']:.2%} | "
                f"{metric['reasoning_draft_coverage']:.2%} | {metric['visible_response_draft_coverage']:.2%} | "
                f"{format_ratio(metric['decode_time_weighted_coverage'])} | {metric['response_lcp_thresholds']['16']} | "
                f"{metric['next_tool_exact_calls']}/{metric['next_tool_measured_calls']} | "
                f"{metric['next_tool_time_coverage']:.2%} | {format_ratio(metric['estimated_e2e_gain'])} |"
            )
    oracle = summary["policy_frontier"]["any_prior"]["all"]
    lines.append(
        f"| any_prior | all | {oracle['response_draft_coverage']:.2%} | "
        f"{oracle['reasoning_draft_coverage']:.2%} | {oracle['visible_response_draft_coverage']:.2%} | "
        f"{format_ratio(oracle['decode_time_weighted_coverage'])} | {oracle['response_lcp_thresholds']['16']} | "
        f"{oracle['next_tool_exact_calls']}/{oracle['next_tool_measured_calls']} | "
        f"{oracle['next_tool_time_coverage']:.2%} | {format_ratio(oracle['estimated_e2e_gain'])} |"
    )
    command_k4 = summary["policy_frontier"]["command_similarity"]["4"]
    lines.extend(
        [
            "",
            "## Graded salvage at command-similarity k=4",
            "",
            "| Observation/response outcome | Calls | Response LCP tokens | LCP >=8 | LCP >=16 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group, metric in command_k4["salvage_groups"].items():
        lines.append(
            f"| {group} | {metric['calls']} | {metric['response_lcp_tokens']} | "
            f"{metric['lcp_at_least_8_calls']} | {metric['lcp_at_least_16_calls']} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- A response boundary is the final tool observation before the model is called again.",
            "- Candidates contain only earlier tool interactions whose following assistant response was already complete.",
            "- `command_similarity` ranks candidates from the current command and causal history without reading current output.",
            "- Full-response LCP starts at the first reasoning token. Reasoning and visible response are also measured separately.",
            "- Next-tool coverage requires exact tool name and arguments; tool-time coverage is weighted by measured duration.",
            "- The joint upper bound selects one candidate per call. TTFT and decode savings are estimated in proportion to accepted tokens; speculative tool time is capped by the current tool gap.",
            "- `any_prior` searches every causal historical branch and is an offline oracle, not a runtime policy.",
            "",
            "## Caveats",
            "",
            "- Response LCP is trace-sequence overlap, not measured target-model acceptance or wall-clock speedup.",
            "- The joint result is an overhead-free upper bound. TTFT is not strictly linear in prompt tokens, and tool speculation requires isolation and exact commit checks.",
            "- Separate visible-response coverage assumes verification starts after reasoning; it cannot be added to full-response coverage.",
            "- When recorded completion length is disabled, token counts are tokenizer-proxy measurements.",
            "- Speculative verification, branch KV, retrieval, and tool isolation costs are not included.",
            "",
        ]
    )
    return "\n".join(lines)


def format_ratio(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


if __name__ == "__main__":
    app()
