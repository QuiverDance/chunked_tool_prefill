"""Prepare metadata-only TraceLab Codex rounds for token-native replay."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

WORKLOAD_FORMAT = "tracelab-codex-pair-v1"
DEFAULT_SAMPLE_SEED = "tracelab-codex-v1"


@dataclass(frozen=True)
class PreparedToolResult:
    tool_call_id: str
    tool_name: str
    completion_offset_s: float
    result_chars: int
    result_tokens: int
    event_index: int


@dataclass(frozen=True)
class PreparedCheckpoint:
    at_s: float
    prompt_tokens: int
    available_result_count: int


@dataclass(frozen=True)
class PreparedTraceLabTrial:
    trial_id: str
    session_id: str
    current_round_index: int
    next_round_index: int
    current_trace_key: str
    next_trace_key: str
    source_model: str
    prompt_tokens: int
    prefix_tokens: int
    newly_append_tokens: int
    static_suffix_tokens: int
    result_suffix_tokens: int
    completion_tokens: int
    tool_phase_duration_s: float
    tool_results: tuple[PreparedToolResult, ...]
    prefill_checkpoints: tuple[PreparedCheckpoint, ...]

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_results)

    @property
    def tool_output_chars(self) -> int:
        return sum(result.result_chars for result in self.tool_results)


@dataclass(frozen=True)
class PreparedTraceLabWorkload:
    source_path: str
    source_sha256: str
    sample_seed: str
    max_context_tokens: int
    cache_block_tokens: int
    requested_limit: int | None
    max_completion_tokens: int | None
    scan_counts: dict[str, int]
    skip_reasons: dict[str, int]
    trials: tuple[PreparedTraceLabTrial, ...]
    static_suffix_policy: str = "max(0,current_input_tokens+current_output_tokens-next_prefix_tokens)"
    result_allocation_policy: str = "next_append_minus_static_proportional_to_result_chars_minimum_one"
    sampling_policy: str = "proportional_fixed_strata_then_seeded_sha256_order"
    workload_format: str = WORKLOAD_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IneligibleTrial(ValueError):
    """A TraceLab round pair that cannot preserve the replay contract."""


def prepare_tracelab_codex_workload(
    source_path: Path,
    *,
    limit: int | None = 100,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
    max_context_tokens: int = 131072,
    cache_block_tokens: int = 16,
    max_completion_tokens: int | None = None,
) -> PreparedTraceLabWorkload:
    """Scan once, validate round pairs, and return a measurement-ready workload."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if cache_block_tokens <= 0:
        raise ValueError("cache_block_tokens must be positive")

    scan_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    eligible: list[PreparedTraceLabTrial] = []
    previous: dict[str, Any] | None = None
    previous_session = ""
    closed_sessions: set[str] = set()

    for row in _iter_jsonl(source_path):
        scan_counts["rows"] += 1
        if row.get("provider") != "codex":
            continue
        scan_counts["codex_rounds"] += 1
        session_id = _required_string(row, "session_id")
        if session_id != previous_session:
            if previous_session:
                closed_sessions.add(previous_session)
            if session_id in closed_sessions:
                raise ValueError(f"Codex session rows are not contiguous: {session_id}")
            previous = None
            previous_session = session_id

        if previous is not None:
            scan_counts["adjacent_round_pairs"] += 1
            try:
                trial = _prepare_trial(
                    previous,
                    row,
                    max_context_tokens=max_context_tokens,
                    cache_block_tokens=cache_block_tokens,
                    max_completion_tokens=max_completion_tokens,
                )
            except IneligibleTrial as error:
                skip_reasons[str(error)] += 1
            else:
                eligible.append(trial)
        previous = row

    scan_counts["eligible_trials"] = len(eligible)
    trials = _stratified_sample(eligible, limit=limit, seed=sample_seed)
    scan_counts["selected_trials"] = len(trials)
    return PreparedTraceLabWorkload(
        source_path=str(source_path.resolve()),
        source_sha256=_sha256(source_path),
        sample_seed=sample_seed,
        max_context_tokens=max_context_tokens,
        cache_block_tokens=cache_block_tokens,
        requested_limit=limit,
        max_completion_tokens=max_completion_tokens,
        scan_counts=dict(sorted(scan_counts.items())),
        skip_reasons=dict(sorted(skip_reasons.items())),
        trials=tuple(trials),
    )


def write_prepared_workload(path: Path, workload: PreparedTraceLabWorkload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n")


def load_prepared_workload(path: Path) -> PreparedTraceLabWorkload:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("workload_format") != WORKLOAD_FORMAT:
        raise ValueError(f"Unsupported TraceLab workload manifest: {path}")
    trials = tuple(_trial_from_dict(item) for item in data.get("trials") or [])
    if not trials:
        raise ValueError(f"TraceLab workload has no trials: {path}")
    workload = PreparedTraceLabWorkload(
        source_path=str(data["source_path"]),
        source_sha256=str(data["source_sha256"]),
        sample_seed=str(data["sample_seed"]),
        max_context_tokens=int(data["max_context_tokens"]),
        cache_block_tokens=int(data["cache_block_tokens"]),
        requested_limit=_optional_int(data.get("requested_limit")),
        max_completion_tokens=_optional_int(data.get("max_completion_tokens")),
        scan_counts={str(key): int(value) for key, value in (data.get("scan_counts") or {}).items()},
        skip_reasons={str(key): int(value) for key, value in (data.get("skip_reasons") or {}).items()},
        trials=trials,
        static_suffix_policy=str(data["static_suffix_policy"]),
        result_allocation_policy=str(data["result_allocation_policy"]),
        sampling_policy=str(data["sampling_policy"]),
    )
    for trial in workload.trials:
        _validate_prepared_trial(trial, workload)
    return workload


def _prepare_trial(
    current: dict[str, Any],
    following: dict[str, Any],
    *,
    max_context_tokens: int,
    cache_block_tokens: int,
    max_completion_tokens: int | None,
) -> PreparedTraceLabTrial:
    session_id = _required_string(current, "session_id")
    if _required_string(following, "session_id") != session_id:
        raise IneligibleTrial("different_session")
    current_round = _required_int(current, "round_index")
    next_round = _required_int(following, "round_index")
    if next_round != current_round + 1:
        raise IneligibleTrial("nonconsecutive_round_index")

    tools = current.get("tools")
    if not isinstance(tools, list) or len(tools) < 2:
        raise IneligibleTrial("not_multi_tool")
    if any(not isinstance(tool, dict) for tool in tools):
        raise IneligibleTrial("invalid_tool_record")
    tool_ids = [_required_string(tool, "tool_call_id") for tool in tools]
    if len(set(tool_ids)) != len(tool_ids):
        raise IneligibleTrial("duplicate_tool_call_id")
    if any(tool.get("result_at") in ("", None) for tool in tools):
        raise IneligibleTrial("missing_result_timestamp")

    next_results = [
        (event_index, event)
        for event_index, event in enumerate(following.get("timing_events") or [], 1)
        if isinstance(event, dict) and event.get("event_type") == "tool_result"
    ]
    next_result_ids = [_required_string(event, "tool_call_id") for _, event in next_results]
    if len(next_result_ids) != len(tool_ids) or set(next_result_ids) != set(tool_ids):
        raise IneligibleTrial("next_round_result_set_mismatch")
    if _required_int(following, "current_user_message_count") != 0:
        raise IneligibleTrial("concurrent_user_input")
    if _required_int(following, "current_tool_result_count") != len(tools):
        raise IneligibleTrial("next_round_result_count_mismatch")

    result_chars = [_nonnegative_int(tool.get("result_chars"), "result_chars") for tool in tools]
    if _required_int(following, "current_tool_result_chars") != sum(result_chars):
        raise IneligibleTrial("next_round_result_chars_mismatch")

    current_input = _nonnegative_int(current.get("input_tokens_total"), "current_input_tokens")
    current_output = _nonnegative_int(current.get("output_tokens"), "current_output_tokens")
    next_input = _nonnegative_int(following.get("input_tokens_total"), "next_input_tokens")
    next_prefix = _nonnegative_int(following.get("prefix_tokens"), "next_prefix_tokens")
    next_append = _nonnegative_int(following.get("newly_append_tokens"), "next_append_tokens")
    next_output = _nonnegative_int(following.get("output_tokens"), "next_output_tokens")
    if next_input != next_prefix + next_append:
        raise IneligibleTrial("next_token_accounting_mismatch")
    if next_prefix % cache_block_tokens:
        raise IneligibleTrial("prefix_not_cache_block_aligned")
    if next_output <= 0:
        raise IneligibleTrial("zero_completion_tokens")
    if max_completion_tokens is not None and next_output > max_completion_tokens:
        raise IneligibleTrial("completion_token_filter")
    if next_input + next_output > max_context_tokens:
        raise IneligibleTrial("skipped_capacity")

    static_suffix_tokens = max(0, current_input + current_output - next_prefix)
    if static_suffix_tokens > next_append:
        raise IneligibleTrial("negative_result_token_budget")
    result_suffix_tokens = next_append - static_suffix_tokens
    if result_suffix_tokens < len(tools):
        raise IneligibleTrial("insufficient_result_token_budget")

    emitted_at = [_timestamp(tool.get("emitted_at"), "emitted_at") for tool in tools]
    completed_at = [_timestamp(tool.get("result_at"), "result_at") for tool in tools]
    phase_start = min(emitted_at)
    completion_offsets = {
        tool_id: max(0.0, (finished - phase_start).total_seconds())
        for tool_id, finished in zip(tool_ids, completed_at, strict=True)
    }
    tool_phase_duration_s = max(completion_offsets.values())
    if not math.isfinite(tool_phase_duration_s):
        raise IneligibleTrial("invalid_tool_duration")

    tokens_by_id = _allocate_tokens(
        result_suffix_tokens,
        {tool_id: chars for tool_id, chars in zip(tool_ids, result_chars, strict=True)},
    )
    tools_by_id = {tool_id: tool for tool_id, tool in zip(tool_ids, tools, strict=True)}
    tool_results = tuple(
        PreparedToolResult(
            tool_call_id=tool_id,
            tool_name=str(tools_by_id[tool_id].get("tool_name") or "tool"),
            completion_offset_s=completion_offsets[tool_id],
            result_chars=_nonnegative_int(tools_by_id[tool_id].get("result_chars"), "result_chars"),
            result_tokens=tokens_by_id[tool_id],
            event_index=event_index,
        )
        for (event_index, _), tool_id in zip(next_results, next_result_ids, strict=True)
    )
    checkpoints = _prefill_checkpoints(
        prefix_tokens=next_prefix,
        static_suffix_tokens=static_suffix_tokens,
        tool_results=tool_results,
        phase_duration_s=tool_phase_duration_s,
    )
    trial_id = f"{session_id}:{current_round}->{next_round}"
    return PreparedTraceLabTrial(
        trial_id=trial_id,
        session_id=session_id,
        current_round_index=current_round,
        next_round_index=next_round,
        current_trace_key=str(current.get("trace_key") or ""),
        next_trace_key=str(following.get("trace_key") or ""),
        source_model=str(following.get("model") or current.get("model") or ""),
        prompt_tokens=next_input,
        prefix_tokens=next_prefix,
        newly_append_tokens=next_append,
        static_suffix_tokens=static_suffix_tokens,
        result_suffix_tokens=result_suffix_tokens,
        completion_tokens=next_output,
        tool_phase_duration_s=tool_phase_duration_s,
        tool_results=tool_results,
        prefill_checkpoints=checkpoints,
    )


def _prefill_checkpoints(
    *,
    prefix_tokens: int,
    static_suffix_tokens: int,
    tool_results: tuple[PreparedToolResult, ...],
    phase_duration_s: float,
) -> tuple[PreparedCheckpoint, ...]:
    if phase_duration_s <= 0:
        return ()

    checkpoints: list[PreparedCheckpoint] = []
    prompt_tokens = prefix_tokens
    if static_suffix_tokens:
        prompt_tokens += static_suffix_tokens
        checkpoints.append(
            PreparedCheckpoint(
                at_s=0.0,
                prompt_tokens=prompt_tokens,
                available_result_count=0,
            )
        )

    available: set[str] = set()
    serialized_prefix_count = 0
    completion_times = sorted({result.completion_offset_s for result in tool_results})
    for completion_time in completion_times:
        available.update(
            result.tool_call_id for result in tool_results if result.completion_offset_s == completion_time
        )
        previous_count = serialized_prefix_count
        while (
            serialized_prefix_count < len(tool_results)
            and tool_results[serialized_prefix_count].tool_call_id in available
        ):
            prompt_tokens += tool_results[serialized_prefix_count].result_tokens
            serialized_prefix_count += 1
        if serialized_prefix_count == previous_count or completion_time >= phase_duration_s:
            continue
        checkpoints.append(
            PreparedCheckpoint(
                at_s=completion_time,
                prompt_tokens=prompt_tokens,
                available_result_count=serialized_prefix_count,
            )
        )
    return tuple(checkpoints)


def _allocate_tokens(total: int, weights_by_id: dict[str, int]) -> dict[str, int]:
    call_ids = list(weights_by_id)
    remaining = total - len(call_ids)
    allocations = {call_id: 1 for call_id in call_ids}
    if remaining <= 0:
        return allocations

    weights = {call_id: max(1, weights_by_id[call_id]) for call_id in call_ids}
    weight_total = sum(weights.values())
    exact = {call_id: remaining * weights[call_id] / weight_total for call_id in call_ids}
    floors = {call_id: math.floor(exact[call_id]) for call_id in call_ids}
    for call_id, value in floors.items():
        allocations[call_id] += value
    leftovers = remaining - sum(floors.values())
    ranked = sorted(
        call_ids,
        key=lambda call_id: (-(exact[call_id] - floors[call_id]), call_id),
    )
    for call_id in ranked[:leftovers]:
        allocations[call_id] += 1
    return allocations


def _stratified_sample(
    trials: list[PreparedTraceLabTrial],
    *,
    limit: int | None,
    seed: str,
) -> list[PreparedTraceLabTrial]:
    if not trials:
        return []
    if limit is None or limit >= len(trials):
        return sorted(trials, key=lambda trial: _stable_rank(seed, trial.trial_id))

    groups: dict[tuple[int, int, int], list[PreparedTraceLabTrial]] = defaultdict(list)
    for trial in trials:
        groups[_stratum(trial)].append(trial)
    for members in groups.values():
        members.sort(key=lambda trial: _stable_rank(seed, trial.trial_id))

    total = len(trials)
    allocations: dict[tuple[int, int, int], int] = {}
    remainders = []
    assigned = 0
    for key, members in groups.items():
        exact = limit * len(members) / total
        count = min(len(members), math.floor(exact))
        allocations[key] = count
        assigned += count
        remainders.append((exact - count, _stable_rank(seed, repr(key)), key))
    for _, _, key in sorted(remainders, reverse=True):
        if assigned >= limit:
            break
        if allocations[key] >= len(groups[key]):
            continue
        allocations[key] += 1
        assigned += 1

    selected = [trial for key, members in groups.items() for trial in members[: allocations[key]]]
    return sorted(selected, key=lambda trial: _stable_rank(seed, trial.trial_id))


def _stratum(trial: PreparedTraceLabTrial) -> tuple[int, int, int]:
    tool_bucket = min(trial.tool_call_count, 4)
    prompt_bucket = _fixed_bucket(trial.prompt_tokens, (65536, 98304, 114688))
    duration_bucket = _fixed_bucket(trial.tool_phase_duration_s, (0.25, 1.0, 5.0))
    return tool_bucket, prompt_bucket, duration_bucket


def _fixed_bucket(value: float, boundaries: tuple[float, ...]) -> int:
    return next((index for index, boundary in enumerate(boundaries) if value <= boundary), len(boundaries))


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _trial_from_dict(data: dict[str, Any]) -> PreparedTraceLabTrial:
    tool_results = tuple(
        PreparedToolResult(
            tool_call_id=str(item["tool_call_id"]),
            tool_name=str(item["tool_name"]),
            completion_offset_s=float(item["completion_offset_s"]),
            result_chars=int(item["result_chars"]),
            result_tokens=int(item["result_tokens"]),
            event_index=int(item["event_index"]),
        )
        for item in data.get("tool_results") or []
    )
    checkpoints = tuple(
        PreparedCheckpoint(
            at_s=float(item["at_s"]),
            prompt_tokens=int(item["prompt_tokens"]),
            available_result_count=int(item["available_result_count"]),
        )
        for item in data.get("prefill_checkpoints") or []
    )
    return PreparedTraceLabTrial(
        trial_id=str(data["trial_id"]),
        session_id=str(data["session_id"]),
        current_round_index=int(data["current_round_index"]),
        next_round_index=int(data["next_round_index"]),
        current_trace_key=str(data["current_trace_key"]),
        next_trace_key=str(data["next_trace_key"]),
        source_model=str(data["source_model"]),
        prompt_tokens=int(data["prompt_tokens"]),
        prefix_tokens=int(data["prefix_tokens"]),
        newly_append_tokens=int(data["newly_append_tokens"]),
        static_suffix_tokens=int(data["static_suffix_tokens"]),
        result_suffix_tokens=int(data["result_suffix_tokens"]),
        completion_tokens=int(data["completion_tokens"]),
        tool_phase_duration_s=float(data["tool_phase_duration_s"]),
        tool_results=tool_results,
        prefill_checkpoints=checkpoints,
    )


def _validate_prepared_trial(
    trial: PreparedTraceLabTrial,
    workload: PreparedTraceLabWorkload,
) -> None:
    if trial.prompt_tokens != trial.prefix_tokens + trial.newly_append_tokens:
        raise ValueError(f"Invalid prompt accounting for {trial.trial_id}")
    if trial.newly_append_tokens != trial.static_suffix_tokens + trial.result_suffix_tokens:
        raise ValueError(f"Invalid suffix accounting for {trial.trial_id}")
    if trial.result_suffix_tokens != sum(result.result_tokens for result in trial.tool_results):
        raise ValueError(f"Invalid result allocation for {trial.trial_id}")
    if trial.prefix_tokens % workload.cache_block_tokens:
        raise ValueError(f"Unaligned prefix for {trial.trial_id}")
    if trial.prompt_tokens + trial.completion_tokens > workload.max_context_tokens:
        raise ValueError(f"Over-capacity trial in prepared workload: {trial.trial_id}")
    previous_prompt_tokens = trial.prefix_tokens
    previous_at = 0.0
    for checkpoint in trial.prefill_checkpoints:
        if checkpoint.prompt_tokens <= previous_prompt_tokens:
            raise ValueError(f"Non-growing checkpoint for {trial.trial_id}")
        if checkpoint.at_s < previous_at or checkpoint.at_s >= trial.tool_phase_duration_s:
            raise ValueError(f"Invalid checkpoint time for {trial.trial_id}")
        previous_prompt_tokens = checkpoint.prompt_tokens
        previous_at = checkpoint.at_s


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise IneligibleTrial(f"missing_{key}")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    return _nonnegative_int(data.get(key), key)


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise IneligibleTrial(f"missing_{name}") from error
    if parsed < 0:
        raise IneligibleTrial(f"negative_{name}")
    return parsed


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise IneligibleTrial(f"missing_{name}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IneligibleTrial(f"invalid_{name}") from error
