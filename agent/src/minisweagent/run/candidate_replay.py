"""Replay-time execution of Candidate Tool Prefill."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from minisweagent.run.candidate_prefill import (
    CandidatePrefillPlan,
    HistoricalToolCall,
    align_down,
    common_prefix_length,
    select_similar_candidates,
)
from minisweagent.run.replay_types import AsyncPrefillCompletion, AsyncPrefillRequest, ReplayStep


class PrefillSeed(Protocol):
    cached_prefix_len: int
    seed_prefix_len: int


class CandidateTraceTool(Protocol):
    output: dict[str, Any]
    duration_s: float
    output_events: list[dict[str, Any]]
    raw_output: str


class CandidatePrefillWorker(Protocol):
    def submit(self, request: AsyncPrefillRequest) -> bool: ...

    def completed_prefills(self) -> list[AsyncPrefillCompletion]: ...

    def retain_requests(self, request_ids: set[str]) -> int: ...

    def snapshot(self) -> dict[str, Any]: ...

    def cancel_active(self) -> dict[str, Any]: ...

    def raise_if_error(self) -> None: ...

    def stop_and_wait(self) -> None: ...


class CandidateReplayHost(Protocol):
    cache_block_tokens: int
    candidate_top_k: int
    max_context_tokens: int | None
    prefill_check_interval_s: float
    time_scale: float
    now: Callable[[], float]

    def new_prefill_worker(self, *, max_pending: int) -> CandidatePrefillWorker: ...

    def candidate_prompt_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        candidate_output: dict[str, Any],
    ) -> list[int]: ...

    def available_prefill_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        completed_outputs: list[dict[str, Any]],
        action_index: int,
        partial_output: dict[str, Any],
    ) -> list[int]: ...

    def next_prefill_chunk(
        self,
        available_token_ids: list[int],
        *,
        last_prefix_len: int,
    ) -> list[int] | None: ...

    def final_tool_prompt_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        trace_tools: Sequence[CandidateTraceTool],
    ) -> list[int]: ...

    def visible_checkpoints(self, tool: CandidateTraceTool) -> Iterator[tuple[float, int]]: ...

    def clamp_stream_output_chars(self, visible_chars: int) -> int: ...

    def streaming_tool_output(self, output: str) -> dict[str, Any]: ...

    def sleep_until(self, phase_start: float, replay_elapsed_s: float) -> None: ...


@dataclass(frozen=True)
class PreparedCandidatePrefill:
    candidates: tuple[HistoricalToolCall, ...]
    plan: CandidatePrefillPlan
    skipped_capacity_count: int


@dataclass
class CandidatePrefillRequests:
    request_ids_by_candidate: dict[int, str]
    token_ids_by_request: dict[str, list[int]]


class CandidateToolPrefillPhase:
    """Coordinate candidate scheduling, verification, and chunked fallback."""

    def __init__(self, host: CandidateReplayHost):
        self.host = host

    def run(
        self,
        *,
        phase_start: float,
        total_duration: float,
        prefill_seed: PrefillSeed,
        instance_id: str,
        step_index: int,
        cache_salt: str,
        cached_prompt_ids: list[int],
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        trace_tools: Sequence[CandidateTraceTool],
        candidate_history: list[HistoricalToolCall],
    ) -> dict[str, Any]:
        stats = candidate_prefill_stats()
        if not actions:
            self.host.sleep_until(phase_start, total_duration)
            return stats

        prepared = self.prepare(
            cached_prompt_ids=cached_prompt_ids,
            history_after_assistant=history_after_assistant,
            actions=actions,
            candidate_history=candidate_history,
        )
        candidates = prepared.candidates
        plan = prepared.plan
        stats["candidate_skipped_capacity_count"] = prepared.skipped_capacity_count
        stats["candidate_selected_count"] = len(plan.branches)
        stats["candidate_shared_prefix_tokens"] = max(
            0,
            plan.shared_prefix_len - prefill_seed.seed_prefix_len,
        )

        phase_deadline = phase_start + total_duration * self.host.time_scale if self.host.time_scale > 0 else None
        if phase_deadline is not None and self.host.now() >= phase_deadline:
            return stats

        worker = self.host.new_prefill_worker(max_pending=max(1, len(plan.branches)))
        viable_candidates = {branch.candidate_index for branch in plan.branches}
        fallback_to_chunked = not viable_candidates
        if fallback_to_chunked:
            stats["candidate_fallback_to_chunked"] = 1
        last_actual_prefix_len = prefill_seed.cached_prefix_len

        try:
            requests = self.submit(
                worker=worker,
                plan=plan,
                instance_id=instance_id,
                step_index=step_index,
                cache_salt=cache_salt,
            )
            stats["candidate_submitted_count"] = len(requests.request_ids_by_candidate)
            stats["prefill_submitted_count"] = len(requests.request_ids_by_candidate)

            for check_time, raw_visible_chars in self.host.visible_checkpoints(trace_tools[0]):
                self.host.sleep_until(phase_start, check_time)
                if phase_deadline is not None and self.host.now() >= phase_deadline:
                    break

                visible_chars = self.host.clamp_stream_output_chars(raw_visible_chars)
                actual_output = trace_tools[0].raw_output[:visible_chars]
                if visible_chars > 0 and not fallback_to_chunked:
                    viable_candidates = self.prune_candidates(
                        candidates=candidates,
                        viable_candidates=viable_candidates,
                        actual_output=actual_output,
                        requests=requests,
                        worker=worker,
                        stats=stats,
                    )
                    fallback_to_chunked = not viable_candidates

                if not fallback_to_chunked:
                    continue

                stats["candidate_fallback_to_chunked"] = 1
                available_token_ids = self.host.available_prefill_token_ids(
                    history_after_assistant=history_after_assistant,
                    actions=actions,
                    completed_outputs=[],
                    action_index=0,
                    partial_output=self.host.streaming_tool_output(actual_output),
                )
                last_actual_prefix_len = self.completed_prefix_frontier(
                    available_token_ids,
                    worker.completed_prefills(),
                    requests.token_ids_by_request,
                    default=last_actual_prefix_len,
                )
                prefill_token_ids = self.host.next_prefill_chunk(
                    available_token_ids,
                    last_prefix_len=last_actual_prefix_len,
                )
                if prefill_token_ids is None:
                    continue

                request_id = f"{cache_salt}:tool_output:{uuid.uuid4().hex}"
                submitted = worker.submit(
                    AsyncPrefillRequest(
                        token_ids=prefill_token_ids,
                        cache_salt=cache_salt,
                        step=ReplayStep(instance_id=instance_id, step_index=step_index),
                        label="tool_output",
                        request_id=request_id,
                    )
                )
                if submitted:
                    requests.token_ids_by_request[request_id] = prefill_token_ids
                    stats["prefill_submitted_count"] += 1
                    last_actual_prefix_len = len(prefill_token_ids)

            self.host.sleep_until(phase_start, total_duration)
            tool_end = phase_deadline if phase_deadline is not None else self.host.now()
            stats.update(worker.snapshot())
            stats.update(worker.cancel_active())
            worker.raise_if_error()

            completed = [completion for completion in worker.completed_prefills() if completion.finished_at <= tool_end]
            stats["candidate_completed_count"] = sum(
                completion.label == "candidate_tool_output" for completion in completed
            )
            stats["candidate_surviving_count"] = len(viable_candidates)
            stats["prefill_completed_count"] = len(completed)

            final_prompt_ids = self.host.final_tool_prompt_token_ids(
                history_after_assistant=history_after_assistant,
                actions=actions,
                trace_tools=trace_tools,
            )
            candidate_verified_prefix = self.completed_prefix_frontier(
                final_prompt_ids,
                completed,
                requests.token_ids_by_request,
                label="candidate_tool_output",
            )
            completed_prefix_len = self.completed_prefix_frontier(
                final_prompt_ids,
                completed,
                requests.token_ids_by_request,
                default=prefill_seed.cached_prefix_len,
            )
            stats["candidate_verified_prefix_tokens"] = candidate_verified_prefix
            stats["candidate_verified_tool_output_tokens"] = max(
                0,
                candidate_verified_prefix - prefill_seed.seed_prefix_len,
            )
            stats["prefill_completed_prompt_tokens"] = completed_prefix_len
            stats["prefilled_prompt_suffix_tokens"] = max(
                0,
                completed_prefix_len - prefill_seed.cached_prefix_len,
            )
            stats["prefilled_tool_output_tokens"] = max(
                0,
                completed_prefix_len - prefill_seed.seed_prefix_len,
            )
            return stats
        finally:
            worker.stop_and_wait()

    def prepare(
        self,
        *,
        cached_prompt_ids: list[int],
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        candidate_history: list[HistoricalToolCall],
    ) -> PreparedCandidatePrefill:
        retrieved = select_similar_candidates(
            str(actions[0].get("command") or ""),
            candidate_history,
            top_k=self.host.candidate_top_k,
        )
        candidates = []
        candidate_prompts = []
        skipped_capacity_count = 0
        for candidate in retrieved:
            visible_output = candidate.raw_output[: self.host.clamp_stream_output_chars(len(candidate.raw_output))]
            prompt = self.host.candidate_prompt_token_ids(
                history_after_assistant=history_after_assistant,
                actions=actions,
                candidate_output=self.host.streaming_tool_output(visible_output),
            )
            if self.host.max_context_tokens is not None and len(prompt) > self.host.max_context_tokens:
                skipped_capacity_count += 1
                continue
            candidates.append(candidate)
            candidate_prompts.append(prompt)

        plan = CandidatePrefillPlan.build(
            cached_prompt_ids,
            candidate_prompts,
            block_size=self.host.cache_block_tokens,
        )
        return PreparedCandidatePrefill(
            candidates=tuple(candidates),
            plan=plan,
            skipped_capacity_count=skipped_capacity_count,
        )

    def submit(
        self,
        *,
        worker: CandidatePrefillWorker,
        plan: CandidatePrefillPlan,
        instance_id: str,
        step_index: int,
        cache_salt: str,
    ) -> CandidatePrefillRequests:
        requests = CandidatePrefillRequests({}, {})
        for branch in plan.branches:
            request_id = f"{cache_salt}:candidate:{branch.candidate_index}:{uuid.uuid4().hex}"
            token_ids = list(branch.token_ids)
            submitted = worker.submit(
                AsyncPrefillRequest(
                    token_ids=token_ids,
                    cache_salt=cache_salt,
                    step=ReplayStep(instance_id=instance_id, step_index=step_index),
                    label="candidate_tool_output",
                    request_id=request_id,
                )
            )
            if submitted:
                requests.request_ids_by_candidate[branch.candidate_index] = request_id
                requests.token_ids_by_request[request_id] = token_ids
        return requests

    def prune_candidates(
        self,
        *,
        candidates: Sequence[HistoricalToolCall],
        viable_candidates: set[int],
        actual_output: str,
        requests: CandidatePrefillRequests,
        worker: CandidatePrefillWorker,
        stats: dict[str, Any],
    ) -> set[int]:
        surviving = {
            candidate_index
            for candidate_index in viable_candidates
            if candidates[candidate_index].raw_output.startswith(actual_output)
        }
        stats["candidate_pruned_count"] += len(viable_candidates - surviving)
        retained_request_ids = {requests.request_ids_by_candidate[candidate_index] for candidate_index in surviving}
        stats["candidate_cancelled_count"] += worker.retain_requests(retained_request_ids)
        return surviving

    def completed_prefix_frontier(
        self,
        actual_prompt_ids: list[int],
        completions: Sequence[AsyncPrefillCompletion],
        token_ids_by_request: dict[str, list[int]],
        *,
        label: str | None = None,
        default: int = 0,
    ) -> int:
        matching_prefixes = (
            align_down(
                common_prefix_length(
                    actual_prompt_ids,
                    token_ids_by_request[completion.request_id],
                ),
                self.host.cache_block_tokens,
            )
            for completion in completions
            if label is None or completion.label == label
        )
        return max(matching_prefixes, default=default)


def candidate_prefill_stats() -> dict[str, int]:
    return {
        "candidate_selected_count": 0,
        "candidate_skipped_capacity_count": 0,
        "candidate_submitted_count": 0,
        "candidate_completed_count": 0,
        "candidate_shared_prefix_tokens": 0,
        "candidate_verified_prefix_tokens": 0,
        "candidate_verified_tool_output_tokens": 0,
        "candidate_pruned_count": 0,
        "candidate_surviving_count": 0,
        "candidate_fallback_to_chunked": 0,
        "candidate_cancelled_count": 0,
        "prefill_submitted_count": 0,
    }
