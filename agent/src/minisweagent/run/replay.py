#!/usr/bin/env python3

"""Replay saved trajectories as trace-driven serving workloads."""

from __future__ import annotations

import copy
import itertools
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from minisweagent.config import get_config_from_spec
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, format_toolcall_observation_messages
from minisweagent.run.benchmarks.utils.token_timing import load_tokenizer
from minisweagent.run.candidate_prefill import HistoricalToolCall
from minisweagent.run.candidate_replay import CandidateToolPrefillPhase
from minisweagent.run.replay_backend import HttpReplayBackend
from minisweagent.run.replay_messages import api_messages, tokenizer_safe_messages
from minisweagent.run.replay_metrics import record_for_output, summarize
from minisweagent.run.replay_types import (
    AsyncPrefillCompletion,
    AsyncPrefillRequest,
    PromptTokenState,
    ReplayError,
    ReplayStep,
)
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

ReplayAlgorithm = Literal["baseline", "chunked", "candidate"]

DEFAULT_PREFILL_CHUNK_TOKENS = 128
DEFAULT_PREFILL_MIN_NEW_TOKENS = DEFAULT_PREFILL_CHUNK_TOKENS
DEFAULT_PREFILL_CHECK_INTERVAL_S = 0.05
DEFAULT_CACHE_BLOCK_TOKENS = 16
DEFAULT_PREFILL_SAFETY_TAIL_TOKENS = 0
DEFAULT_CANDIDATE_TOP_K = 4
DEFAULT_TIME_SCALE = 1.0


@dataclass(frozen=True)
class TraceToolCall:
    output: dict[str, Any]
    duration_s: float
    output_events: list[dict[str, Any]]
    missing_timing: bool

    @property
    def raw_output(self) -> str:
        return str(self.output.get("output") or "")


@dataclass(frozen=True)
class ToolPrefillSeed:
    cached_prefix_len: int
    seed_prefix_len: int


@dataclass(frozen=True)
class ToolPhaseResult:
    stats: dict[str, Any]
    prefill_seed: ToolPrefillSeed | None


@dataclass(frozen=True)
class ReplayTurn:
    step_index: int
    leading_messages: list[dict[str, Any]]
    assistant: dict[str, Any]
    actions: list[dict[str, Any]]
    trace_tools: list[TraceToolCall]
    model_call: dict[str, Any]
    trace_completion_tokens: int
    has_next_assistant: bool
    next_turn_follows_tools: bool


@dataclass(frozen=True)
class ReplayScenario:
    path: Path
    instance_id: str
    turns: list[ReplayTurn]
    terminal_invalid: dict[str, Any] | None = None


@dataclass
class StepMeasurement:
    turn: ReplayTurn
    prompt_tokens: int | None = None
    valid: bool = False
    skip_reason: str = ""
    replay_ttft_s: float | None = None
    replay_model_total_s: float | None = None
    replay_decode_s: float | None = None
    replay_completion_tokens: int | None = None
    cached_tokens: int | None = None
    tool_stats: dict[str, Any] = field(default_factory=dict)
    problem_e2e_s: float | None = None


@dataclass(frozen=True)
class PreparedRunResult:
    measurements: list[StepMeasurement]
    problem_e2e_s: float
    completed_all_turns: bool


class AsyncPrefillWorker:
    """Sequential background prefill worker for simulated tool time."""

    def __init__(
        self,
        backend: HttpReplayBackend,
        *,
        now: Callable[[], float] = time.perf_counter,
        max_pending: int = 1,
    ):
        self.backend = backend
        self.now = now
        self.max_pending = max(1, max_pending)
        self.condition = threading.Condition()
        self.pending: deque[AsyncPrefillRequest] = deque()
        self.active_request: AsyncPrefillRequest | None = None
        self.active = False
        self.closed = False
        self.error: ReplayError | None = None
        self.completions: list[AsyncPrefillCompletion] = []
        self.cancelled_request_ids: set[str] = set()
        self.started_count = 0
        self.thread = threading.Thread(target=self.run, name="trace-replay-prefill-worker", daemon=True)
        self.thread.start()

    def submit(self, request: AsyncPrefillRequest) -> bool:
        with self.condition:
            if self.error is not None:
                raise self.error
            if self.closed or len(self.pending) >= self.max_pending:
                return False
            self.pending.append(request)
            self.condition.notify()
            return True

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            active_request = self.active_request
            pending_request = self.pending[0] if self.pending else None
            return {
                "prefill_count": self.started_count,
                "prefill_started_count": self.started_count,
                "prefill_active_at_tool_end": int(self.active),
                "prefill_pending_at_tool_end": int(bool(self.pending)),
                "active_prefill_prefix_len_at_tool_end": len(active_request.token_ids)
                if active_request is not None
                else None,
                "pending_prefill_prefix_len_at_tool_end": len(pending_request.token_ids)
                if pending_request is not None
                else None,
            }

    def completed_prefills(self) -> list[AsyncPrefillCompletion]:
        with self.condition:
            return list(self.completions)

    def raise_if_error(self) -> None:
        with self.condition:
            error = self.error
        if error is not None:
            raise error

    def cancel_active(self) -> dict[str, Any]:
        with self.condition:
            request = self.active_request
            self.pending.clear()
            if request is None:
                return {
                    "active_prefill_cancel_requested_at_tool_end": 0,
                    "active_prefill_cancel_latency_s": None,
                    "active_prefill_cancel_error": "",
                }
            self.cancelled_request_ids.add(request.request_id)

        started_at = self.now()
        error = ""
        try:
            self.backend.cancel_prefill(request.request_id)
        except ReplayError as e:
            error = str(e)
        except Exception as e:
            error = f"prefill_cancel_failed:{type(e).__name__}"
        if error:
            with self.condition:
                self.cancelled_request_ids.discard(request.request_id)

        return {
            "active_prefill_cancel_requested_at_tool_end": 1,
            "active_prefill_cancel_latency_s": self.now() - started_at,
            "active_prefill_cancel_error": error,
        }

    def retain_requests(self, request_ids: set[str]) -> int:
        with self.condition:
            pending_before = len(self.pending)
            self.pending = deque(request for request in self.pending if request.request_id in request_ids)
            removed = pending_before - len(self.pending)
            active = self.active_request
            if active is None or active.request_id in request_ids:
                return removed
            self.cancelled_request_ids.add(active.request_id)

        try:
            self.backend.cancel_prefill(active.request_id)
        except Exception:
            with self.condition:
                self.cancelled_request_ids.discard(active.request_id)
            return removed
        return removed + 1

    def stop_without_drain(self) -> None:
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout=0.1)

    def run(self) -> None:
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if not self.pending and self.closed:
                    return
                request = self.pending.popleft()
                self.active = True
                self.active_request = request
                self.started_count += 1

            assert request is not None
            error: ReplayError | None = None
            try:
                self.backend.prefill(
                    request.token_ids,
                    cache_salt=request.cache_salt,
                    step=request.step,
                    label=request.label,
                    request_id=request.request_id,
                )
            except ReplayError as e:
                error = e
            except Exception as e:
                error = ReplayError(f"prefill_failed:{type(e).__name__}")

            with self.condition:
                cancelled = request.request_id in self.cancelled_request_ids
                if error is not None and not cancelled:
                    self.error = error
                    self.pending.clear()
                if error is None and not cancelled:
                    self.completions.append(
                        AsyncPrefillCompletion(
                            request_id=request.request_id,
                            label=request.label,
                            prefix_len=len(request.token_ids),
                            finished_at=self.now(),
                        )
                    )
                self.cancelled_request_ids.discard(request.request_id)
                self.active = False
                self.active_request = None
                self.condition.notify_all()

            if error is not None and not cancelled:
                return


class ReplayTokenizer:
    def __init__(self, tokenizer: Any):
        if tokenizer is None:
            raise ReplayError("missing_tokenizer")
        self.tokenizer = tokenizer

    @classmethod
    def from_path(cls, tokenizer_path: str, *, local_files_only: bool = True) -> ReplayTokenizer:
        if not tokenizer_path:
            raise ReplayError("missing_tokenizer_path")
        if (Path(tokenizer_path) / "tekken.json").exists():
            return MistralReplayTokenizer.from_path(tokenizer_path)
        return cls(load_tokenizer(tokenizer_path, local_files_only=local_files_only))

    def encode_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        return self.encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt).token_ids

    def encode_messages_with_state(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> PromptTokenState:
        text = self.render_messages(messages, add_generation_prompt=add_generation_prompt)
        return PromptTokenState(text=text, token_ids=self.encode_prompt_text(text))

    def render_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        clean_messages = api_messages(messages)
        errors = []
        for candidate in (clean_messages, tokenizer_safe_messages(clean_messages)):
            try:
                return str(
                    self.tokenizer.apply_chat_template(
                        candidate,
                        tools=[BASH_TOOL],
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                )
            except TypeError as e:
                errors.append(type(e).__name__)
                try:
                    return str(
                        self.tokenizer.apply_chat_template(
                            candidate,
                            tokenize=False,
                            add_generation_prompt=add_generation_prompt,
                        )
                    )
                except Exception as inner:
                    errors.append(type(inner).__name__)
            except Exception as e:
                errors.append(type(e).__name__)
        raise ReplayError(f"chat_template_failed:{','.join(errors)}")

    def encode_prompt_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text or "", add_special_tokens=False))


class MistralReplayTokenizer(ReplayTokenizer):
    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer)
        self.text_tokenizer = tokenizer.instruct_tokenizer.tokenizer

    @classmethod
    def from_path(cls, tokenizer_path: str) -> MistralReplayTokenizer:
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as e:
            raise ReplayError("missing_mistral_common") from e
        return cls(MistralTokenizer.from_file(Path(tokenizer_path) / "tekken.json"))

    def encode_messages_with_state(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> PromptTokenState:
        try:
            from mistral_common.protocol.instruct.request import (
                ChatCompletionRequest,
                convert_openai_messages,
                convert_openai_tools,
            )
        except ImportError as e:
            raise ReplayError("missing_mistral_common") from e

        clean_messages = mistral_safe_messages(api_messages(messages))
        if not clean_messages:
            return PromptTokenState(text="", token_ids=[])

        request = ChatCompletionRequest(
            model="mistral-small",
            messages=convert_openai_messages(clean_messages),
            tools=convert_openai_tools([BASH_TOOL]),
            continue_final_message=not add_generation_prompt and clean_messages[-1].get("role") == "assistant",
        )
        encoded = self.tokenizer.encode_chat_completion(request)
        return PromptTokenState(text="", token_ids=list(encoded.tokens))

    def render_messages(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        state = self.encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)
        return self.tokenizer.decode(state.token_ids)

    def encode_prompt_text(self, text: str) -> list[int]:
        return list(self.text_tokenizer.encode(text or "", bos=False, eos=False))


class TraceReplayRunner:
    def __init__(
        self,
        backend: HttpReplayBackend,
        tokenizer: ReplayTokenizer,
        config: dict[str, Any],
        *,
        algorithm: ReplayAlgorithm,
        max_context_tokens: int | None = None,
        prefill_min_new_tokens: int = DEFAULT_PREFILL_MIN_NEW_TOKENS,
        prefill_chunk_tokens: int | None = None,
        prefill_check_interval_s: float = DEFAULT_PREFILL_CHECK_INTERVAL_S,
        prefill_safety_tail_tokens: int = DEFAULT_PREFILL_SAFETY_TAIL_TOKENS,
        stream_output_char_limit: int | None = None,
        cache_block_tokens: int = DEFAULT_CACHE_BLOCK_TOKENS,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        time_scale: float = DEFAULT_TIME_SCALE,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.perf_counter,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = config.get("model") or {}
        self.algorithm = algorithm
        self.max_context_tokens = max_context_tokens
        self.prefill_chunk_tokens = prefill_chunk_tokens or prefill_min_new_tokens
        self.prefill_check_interval_s = prefill_check_interval_s
        self.prefill_safety_tail_tokens = prefill_safety_tail_tokens
        self.stream_output_char_limit = stream_output_char_limit
        self.cache_block_tokens = cache_block_tokens
        self.candidate_top_k = candidate_top_k
        self.time_scale = time_scale
        self.sleep = sleep
        self.now = now

    @property
    def observation_template(self) -> str:
        return str(
            self.model_config.get("observation_template")
            or "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
            "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
        )

    @property
    def stream_observation_template(self) -> str:
        return str(self.model_config.get("stream_observation_template") or "<output>\n{{ output.output }}")

    def run_trajectory(self, path: Path, data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scenario = prepare_replay_scenario(path, data)
        run_result = self.run_prepared_scenario(scenario)
        records = records_from_measurements(scenario, self.algorithm, run_result.measurements)
        invalid = [scenario.terminal_invalid] if run_result.completed_all_turns and scenario.terminal_invalid else []
        return records, invalid

    def run_prepared_scenario(self, scenario: ReplayScenario) -> PreparedRunResult:
        cache_salt = f"{scenario.instance_id}:{self.algorithm}:{uuid.uuid4().hex}"
        history: list[dict[str, Any]] = []
        measurements: list[StepMeasurement] = []
        next_prompt_state: PromptTokenState | None = None
        candidate_history: list[HistoricalToolCall] = []
        started_at = self.now()
        completed_all_turns = True

        for turn_index, turn in enumerate(scenario.turns):
            if turn.leading_messages:
                next_prompt_state = None
                history.extend(turn.leading_messages)

            if next_prompt_state is None:
                prompt_state = self.tokenizer.encode_messages_with_state(history, add_generation_prompt=True)
            else:
                prompt_state = next_prompt_state
                next_prompt_state = None
            prompt_ids = prompt_state.token_ids
            step = ReplayStep(instance_id=scenario.instance_id, step_index=turn.step_index)
            if self.max_context_tokens is not None and len(prompt_ids) > self.max_context_tokens:
                measurements.extend(skipped_measurements_from_turns(scenario.turns, turn_index, "skipped_capacity"))
                completed_all_turns = False
                break

            measurement = self.measure_model_call(
                turn=turn,
                step=step,
                prompt_ids=prompt_ids,
                cache_salt=cache_salt,
            )
            if not measurement.valid:
                measurements.append(measurement)
                measurements.extend(
                    skipped_measurements_from_turns(
                        scenario.turns,
                        turn_index + 1,
                        measurement.skip_reason or "replay_failed",
                    )
                )
                completed_all_turns = False
                break

            history_after_assistant = history + [turn.assistant]
            try:
                tool_phase = self.simulate_tool_phase(
                    instance_id=scenario.instance_id,
                    step_index=turn.step_index,
                    cache_salt=cache_salt,
                    cached_prompt_ids=prompt_ids,
                    history_after_assistant=history_after_assistant,
                    actions=turn.actions,
                    trace_tools=turn.trace_tools,
                    candidate_history=candidate_history,
                    prefill_enabled=self.algorithm in {"chunked", "candidate"} and turn.has_next_assistant,
                )
            except ReplayError as e:
                measurement.valid = False
                measurement.skip_reason = str(e)
                measurements.append(measurement)
                measurements.extend(skipped_measurements_from_turns(scenario.turns, turn_index + 1, str(e)))
                completed_all_turns = False
                break

            tool_stats = tool_phase.stats
            prefill_seed = tool_phase.prefill_seed
            final_tool_messages = self.observation_messages(turn.actions, [tool.output for tool in turn.trace_tools])
            final_messages = history_after_assistant + final_tool_messages
            final_prompt_state = None
            if prefill_seed is not None and turn.has_next_assistant:
                final_prompt_state = self.tokenizer.encode_messages_with_state(
                    final_messages, add_generation_prompt=True
                )
                unprefilled_suffix_tokens = max(
                    0,
                    len(final_prompt_state.token_ids)
                    - max(
                        int(tool_stats.get("prefill_completed_prompt_tokens") or 0),
                        prefill_seed.cached_prefix_len,
                    ),
                )
                tool_stats["unprefilled_prompt_suffix_tokens"] = unprefilled_suffix_tokens
                tool_stats["unprefilled_tool_output_tokens"] = max(
                    0,
                    len(final_prompt_state.token_ids)
                    - max(
                        int(tool_stats.get("prefill_completed_prompt_tokens") or 0),
                        prefill_seed.seed_prefix_len,
                    ),
                )
            measurement.tool_stats = tool_stats
            measurements.append(measurement)

            for action, tool in zip(turn.actions, turn.trace_tools):
                candidate_history.append(
                    HistoricalToolCall(
                        command=str(action.get("command") or ""),
                        output=tool.output,
                        call_index=len(candidate_history),
                    )
                )

            history = final_messages
            if final_prompt_state is not None and turn.next_turn_follows_tools:
                next_prompt_state = final_prompt_state
            else:
                next_prompt_state = None

        problem_e2e_s = self.now() - started_at
        if measurements:
            measurements[0].problem_e2e_s = problem_e2e_s
        return PreparedRunResult(
            measurements=measurements,
            problem_e2e_s=problem_e2e_s,
            completed_all_turns=completed_all_turns,
        )

    def measure_model_call(
        self,
        *,
        turn: ReplayTurn,
        step: ReplayStep,
        prompt_ids: list[int],
        cache_salt: str,
    ) -> StepMeasurement:
        measurement = StepMeasurement(turn=turn, prompt_tokens=len(prompt_ids))
        try:
            result = self.backend.generate_tokens(
                prompt_ids,
                max_tokens=turn.trace_completion_tokens,
                cache_salt=cache_salt,
                step=step,
                label="trace_decode",
            )
        except ReplayError as e:
            reason = "skipped_capacity" if is_capacity_error(str(e)) else str(e)
            measurement.skip_reason = reason
            return measurement

        measurement.valid = True
        measurement.replay_ttft_s = result.get("ttft_s")
        measurement.replay_model_total_s = result.get("model_total_s")
        measurement.replay_decode_s = result.get("decode_s")
        measurement.replay_completion_tokens = result.get("completion_tokens")
        measurement.cached_tokens = result.get("cached_tokens")
        return measurement

    def prefill_seed(
        self,
        history_after_assistant: list[dict[str, Any]],
        *,
        cached_prompt_ids: list[int],
    ) -> ToolPrefillSeed:
        seed_state = self.tokenizer.encode_messages_with_state(history_after_assistant, add_generation_prompt=False)
        return ToolPrefillSeed(
            cached_prefix_len=align_down(
                common_prefix_length(cached_prompt_ids, seed_state.token_ids),
                self.cache_block_tokens,
            ),
            seed_prefix_len=len(seed_state.token_ids),
        )

    def simulate_tool_phase(
        self,
        *,
        instance_id: str,
        step_index: int,
        cache_salt: str,
        cached_prompt_ids: list[int],
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        trace_tools: list[TraceToolCall],
        candidate_history: list[HistoricalToolCall],
        prefill_enabled: bool,
    ) -> ToolPhaseResult:
        total_duration = sum(max(0.0, tool.duration_s) for tool in trace_tools)
        stats = {
            "tool_call_count": len(trace_tools),
            "simulated_tool_duration_s": total_duration,
            "tool_output_chars": sum(len(tool.raw_output) for tool in trace_tools),
            "tool_output_events": sum(len(tool.output_events) for tool in trace_tools),
            "missing_tool_timing_count": sum(1 for tool in trace_tools if tool.missing_timing),
            "prefill_count": 0,
            "prefill_submitted_count": 0,
            "prefill_started_count": 0,
            "prefill_completed_count": 0,
            "prefill_completed_prompt_tokens": 0,
            "prefilled_prompt_suffix_tokens": 0,
            "prefilled_tool_output_tokens": 0,
            "unprefilled_prompt_suffix_tokens": None,
            "unprefilled_tool_output_tokens": None,
            "prefill_active_at_tool_end": 0,
            "prefill_pending_at_tool_end": 0,
            "active_prefill_prefix_len_at_tool_end": None,
            "pending_prefill_prefix_len_at_tool_end": None,
            "active_prefill_cancel_requested_at_tool_end": 0,
            "active_prefill_cancel_latency_s": None,
            "active_prefill_cancel_error": "",
        }
        if not prefill_enabled or not trace_tools:
            self.sleep_scaled(total_duration)
            return ToolPhaseResult(stats=stats, prefill_seed=None)

        if self.algorithm == "candidate":
            return self.simulate_candidate_tool_phase(
                instance_id=instance_id,
                step_index=step_index,
                cache_salt=cache_salt,
                cached_prompt_ids=cached_prompt_ids,
                history_after_assistant=history_after_assistant,
                actions=actions,
                trace_tools=trace_tools,
                candidate_history=candidate_history,
                stats=stats,
                total_duration=total_duration,
            )

        phase_start = self.now()
        phase_deadline = phase_start + total_duration * self.time_scale if self.time_scale > 0 else None
        prefill_seed = self.prefill_seed(
            history_after_assistant,
            cached_prompt_ids=cached_prompt_ids,
        )
        if phase_deadline is not None and self.now() >= phase_deadline:
            return ToolPhaseResult(stats=stats, prefill_seed=prefill_seed)

        worker = AsyncPrefillWorker(self.backend, now=self.now)
        last_prefix_len = prefill_seed.cached_prefix_len
        completed_outputs: list[dict[str, Any]] = []
        elapsed_before_tool = 0.0

        try:
            for action_index, tool in enumerate(trace_tools):
                available_token_ids: list[int] | None = None
                last_visible_chars = -1
                checkpoints = itertools.chain(
                    [(0.0, 0)],
                    iter_visible_checkpoints(
                        tool.output_events,
                        tool.duration_s,
                        len(tool.raw_output),
                        self.prefill_check_interval_s,
                    ),
                )
                for check_time, raw_visible_chars in checkpoints:
                    self.sleep_until(phase_start, elapsed_before_tool + check_time)
                    if phase_deadline is not None and self.now() >= phase_deadline:
                        break
                    visible_chars = self.clamp_stream_output_chars(raw_visible_chars)
                    if visible_chars != last_visible_chars:
                        partial_output = self.streaming_tool_output(tool.raw_output[:visible_chars])
                        available_token_ids = self.available_prefill_token_ids(
                            history_after_assistant=history_after_assistant,
                            actions=actions,
                            completed_outputs=completed_outputs,
                            action_index=action_index,
                            partial_output=partial_output,
                        )
                        last_visible_chars = visible_chars
                    if available_token_ids is None:
                        continue
                    if phase_deadline is not None and self.now() >= phase_deadline:
                        break

                    prefill_token_ids = self.next_prefill_chunk(available_token_ids, last_prefix_len=last_prefix_len)
                    if prefill_token_ids is None:
                        continue
                    prefix_len = len(prefill_token_ids)

                    submitted = worker.submit(
                        AsyncPrefillRequest(
                            token_ids=prefill_token_ids,
                            cache_salt=cache_salt,
                            step=ReplayStep(instance_id=instance_id, step_index=step_index),
                            label="tool_output",
                            request_id=f"{cache_salt}:tool_output:{uuid.uuid4().hex}",
                        )
                    )
                    if not submitted:
                        continue
                    stats["prefill_submitted_count"] += 1
                    last_prefix_len = prefix_len

                self.sleep_until(phase_start, elapsed_before_tool + max(0.0, tool.duration_s))
                completed_outputs.append(tool.output)
                elapsed_before_tool += max(0.0, tool.duration_s)

            tool_end = phase_deadline if phase_deadline is not None else self.now()
            worker.raise_if_error()
            completed = [completion for completion in worker.completed_prefills() if completion.finished_at <= tool_end]
            completed_prefix_len = max(
                (completion.prefix_len for completion in completed),
                default=prefill_seed.cached_prefix_len,
            )
            stats["prefill_completed_count"] = len(completed)
            stats["prefill_completed_prompt_tokens"] = completed_prefix_len
            stats["prefilled_prompt_suffix_tokens"] = max(
                0,
                completed_prefix_len - prefill_seed.cached_prefix_len,
            )
            stats["prefilled_tool_output_tokens"] = max(
                0,
                completed_prefix_len - prefill_seed.seed_prefix_len,
            )
            stats.update(worker.snapshot())
            stats.update(worker.cancel_active())
            return ToolPhaseResult(stats=stats, prefill_seed=prefill_seed)
        finally:
            worker.stop_without_drain()

    def simulate_candidate_tool_phase(
        self,
        *,
        instance_id: str,
        step_index: int,
        cache_salt: str,
        cached_prompt_ids: list[int],
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        trace_tools: list[TraceToolCall],
        candidate_history: list[HistoricalToolCall],
        stats: dict[str, Any],
        total_duration: float,
    ) -> ToolPhaseResult:
        phase_start = self.now()
        prefill_seed = self.prefill_seed(history_after_assistant, cached_prompt_ids=cached_prompt_ids)
        candidate_stats = CandidateToolPrefillPhase(self).run(
            phase_start=phase_start,
            total_duration=total_duration,
            prefill_seed=prefill_seed,
            instance_id=instance_id,
            step_index=step_index,
            cache_salt=cache_salt,
            cached_prompt_ids=cached_prompt_ids,
            history_after_assistant=history_after_assistant,
            actions=actions,
            trace_tools=trace_tools,
            candidate_history=candidate_history,
        )
        stats.update(candidate_stats)
        return ToolPhaseResult(stats=stats, prefill_seed=prefill_seed)

    def new_prefill_worker(self, *, max_pending: int) -> AsyncPrefillWorker:
        return AsyncPrefillWorker(
            self.backend,
            now=self.now,
            max_pending=max_pending,
        )

    def candidate_prompt_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        candidate_output: dict[str, Any],
    ) -> list[int]:
        candidate_messages = history_after_assistant + self.observation_messages(
            actions[:1],
            [candidate_output],
            streaming=True,
        )
        return self.tokenizer.encode_messages_with_state(
            candidate_messages,
            add_generation_prompt=True,
        ).token_ids

    def final_tool_prompt_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        trace_tools: list[TraceToolCall],
    ) -> list[int]:
        final_messages = history_after_assistant + self.observation_messages(
            actions,
            [tool.output for tool in trace_tools],
        )
        return self.tokenizer.encode_messages_with_state(
            final_messages,
            add_generation_prompt=True,
        ).token_ids

    def visible_checkpoints(self, tool: TraceToolCall) -> Iterator[tuple[float, int]]:
        return itertools.chain(
            [(0.0, 0)],
            iter_visible_checkpoints(
                tool.output_events,
                tool.duration_s,
                len(tool.raw_output),
                self.prefill_check_interval_s,
            ),
        )

    def available_prefill_token_ids(
        self,
        *,
        history_after_assistant: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        completed_outputs: list[dict[str, Any]],
        action_index: int,
        partial_output: dict[str, Any],
    ) -> list[int]:
        partial_messages = history_after_assistant + self.observation_messages(
            actions[: action_index + 1],
            completed_outputs + [partial_output],
            streaming=True,
        )
        partial_state = self.tokenizer.encode_messages_with_state(partial_messages, add_generation_prompt=True)
        prefix_len = align_down(len(partial_state.token_ids) - self.prefill_safety_tail_tokens, self.cache_block_tokens)
        return partial_state.token_ids[:prefix_len]

    def next_prefill_chunk(self, available_token_ids: list[int], *, last_prefix_len: int) -> list[int] | None:
        chunk_tokens = align_down(self.prefill_chunk_tokens, self.cache_block_tokens)
        if chunk_tokens <= 0:
            chunk_tokens = self.cache_block_tokens
        prefix_len = last_prefix_len + chunk_tokens
        if prefix_len <= last_prefix_len:
            return None
        if len(available_token_ids) < prefix_len:
            return None
        return available_token_ids[:prefix_len]

    def clamp_stream_output_chars(self, visible_chars: int) -> int:
        if self.stream_output_char_limit is None:
            return visible_chars
        return min(visible_chars, max(0, self.stream_output_char_limit))

    def streaming_tool_output(self, output: str) -> dict[str, Any]:
        return {
            "output": output,
            "returncode": "",
            "exception_info": "",
            "extra": {},
        }

    def observation_messages(
        self,
        actions: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        *,
        streaming: bool = False,
    ) -> list[dict[str, Any]]:
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.stream_observation_template if streaming else self.observation_template,
            template_vars={},
            multimodal_regex=str(self.model_config.get("multimodal_regex") or ""),
        )

    def sleep_scaled(self, seconds: float) -> None:
        if self.time_scale <= 0 or seconds <= 0:
            return
        self.sleep(seconds * self.time_scale)

    def sleep_until(self, phase_start: float, replay_elapsed_s: float) -> None:
        if self.time_scale <= 0:
            return
        target = phase_start + max(0.0, replay_elapsed_s) * self.time_scale
        delay = target - self.now()
        if delay > 0:
            self.sleep(delay)


def prepare_replay_scenario(path: Path, data: dict[str, Any]) -> ReplayScenario:
    messages = data.get("messages") or []
    instance_id = instance_id_from_data(path, data)
    turns: list[ReplayTurn] = []
    leading_messages: list[dict[str, Any]] = []
    step_index = 0
    index = 0

    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "exit":
            break
        if not is_assistant_message(message):
            leading_messages.append(copy.deepcopy(message))
            index += 1
            continue

        next_index = next_assistant_index(messages, index + 1)
        tool_messages = tool_messages_after(messages, index + 1)
        after_tools_index = index + 1 + len(tool_messages)
        model_call = copy.deepcopy(trace_model_call(message))
        trace_completion_tokens = int_or_none(model_call.get("completion_tokens"))
        if trace_completion_tokens is None:
            return ReplayScenario(
                path=path,
                instance_id=instance_id,
                turns=turns,
                terminal_invalid=invalid_record(path, instance_id, step_index, "missing_trace_completion_tokens"),
            )

        assistant = copy.deepcopy(message)
        actions = actions_from_message(assistant)
        turns.append(
            ReplayTurn(
                step_index=step_index,
                leading_messages=leading_messages,
                assistant=assistant,
                actions=actions,
                trace_tools=trace_tool_calls(actions, tool_messages, assistant),
                model_call=model_call,
                trace_completion_tokens=trace_completion_tokens,
                has_next_assistant=next_index is not None,
                next_turn_follows_tools=next_index == after_tools_index,
            )
        )
        leading_messages = []
        step_index += 1
        index = after_tools_index

    return ReplayScenario(path=path, instance_id=instance_id, turns=turns)


def records_from_measurements(
    scenario: ReplayScenario,
    algorithm: ReplayAlgorithm,
    measurements: list[StepMeasurement],
) -> list[dict[str, Any]]:
    return [record_from_measurement(scenario, algorithm, measurement) for measurement in measurements]


def record_from_measurement(
    scenario: ReplayScenario,
    algorithm: ReplayAlgorithm,
    measurement: StepMeasurement,
) -> dict[str, Any]:
    turn = measurement.turn
    model_call = turn.model_call
    record = {
        "trajectory_path": str(scenario.path),
        "instance_id": scenario.instance_id,
        "algorithm": algorithm,
        "step_index": turn.step_index,
        "trace_prompt_tokens": int_or_none(model_call.get("prompt_tokens")),
        "trace_completion_tokens": turn.trace_completion_tokens,
        "requested_completion_tokens": turn.trace_completion_tokens,
        "trace_ttft_s": number(model_call.get("ttft_s")),
        "trace_model_total_s": number(model_call.get("model_total_s")),
        "trace_decode_s": number(model_call.get("decode_s")),
        "trace_finish_reason": model_call.get("finish_reason"),
        "valid": measurement.valid,
        "skip_reason": measurement.skip_reason,
    }
    if measurement.prompt_tokens is not None:
        record["prompt_tokens"] = measurement.prompt_tokens
    if measurement.replay_ttft_s is not None:
        record["replay_ttft_s"] = measurement.replay_ttft_s
    if measurement.replay_model_total_s is not None:
        record["replay_model_total_s"] = measurement.replay_model_total_s
    if measurement.replay_decode_s is not None:
        record["replay_decode_s"] = measurement.replay_decode_s
    if measurement.replay_completion_tokens is not None:
        record["replay_completion_tokens"] = measurement.replay_completion_tokens
    if measurement.cached_tokens is not None:
        record["cached_tokens"] = measurement.cached_tokens
    if measurement.problem_e2e_s is not None:
        record["problem_e2e_s"] = measurement.problem_e2e_s
    record.update(measurement.tool_stats)
    return record


def skipped_measurements_from_turns(
    turns: list[ReplayTurn],
    start_turn_index: int,
    reason: str,
) -> list[StepMeasurement]:
    normalized = "skipped_capacity" if is_capacity_error(reason) else reason
    measurements = []
    first = True
    for turn in turns[start_turn_index:]:
        skip_reason = "skipped_after_capacity" if normalized == "skipped_capacity" and not first else normalized
        measurements.append(StepMeasurement(turn=turn, valid=False, skip_reason=skip_reason))
        first = False
    return measurements


def trace_tool_calls(
    actions: list[dict[str, Any]],
    tool_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> list[TraceToolCall]:
    calls = []
    previous_timestamp = number((assistant_message.get("extra") or {}).get("timestamp"))
    for index, _ in enumerate(actions):
        message = tool_messages[index] if index < len(tool_messages) else {}
        output = output_from_tool_message(message)
        metric = tool_metric_for_message(message)
        timestamp = number((message.get("extra") or {}).get("timestamp"))
        fallback_duration = (
            max(0.0, timestamp - previous_timestamp)
            if timestamp is not None and previous_timestamp is not None
            else 0.0
        )
        duration = first_number(metric.get("duration_s") if metric else None, fallback_duration) or 0.0
        events = list(metric.get("output_events") or []) if metric else []
        calls.append(
            TraceToolCall(
                output=output,
                duration_s=duration,
                output_events=normalize_output_events(events, duration, len(str(output.get("output") or ""))),
                missing_timing=metric is None,
            )
        )
        previous_timestamp = timestamp if timestamp is not None else previous_timestamp
    return calls


def output_from_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") or {}
    output = extra.get("raw_output")
    if output is None:
        output = message.get("content") or ""
    return {
        "output": str(output or ""),
        "returncode": extra.get("returncode", 0),
        "exception_info": extra.get("exception_info", ""),
        "extra": {},
    }


def tool_metric_for_message(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = ((message.get("extra") or {}).get("token_timing") or {}).get("tool_calls") or []
    return calls[0] if calls else None


def normalize_output_events(events: list[dict[str, Any]], duration_s: float, output_chars: int) -> list[dict[str, Any]]:
    normalized = []
    for event in events:
        t = number(event.get("t"))
        chars = int_or_none(event.get("output_chars"))
        if t is None or chars is None:
            continue
        normalized.append({"t": max(0.0, t), "output_chars": max(0, min(chars, output_chars))})
    normalized.sort(key=lambda event: event["t"])
    return normalized


def iter_visible_checkpoints(
    events: list[dict[str, Any]],
    duration_s: float,
    output_chars: int,
    interval_s: float,
) -> Iterator[tuple[float, int]]:
    if interval_s <= 0:
        yield from event_time_checkpoints(events, duration_s, output_chars)
        return

    index = 1
    event_index = 0
    visible = 0
    while (check := round(index * interval_s, 12)) < duration_s:
        while event_index < len(events) and (number(events[event_index].get("t")) or 0.0) <= check:
            visible = max(visible, int_or_none(events[event_index].get("output_chars")) or 0)
            event_index += 1
        yield max(0.0, check), max(0, min(visible, output_chars))
        index += 1


def event_time_checkpoints(
    events: list[dict[str, Any]],
    duration_s: float,
    output_chars: int,
) -> Iterator[tuple[float, int]]:
    check_time = None
    visible = 0
    for event in events:
        event_time = max(0.0, number(event.get("t")) or 0.0)
        if event_time >= duration_s:
            break
        if check_time is None:
            check_time = event_time
        if event_time != check_time:
            yield check_time, max(0, min(visible, output_chars))
            check_time = event_time
        visible = max(visible, int_or_none(event.get("output_chars")) or 0)
    if check_time is not None:
        yield check_time, max(0, min(visible, output_chars))


def collect_trajectory_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.traj.json"))
    raise typer.BadParameter(f"Replay path does not exist: {path}")


def load_trajectory(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"messages": data, "trajectory_format": "message-list"}
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported trajectory JSON in {path}")
    return data


def instance_id_from_data(path: Path, data: dict[str, Any]) -> str:
    return str(data.get("instance_id") or data.get("info", {}).get("swebench_pro", {}).get("instance_id") or path.stem)


def trace_model_call(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") or {}
    token_timing = extra.get("token_timing") or {}
    model_call = token_timing.get("model_call") or {}
    if model_call:
        return model_call
    usage = (extra.get("response") or {}).get("usage") or {}
    timing = extra.get("model_timing") or {}
    choices = (extra.get("response") or {}).get("choices") or [{}]
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "ttft_s": timing.get("ttft_s"),
        "model_total_s": timing.get("model_total_s"),
        "decode_s": timing.get("decode_s"),
    }


def actions_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    actions = message.get("extra", {}).get("actions") or []
    if actions:
        return [dict(action) for action in actions]
    parsed = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if function.get("name") == "bash" and isinstance(args, dict) and "command" in args:
            parsed.append({"command": args["command"], "tool_call_id": tool_call.get("id")})
    return parsed


def tool_messages_after(messages: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    tools = []
    for message in messages[start:]:
        role = message.get("role")
        if role == "tool":
            tools.append(message)
            continue
        break
    return tools


def is_assistant_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant"


def next_assistant_index(messages: list[dict[str, Any]], start: int) -> int | None:
    for index in range(start, len(messages)):
        role = messages[index].get("role")
        if role == "assistant":
            return index
        if role == "exit":
            return None
    return None


def invalid_record(path: Path, instance_id: str, step_index: int, reason: str) -> dict[str, Any]:
    return {
        "trajectory_path": str(path),
        "instance_id": instance_id,
        "step_index": step_index,
        "valid": False,
        "skip_reason": reason,
    }


def is_capacity_error(reason: str) -> bool:
    lower = reason.lower()
    return "maximum context length" in lower or "input_tokens" in lower or ("context" in lower and "length" in lower)


def number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(*values: Any) -> float | None:
    for value in values:
        parsed = number(value)
        if parsed is not None:
            return parsed
    return None


def int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def align_down(value: int, block_size: int) -> int:
    value = max(0, value)
    if block_size <= 1:
        return value
    return (value // block_size) * block_size


def common_prefix_length(first: list[int], second: list[int]) -> int:
    for index, (first_token, second_token) in enumerate(zip(first, second)):
        if first_token != second_token:
            return index
    return min(len(first), len(second))


def mistral_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    safe_messages = [
        {key: copy.deepcopy(value) for key, value in message.items() if key in allowed} for message in messages
    ]
    tool_ids: dict[str, str] = {}

    def safe_id(value: Any) -> str:
        key = str(value or f"missing_{len(tool_ids)}")
        if key not in tool_ids:
            tool_ids[key] = f"tc{len(tool_ids):07d}"
        return tool_ids[key]

    for message in safe_messages:
        if message.get("content") is None:
            message["content"] = ""
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                tool_call["id"] = safe_id(tool_call.get("id"))
        if "tool_call_id" in message:
            message["tool_call_id"] = safe_id(message.get("tool_call_id"))
    return safe_messages


def config_from_trajectory(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("info", {}).get("config", {})


def load_replay_config(first_data: dict[str, Any], config_spec: list[str]) -> dict[str, Any]:
    configs = [config_from_trajectory(first_data)]
    configs.extend(get_config_from_spec(spec) for spec in config_spec)
    return recursive_merge(*configs)


def tokenizer_from_config(config: dict[str, Any]) -> ReplayTokenizer:
    agent_config = config.get("agent") or {}
    return ReplayTokenizer.from_path(
        str(agent_config.get("tokenizer_path") or ""),
        local_files_only=bool(agent_config.get("tokenizer_local_files_only", True)),
    )


def backend_from_config(config: dict[str, Any]) -> HttpReplayBackend:
    model_config = config.get("model") or {}
    model_kwargs = model_config.get("model_kwargs") or {}
    replay_config = config.get("replay") or {}
    return HttpReplayBackend(
        model_name=str(replay_config.get("served_model_name") or model_config.get("model_name") or ""),
        api_base=str(replay_config.get("api_base") or model_kwargs.get("api_base") or ""),
        prefill_url=str(replay_config.get("prefill_url") or model_kwargs.get("prefill_url") or ""),
        completion_url=str(replay_config.get("completion_url") or replay_config.get("completions_url") or ""),
        timeout=float(replay_config.get("timeout") or 600),
        ignore_eos=bool_config(replay_config, "ignore_eos", True),
    )


def max_context_tokens_from_config(config: dict[str, Any]) -> int | None:
    value = (config.get("replay") or {}).get("max_context_tokens")
    if value in ("", None):
        return None
    return int(value)


def int_config(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    return default if value in ("", None) else int(value)


def float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    return default if value in ("", None) else float(value)


def optional_int_config(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    return None if value in ("", None) else int(value)


def bool_config(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    if value in ("", None):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def runner_kwargs(config: dict[str, Any], *, algorithm: ReplayAlgorithm) -> dict[str, Any]:
    replay_config = config.get("replay") or {}
    candidate_config = replay_config.get("candidate_prefill") or {}
    cache_block_tokens = int_config(replay_config, "cache_block_tokens", DEFAULT_CACHE_BLOCK_TOKENS)
    prefill_chunk_tokens = int_config(
        replay_config,
        "prefill_chunk_tokens",
        int_config(replay_config, "prefill_min_new_tokens", DEFAULT_PREFILL_CHUNK_TOKENS),
    )
    return {
        "algorithm": algorithm,
        "max_context_tokens": max_context_tokens_from_config(config),
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_check_interval_s": float_config(
            replay_config,
            "prefill_check_interval_s",
            float_config(replay_config, "prefill_min_interval_s", DEFAULT_PREFILL_CHECK_INTERVAL_S),
        ),
        "prefill_safety_tail_tokens": int_config(
            replay_config,
            "prefill_safety_tail_tokens",
            DEFAULT_PREFILL_SAFETY_TAIL_TOKENS,
        ),
        "stream_output_char_limit": optional_int_config(replay_config, "stream_output_char_limit"),
        "cache_block_tokens": cache_block_tokens,
        "candidate_top_k": int_config(candidate_config, "top_k", DEFAULT_CANDIDATE_TOP_K),
        "time_scale": float_config(replay_config, "time_scale", DEFAULT_TIME_SCALE),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_replay_outputs(
    output: Path, records: list[dict[str, Any]], invalid_records: list[dict[str, Any]]
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    output_records = [record_for_output(record) for record in records]
    output_invalid = [
        record_for_output(record)
        for record in [record for record in records if not record.get("valid")] + invalid_records
    ]
    write_jsonl(output / "replay_results.jsonl", output_records)
    write_jsonl(output / "invalid_steps.jsonl", output_invalid)
    summary = summarize(records, invalid_records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@app.command(help="Replay saved trajectories as trace-driven serving workloads.")
def main(
    path: Path = typer.Argument(..., help="A .traj.json file or directory containing trajectories."),
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for replay results."),
    algorithm: ReplayAlgorithm = typer.Option(
        "baseline",
        "--algorithm",
        help="Replay algorithm: baseline, chunked, or candidate.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit the number of trajectory files."),
    config_spec: list[str] = typer.Option(
        [], "-c", "--config", help="Config overrides, merged after trajectory config."
    ),
) -> None:
    trajectory_files = collect_trajectory_files(path)
    if limit is not None:
        trajectory_files = trajectory_files[:limit]
    if not trajectory_files:
        raise typer.BadParameter(f"No trajectory files found in {path}")

    first_data = load_trajectory(trajectory_files[0])
    config = load_replay_config(first_data, config_spec)
    tokenizer = tokenizer_from_config(config)
    backend = backend_from_config(config)

    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    for trajectory_path in trajectory_files:
        data: dict[str, Any] | None = None
        try:
            data = first_data if trajectory_path == trajectory_files[0] else load_trajectory(trajectory_path)
            trajectory_config = load_replay_config(data, config_spec)
            runner = TraceReplayRunner(
                backend,
                tokenizer,
                trajectory_config,
                **runner_kwargs(trajectory_config, algorithm=algorithm),
            )
            replay_records, invalid = runner.run_trajectory(trajectory_path, data)
            records.extend(replay_records)
            invalid_records.extend(invalid)
        except Exception as e:
            instance_id = instance_id_from_data(trajectory_path, data) if data is not None else trajectory_path.stem
            invalid_records.append(
                invalid_record(trajectory_path, instance_id, 0, f"trajectory_failed:{type(e).__name__}")
            )
            console.print(f"[bold yellow]Skipped {trajectory_path}: {type(e).__name__}: {e}[/bold yellow]")
        summary = write_replay_outputs(output, records, invalid_records)

    console.print(
        f"Replay complete: [bold green]{summary['valid']} valid[/bold green], "
        f"[bold yellow]{summary['skipped']} skipped[/bold yellow]. "
        f"Results saved to [bold green]{output}[/bold green]"
    )


if __name__ == "__main__":
    app()
