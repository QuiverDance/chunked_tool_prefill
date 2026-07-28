#!/usr/bin/env python3

"""Replay saved trajectories by prefilling each completed tool result."""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from minisweagent.run.replay import (
    AsyncPrefillWorker,
    ReplayTurn,
    ToolPhaseResult,
    TraceReplayRunner,
    TraceToolCall,
    backend_from_config,
    empty_tool_stats,
    finalize_prefill_stats,
    instance_id_from_data,
    invalid_record,
    load_replay_config,
    runner_kwargs,
    tokenizer_from_config,
    write_replay_outputs,
)
from minisweagent.run.replay_sources import SWEChatFormat, collect_replay_sources, load_replay_source
from minisweagent.run.replay_types import AsyncPrefillRequest, PromptTokenState, ReplayStep

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

IncrementalReplayAlgorithm = Literal["baseline", "incremental"]


@dataclass(frozen=True)
class CompletedToolCall:
    completed_at_s: float
    action: dict[str, Any]
    trace: TraceToolCall


class IncrementalReplayRunner(TraceReplayRunner):
    def prefill_enabled_for_turn(self, turn: ReplayTurn) -> bool:
        return self.algorithm == "incremental" and turn.has_next_assistant

    def final_tool_messages(self, turn: ReplayTurn) -> list[dict[str, Any]]:
        return self.observation_messages_for(completed_tool_calls(turn.actions, turn.trace_tools))

    def encode_messages_with_state(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> PromptTokenState:
        return self.tokenizer.encode_messages_preserving_tool_result_order(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

    def observation_messages_for(self, calls: list[CompletedToolCall]) -> list[dict[str, Any]]:
        return self.observation_messages(
            [call.action for call in calls],
            [call.trace.output for call in calls],
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
        candidate_history: list[Any],
        prefill_enabled: bool,
    ) -> ToolPhaseResult:
        completed = completed_tool_calls(actions, trace_tools)
        tool_duration_s = max((call.completed_at_s for call in completed), default=0.0)
        stats = empty_tool_stats(trace_tools, tool_duration_s)
        prefill_events = [call for call in completed if call.completed_at_s < tool_duration_s]
        if not prefill_enabled or not prefill_events:
            self.sleep_scaled(tool_duration_s)
            return ToolPhaseResult(stats=stats, prefill_seed=None)

        phase_start = self.now()
        phase_deadline = phase_start + tool_duration_s * self.time_scale if self.time_scale > 0 else None
        prefill_seed = self.prefill_seed(
            history_after_assistant,
            cached_prompt_ids=cached_prompt_ids,
        )
        if phase_deadline is not None and self.now() >= phase_deadline:
            return ToolPhaseResult(stats=stats, prefill_seed=prefill_seed)

        event_groups = [
            (completed_at_s, list(group))
            for completed_at_s, group in itertools.groupby(
                prefill_events,
                key=lambda call: call.completed_at_s,
            )
        ]
        worker = AsyncPrefillWorker(
            self.backend,
            now=self.now,
            max_pending=1,
        )
        completed_so_far: list[CompletedToolCall] = []

        try:
            for completed_at_s, group in event_groups:
                self.sleep_until(phase_start, completed_at_s)
                if phase_deadline is not None and self.now() >= phase_deadline:
                    break
                completed_so_far.extend(group)
                partial_messages = history_after_assistant + self.observation_messages_for(completed_so_far)
                partial_ids = self.encode_messages_with_state(
                    partial_messages,
                    add_generation_prompt=False,
                ).token_ids
                if phase_deadline is not None and self.now() >= phase_deadline:
                    break
                if self.max_context_tokens is not None and len(partial_ids) > self.max_context_tokens:
                    continue

                request = AsyncPrefillRequest(
                    token_ids=partial_ids,
                    cache_salt=cache_salt,
                    step=ReplayStep(instance_id=instance_id, step_index=step_index),
                    label="completed_tool_output",
                    request_id=f"{cache_salt}:completed_tool_output:{uuid.uuid4().hex}",
                )
                submitted, replaced = worker.submit_latest(request)
                stats["prefill_coalesced_count"] += replaced
                if submitted:
                    stats["prefill_submitted_count"] += 1

            self.sleep_until(phase_start, tool_duration_s)
            worker.raise_if_error()
            tool_end = phase_deadline if phase_deadline is not None else self.now()
            finalize_prefill_stats(stats, worker, prefill_seed, tool_end=tool_end)
            return ToolPhaseResult(stats=stats, prefill_seed=prefill_seed)
        finally:
            worker.stop_and_wait()


def completed_tool_calls(
    actions: list[dict[str, Any]],
    trace_tools: list[TraceToolCall],
) -> list[CompletedToolCall]:
    calls = []
    sequential_completion_s = 0.0
    for action, trace in zip(actions, trace_tools):
        sequential_completion_s += max(0.0, trace.duration_s)
        completed_at_s = trace.completion_offset_s
        if completed_at_s is None:
            completed_at_s = sequential_completion_s
        calls.append(
            CompletedToolCall(
                completed_at_s=max(0.0, completed_at_s),
                action=action,
                trace=trace,
            )
        )
    calls.sort(key=lambda call: call.completed_at_s)
    return calls


@app.command(help="Replay saved trajectories with baseline or incremental tool-result prefill.")
def main(
    path: Path = typer.Argument(
        ...,
        help="A replay trace, a SWE-chat trace, or a directory containing either.",
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for replay results."),
    algorithm: IncrementalReplayAlgorithm = typer.Option(
        "incremental",
        "--algorithm",
        help="Replay algorithm: baseline or incremental.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit the number of trajectory files."),
    swe_chat_format: SWEChatFormat = typer.Option(
        "opencode-json",
        "--swe-chat-format",
        help="SWE-chat trace format selected from a dataset root index.",
    ),
    config_spec: list[str] = typer.Option(
        [],
        "-c",
        "--config",
        help="Config overrides, merged after trajectory config.",
    ),
) -> None:
    trajectory_files = collect_replay_sources(path, swe_chat_format=swe_chat_format)
    if limit is not None:
        trajectory_files = trajectory_files[:limit]
    if not trajectory_files:
        raise typer.BadParameter(f"No trajectory files found in {path}")

    first_data = load_replay_source(trajectory_files[0])
    config = load_replay_config(first_data, config_spec)
    tokenizer = tokenizer_from_config(config)
    backend = backend_from_config(config)
    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for trajectory_path in trajectory_files:
        backend.reset_prefix_cache()
        data: dict[str, Any] | None = None
        try:
            data = first_data if trajectory_path == trajectory_files[0] else load_replay_source(trajectory_path)
            trajectory_config = load_replay_config(data, config_spec)
            runner = IncrementalReplayRunner(
                backend,
                tokenizer,
                trajectory_config,
                **runner_kwargs(trajectory_config, algorithm=algorithm),
            )
            replay_records, invalid = runner.run_trajectory(trajectory_path, data)
            records.extend(replay_records)
            invalid_records.extend(invalid)
        except Exception as error:
            instance_id = instance_id_from_data(trajectory_path, data) if data is not None else trajectory_path.stem
            invalid_records.append(
                invalid_record(
                    trajectory_path,
                    instance_id,
                    0,
                    f"trajectory_failed:{type(error).__name__}",
                )
            )
            console.print(f"[bold yellow]Skipped {trajectory_path}: {type(error).__name__}: {error}[/bold yellow]")

    summary = write_replay_outputs(output, records, invalid_records)
    console.print(
        f"Replay complete: [bold green]{summary['valid']} valid[/bold green], "
        f"[bold yellow]{summary['skipped']} skipped[/bold yellow]. "
        f"Results saved to [bold green]{output}[/bold green]"
    )


if __name__ == "__main__":
    app()
