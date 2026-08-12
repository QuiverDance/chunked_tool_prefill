#!/usr/bin/env python3

"""Prepare and run token-native TraceLab Codex replay trials."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import requests
import typer
from rich.console import Console

from minisweagent.run.replay import AsyncPrefillWorker, ToolPrefillSeed, finalize_prefill_stats
from minisweagent.run.replay_backend import HttpReplayBackend
from minisweagent.run.replay_metrics import stats
from minisweagent.run.replay_types import AsyncPrefillRequest, ReplayError, ReplayStep
from minisweagent.run.tracelab_workload import (
    DEFAULT_SAMPLE_SEED,
    PreparedTraceLabTrial,
    PreparedTraceLabWorkload,
    load_prepared_workload,
    prepare_tracelab_codex_workload,
    write_prepared_workload,
)

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

ReplayAlgorithm = Literal["baseline", "incremental"]


class PairedTrialRetry(ReplayError):
    """Raised on both runners when either trial result fails authority checks."""


class TraceLabBackend(Protocol):
    def reset_prefix_cache(self) -> None: ...

    def prefill(
        self,
        token_ids: list[int],
        *,
        cache_salt: str,
        step: ReplayStep,
        label: str,
        request_id: str | None = None,
    ) -> None: ...

    def cancel_prefill(self, request_id: str) -> None: ...

    def generate_tokens(
        self,
        token_ids: list[int],
        *,
        max_tokens: int,
        cache_salt: str,
        step: ReplayStep,
        label: str,
    ) -> dict[str, Any]: ...


class MeasurementBarrier(Protocol):
    def ready(self) -> None: ...

    def before_trial(self, trial_index: int, attempt: int) -> None: ...

    def after_trial(self, trial_index: int, attempt: int, error: ReplayError | None) -> None: ...

    def fail(self, trial_index: int, error: Exception) -> None: ...


@dataclass(frozen=True)
class MaterializedTrial:
    trial: PreparedTraceLabTrial
    prompt_ids: list[int]
    seed_ids: list[int]
    checkpoint_ids: tuple[list[int], ...]


class TraceLabReplayRunner:
    """Run prepared trials without parsing or token construction inside timed regions."""

    def __init__(
        self,
        backend: TraceLabBackend,
        *,
        algorithm: ReplayAlgorithm,
        time_scale: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.perf_counter,
        run_id: str | None = None,
        measurement_barrier: MeasurementBarrier | None = None,
    ):
        if time_scale < 0:
            raise ValueError("time_scale must be nonnegative")
        self.backend = backend
        self.algorithm = algorithm
        self.time_scale = time_scale
        self.sleep = sleep
        self.now = now
        self.run_id = run_id or uuid.uuid4().hex
        self.measurement_barrier = measurement_barrier

    def run_workload(
        self,
        workload: PreparedTraceLabWorkload,
        *,
        warmup_trials: int = 1,
        start_index: int = 0,
        max_trial_attempts: int = 3,
        record_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if warmup_trials < 0:
            raise ValueError("warmup_trials must be nonnegative")
        if not 0 <= start_index <= len(workload.trials):
            raise ValueError("start_index must be within the prepared workload")
        if max_trial_attempts <= 0:
            raise ValueError("max_trial_attempts must be positive")

        if warmup_trials and workload.trials:
            warmup_trial = min(
                workload.trials,
                key=lambda trial: (
                    trial.prompt_tokens + trial.completion_tokens,
                    trial.tool_phase_duration_s,
                    trial.trial_id,
                ),
            )
            materialized_warmup = materialize_trial(warmup_trial)
            for warmup_index in range(warmup_trials):
                try:
                    self.run_trial(
                        materialized_warmup,
                        trial_index=-1 - warmup_index,
                        time_scale_override=0.0,
                    )
                except Exception as error:
                    if self.measurement_barrier is not None:
                        self.measurement_barrier.fail(-1 - warmup_index, error)
                    raise

        if self.measurement_barrier is not None:
            self.measurement_barrier.ready()

        records: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for trial_index, trial in enumerate(workload.trials[start_index:], start=start_index):
            materialized = materialize_trial(trial)
            for attempt in range(max_trial_attempts):
                try:
                    record = self.run_trial(
                        materialized,
                        trial_index=trial_index,
                        attempt=attempt,
                    )
                except PairedTrialRetry as error:
                    if attempt + 1 < max_trial_attempts:
                        continue
                    if self.measurement_barrier is not None:
                        self.measurement_barrier.fail(trial_index, error)
                        raise
                    invalid.append(_invalid_trial_record(trial, trial_index, self.algorithm, error))
                    break
                except Exception as error:
                    if self.measurement_barrier is not None:
                        self.measurement_barrier.fail(trial_index, error)
                        raise
                    invalid.append(_invalid_trial_record(trial, trial_index, self.algorithm, error))
                    break
                records.append(record)
                if record_callback is not None:
                    record_callback(record)
                break
        return records, invalid

    def run_trial(
        self,
        materialized: MaterializedTrial,
        *,
        trial_index: int,
        attempt: int = 0,
        time_scale_override: float | None = None,
    ) -> dict[str, Any]:
        """Execute one trial; setup and materialization are explicitly outside E2E."""
        trial = materialized.trial
        scale = self.time_scale if time_scale_override is None else time_scale_override
        cache_salt = f"tracelab:{self.run_id}:{self.algorithm}:{trial_index}:{attempt}"
        step = ReplayStep(instance_id=trial.trial_id, step_index=trial.next_round_index)

        reset_started = self.now()
        self.backend.reset_prefix_cache()
        reset_finished = self.now()
        if materialized.seed_ids:
            self.backend.prefill(
                materialized.seed_ids,
                cache_salt=cache_salt,
                step=step,
                label="reported_cached_prefix_seed",
                request_id=f"{cache_salt}:seed",
            )
        setup_finished = self.now()

        worker = None
        if self.algorithm == "incremental" and trial.prefill_checkpoints:
            worker = AsyncPrefillWorker(self.backend, now=self.now, max_pending=1)

        tool_stats = _empty_stats(trial)
        if self.measurement_barrier is not None and trial_index >= 0:
            self.measurement_barrier.before_trial(trial_index, attempt)
        phase_started = self.now()
        try:
            if worker is None:
                self._sleep_until(phase_started, trial.tool_phase_duration_s, scale)
            else:
                for checkpoint, checkpoint_ids in zip(
                    trial.prefill_checkpoints,
                    materialized.checkpoint_ids,
                    strict=True,
                ):
                    self._sleep_until(phase_started, checkpoint.at_s, scale)
                    request = AsyncPrefillRequest(
                        token_ids=checkpoint_ids,
                        cache_salt=cache_salt,
                        step=step,
                        label="tracelab_completed_results",
                        request_id=f"{cache_salt}:checkpoint:{checkpoint.available_result_count}:{uuid.uuid4().hex}",
                    )
                    submitted, replaced = worker.submit_latest(request)
                    tool_stats["prefill_coalesced_count"] += replaced
                    if submitted:
                        tool_stats["prefill_submitted_count"] += 1

                self._sleep_until(phase_started, trial.tool_phase_duration_s, scale)
                worker.raise_if_error()
                tool_end = phase_started + trial.tool_phase_duration_s * scale if scale > 0 else self.now()
                finalize_prefill_stats(
                    tool_stats,
                    worker,
                    ToolPrefillSeed(
                        cached_prefix_len=trial.prefix_tokens,
                        seed_prefix_len=trial.prefix_tokens + trial.static_suffix_tokens,
                    ),
                    tool_end=tool_end,
                )
        finally:
            if worker is not None:
                worker.stop_and_wait()

        tool_finished = self.now()
        result = self.backend.generate_tokens(
            materialized.prompt_ids,
            max_tokens=trial.completion_tokens,
            cache_salt=cache_salt,
            step=step,
            label="tracelab_next_round_decode",
        )
        finished = self.now()
        replay_completion_tokens = result.get("completion_tokens")
        cached_tokens = result.get("cached_tokens")
        cached_token_readback_inferred_zero = False
        if not isinstance(cached_tokens, int) and trial.prefix_tokens == 0 and worker is None:
            cached_tokens = 0
            cached_token_readback_inferred_zero = True
        validation_error = _final_response_error(
            trial,
            replay_completion_tokens=replay_completion_tokens,
            cached_tokens=cached_tokens,
        )
        if self.measurement_barrier is not None and trial_index >= 0:
            self.measurement_barrier.after_trial(trial_index, attempt, validation_error)
        if validation_error is not None:
            raise validation_error

        assert isinstance(cached_tokens, int)
        final_cached_prompt_suffix_tokens = cached_tokens - trial.prefix_tokens
        final_cached_tool_result_tokens = max(
            0,
            cached_tokens - trial.prefix_tokens - trial.static_suffix_tokens,
        )

        completed_prompt_tokens = max(
            int(tool_stats.get("prefill_completed_prompt_tokens") or 0),
            trial.prefix_tokens,
        )
        tool_stats["unprefilled_prompt_suffix_tokens"] = max(
            0,
            trial.prompt_tokens - completed_prompt_tokens,
        )
        tool_stats["unprefilled_tool_output_tokens"] = max(
            0,
            trial.prompt_tokens
            - max(
                completed_prompt_tokens,
                trial.prefix_tokens + trial.static_suffix_tokens,
            ),
        )
        return {
            "trial_id": trial.trial_id,
            "trial_index": trial_index,
            "trial_attempt": attempt,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "valid": True,
            "skip_reason": "",
            "measurement_valid_for_trace_timing": math.isclose(scale, 1.0),
            "synthetic_token_workload": True,
            "source_model": trial.source_model,
            "session_id": trial.session_id,
            "current_round_index": trial.current_round_index,
            "next_round_index": trial.next_round_index,
            "current_trace_key": trial.current_trace_key,
            "next_trace_key": trial.next_trace_key,
            "prompt_tokens": trial.prompt_tokens,
            "trace_prompt_tokens": trial.prompt_tokens,
            "trace_completion_tokens": trial.completion_tokens,
            "requested_completion_tokens": trial.completion_tokens,
            "replay_completion_tokens": replay_completion_tokens,
            "cached_tokens": cached_tokens,
            "cached_token_readback_inferred_zero": cached_token_readback_inferred_zero,
            "replay_ttft_s": result.get("ttft_s"),
            "replay_model_total_s": result.get("model_total_s"),
            "replay_decode_s": result.get("decode_s"),
            "problem_e2e_s": finished - phase_started,
            "tool_phase_elapsed_s": tool_finished - phase_started,
            "time_scale": scale,
            "setup_cache_reset_s": reset_finished - reset_started,
            "setup_seed_prefill_s": setup_finished - reset_finished,
            "setup_total_s": setup_finished - reset_started,
            "reported_prefix_tokens": trial.prefix_tokens,
            "static_suffix_tokens": trial.static_suffix_tokens,
            "result_suffix_tokens": trial.result_suffix_tokens,
            "final_cached_prompt_suffix_tokens": final_cached_prompt_suffix_tokens,
            "final_cached_tool_result_tokens": final_cached_tool_result_tokens,
            "prefill_checkpoint_count": len(trial.prefill_checkpoints),
            **tool_stats,
        }

    def _sleep_until(self, phase_started: float, trace_elapsed_s: float, scale: float) -> None:
        if scale <= 0:
            return
        delay = phase_started + max(0.0, trace_elapsed_s) * scale - self.now()
        if delay > 0:
            self.sleep(delay)


def _final_response_error(
    trial: PreparedTraceLabTrial,
    *,
    replay_completion_tokens: Any,
    cached_tokens: Any,
) -> ReplayError | None:
    if replay_completion_tokens != trial.completion_tokens:
        return ReplayError(f"completion_token_mismatch:{replay_completion_tokens}:{trial.completion_tokens}")
    if not isinstance(cached_tokens, int):
        return ReplayError("missing_cached_token_readback")
    if cached_tokens < trial.prefix_tokens:
        return ReplayError(f"cached_prefix_mismatch:{cached_tokens}:{trial.prefix_tokens}")
    if cached_tokens > trial.prompt_tokens:
        return ReplayError(f"cached_prompt_overflow:{cached_tokens}:{trial.prompt_tokens}")
    return None


def _invalid_trial_record(
    trial: PreparedTraceLabTrial,
    trial_index: int,
    algorithm: ReplayAlgorithm,
    error: Exception,
) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "trial_index": trial_index,
        "algorithm": algorithm,
        "valid": False,
        "skip_reason": str(error) if isinstance(error, ReplayError) else f"trial_failed:{type(error).__name__}",
    }


class FilePairBarrier:
    """Keep two GPU runners in lockstep without work inside timed regions."""

    def __init__(
        self,
        directory: Path,
        *,
        participant: str,
        peer: str,
        timeout_s: float = 600,
        poll_s: float = 0.01,
    ):
        if not participant or not peer or participant == peer:
            raise ValueError("File barrier requires two distinct participant names")
        if any(not value.replace("_", "").isalnum() for value in (participant, peer)):
            raise ValueError("Barrier participant names must be alphanumeric")
        self.directory = directory
        self.participant = participant
        self.peer = peer
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.directory.mkdir(parents=True, exist_ok=True)

    def ready(self) -> None:
        self._signal("ready", -1, 0, "ready")
        self._wait_for("ready", -1, 0, self.peer)

    def before_trial(self, trial_index: int, attempt: int) -> None:
        self._signal("prepared", trial_index, attempt, "ready")
        self._wait_for("prepared", trial_index, attempt, self.peer)

    def after_trial(self, trial_index: int, attempt: int, error: ReplayError | None) -> None:
        status = "ok" if error is None else f"error:{type(error).__name__}:{error}"
        local = self._signal("measured", trial_index, attempt, status)
        peer = self._wait_for("measured", trial_index, attempt, self.peer)
        outcomes = {
            self.participant: local.read_text().strip(),
            self.peer: peer.read_text().strip(),
        }
        errors = [f"{participant}={outcome}" for participant, outcome in sorted(outcomes.items()) if outcome != "ok"]
        if errors:
            raise PairedTrialRetry("paired_trial_retry:" + ";".join(errors))

    def fail(self, trial_index: int, error: Exception) -> None:
        path = self.directory / f"failed-{self.participant}-{trial_index}"
        path.write_text(f"{type(error).__name__}: {error}\n")

    def _marker(self, stage: str, participant: str, trial_index: int, attempt: int) -> Path:
        return self.directory / f"{stage}-{participant}-{trial_index}-{attempt}"

    def _signal(self, stage: str, trial_index: int, attempt: int, value: str) -> Path:
        target = self._marker(stage, self.participant, trial_index, attempt)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(value + "\n")
        temporary.replace(target)
        return target

    def _wait_for(self, stage: str, trial_index: int, attempt: int, participant: str) -> Path:
        target = self._marker(stage, participant, trial_index, attempt)
        deadline = time.monotonic() + self.timeout_s
        while not target.is_file():
            failures = sorted(self.directory.glob("failed-*"))
            if failures:
                details = failures[0].read_text().strip()
                raise ReplayError(f"peer_measurement_failed:{details}")
            if time.monotonic() >= deadline:
                raise ReplayError(f"measurement_barrier_timeout:{stage}:{trial_index}:{participant}")
            time.sleep(self.poll_s)
        return target


def materialize_trial(trial: PreparedTraceLabTrial) -> MaterializedTrial:
    """Create all token lists before cache setup and measurement begin."""
    pattern = _token_pattern(trial.trial_id)
    repeats = math.ceil(trial.prompt_tokens / len(pattern))
    prompt_ids = (pattern * repeats)[: trial.prompt_tokens]
    seed_ids = prompt_ids[: trial.prefix_tokens]
    checkpoint_ids = tuple(prompt_ids[: checkpoint.prompt_tokens] for checkpoint in trial.prefill_checkpoints)
    return MaterializedTrial(
        trial=trial,
        prompt_ids=prompt_ids,
        seed_ids=seed_ids,
        checkpoint_ids=checkpoint_ids,
    )


def _token_pattern(trial_id: str) -> list[int]:
    digest = hashlib.sha256(trial_id.encode()).digest()
    shift = int.from_bytes(digest[:2], "big") % 4096
    return [1000 + ((shift + index * 37) % 8192) for index in range(256)]


def _empty_stats(trial: PreparedTraceLabTrial) -> dict[str, Any]:
    return {
        "tool_call_count": trial.tool_call_count,
        "simulated_tool_duration_s": trial.tool_phase_duration_s,
        "tool_output_chars": trial.tool_output_chars,
        "tool_output_events": trial.tool_call_count,
        "missing_tool_timing_count": 0,
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
        "prefill_coalesced_count": 0,
        "active_prefill_prefix_len_at_tool_end": None,
        "pending_prefill_prefix_len_at_tool_end": None,
        "active_prefill_cancel_requested_at_tool_end": 0,
        "active_prefill_cancel_latency_s": None,
        "active_prefill_cancel_error": "",
    }


def write_run_outputs(
    output: Path,
    workload: PreparedTraceLabWorkload,
    records: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    *,
    manifest_path: Path,
    algorithm: ReplayAlgorithm,
    time_scale: float,
    warmup_trials: int,
    resumed_from_trial: int,
    api_base: str,
    model_name: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "replay_results.jsonl", records)
    _write_jsonl(output / "invalid_trials.jsonl", invalid)
    valid = [record for record in records if record.get("valid")]
    summary = {
        "workload_format": workload.workload_format,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "source_path": workload.source_path,
        "source_sha256": workload.source_sha256,
        "algorithm": algorithm,
        "api_base": api_base,
        "model_name": model_name,
        "server_version": _server_version(api_base),
        "runner_source_sha256": _sha256(Path(__file__)),
        "workload_source_sha256": _sha256(Path(prepare_tracelab_codex_workload.__code__.co_filename)),
        "time_scale": time_scale,
        "warmup_trials": warmup_trials,
        "resumed_from_trial": resumed_from_trial,
        "measurement_valid_for_trace_timing": math.isclose(time_scale, 1.0),
        "selected_trials": len(workload.trials),
        "valid": len(valid),
        "skipped": len(invalid),
        "replay_ttft_s": stats([record.get("replay_ttft_s") for record in valid]),
        "problem_e2e_s": stats([record.get("problem_e2e_s") for record in valid]),
        "replay_model_total_s": stats([record.get("replay_model_total_s") for record in valid]),
        "setup_total_s_excluded_from_e2e": stats([record.get("setup_total_s") for record in valid]),
        "cached_tokens": stats([record.get("cached_tokens") for record in valid]),
        "final_cached_prompt_suffix_tokens": stats(
            [record.get("final_cached_prompt_suffix_tokens") for record in valid]
        ),
        "final_cached_tool_result_tokens": stats([record.get("final_cached_tool_result_tokens") for record in valid]),
        "prefilled_tool_output_tokens": stats([record.get("prefilled_tool_output_tokens") for record in valid]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _load_resume_records(
    path: Path,
    workload: PreparedTraceLabWorkload,
    algorithm: ReplayAlgorithm,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Resume checkpoint not found: {path}")

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) > len(workload.trials):
        raise ValueError(f"Resume checkpoint has too many records: {path}")
    for trial_index, record in enumerate(records):
        trial = workload.trials[trial_index]
        if record.get("trial_index") != trial_index:
            raise ValueError(f"Non-contiguous resume checkpoint at trial {trial_index}: {path}")
        if record.get("trial_id") != trial.trial_id:
            raise ValueError(f"Resume checkpoint trial mismatch at index {trial_index}: {path}")
        if record.get("algorithm") != algorithm or record.get("valid") is not True:
            raise ValueError(f"Invalid resume checkpoint record at index {trial_index}: {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_version(api_base: str) -> dict[str, Any] | str:
    url = f"{api_base.removesuffix('/v1').rstrip('/')}/version"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        value = response.json()
    except Exception as error:
        return f"unavailable:{type(error).__name__}"
    return value if isinstance(value, dict) else str(value)


@app.command("prepare", help="Prepare a deterministic TraceLab Codex workload manifest.")
def prepare_command(
    source: Path = typer.Argument(..., help="TraceLab syfi_coding_trace.jsonl.gz."),
    output: Path = typer.Option(..., "-o", "--output", help="Prepared workload manifest."),
    limit: int = typer.Option(100, "--limit", min=1, help="Number of stratified trials."),
    all_trials: bool = typer.Option(False, "--all", help="Select every eligible trial instead of sampling."),
    sample_seed: str = typer.Option(DEFAULT_SAMPLE_SEED, "--sample-seed"),
    max_context_tokens: int = typer.Option(131072, "--max-context-tokens"),
    cache_block_tokens: int = typer.Option(16, "--cache-block-tokens"),
    max_completion_tokens: int | None = typer.Option(None, "--max-completion-tokens"),
) -> None:
    workload = prepare_tracelab_codex_workload(
        source,
        limit=None if all_trials else limit,
        sample_seed=sample_seed,
        max_context_tokens=max_context_tokens,
        cache_block_tokens=cache_block_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    write_prepared_workload(output, workload)
    console.print(
        f"Prepared [bold green]{len(workload.trials)} trials[/bold green] in "
        f"[bold green]{output}[/bold green] from "
        f"{workload.scan_counts.get('eligible_trials', 0)} eligible pairs."
    )


@app.command("run", help="Run a prepared TraceLab workload without trace preparation in timed regions.")
def run_command(
    manifest: Path = typer.Argument(..., help="Prepared TraceLab workload manifest."),
    output: Path = typer.Option(..., "-o", "--output", help="Replay output directory."),
    algorithm: ReplayAlgorithm = typer.Option("incremental", "--algorithm"),
    api_base: str = typer.Option(..., "--api-base"),
    model_name: str = typer.Option("mistral-small32-24b", "--model-name"),
    prefill_url: str = typer.Option("", "--prefill-url"),
    timeout: float = typer.Option(600, "--timeout"),
    time_scale: float = typer.Option(1.0, "--time-scale"),
    warmup_trials: int = typer.Option(1, "--warmup-trials"),
    max_trial_attempts: int = typer.Option(3, "--max-trial-attempts", min=1),
    resume: bool = typer.Option(False, "--resume", help="Append after a validated partial result checkpoint."),
    sync_directory: Path | None = typer.Option(None, "--sync-directory"),
    participant: str = typer.Option("", "--participant"),
    peer: str = typer.Option("", "--peer"),
) -> None:
    workload = load_prepared_workload(manifest)
    output.mkdir(parents=True, exist_ok=True)
    partial_results = output / "replay_results.partial.jsonl"
    existing_records = _load_resume_records(partial_results, workload, algorithm) if resume else []
    start_index = len(existing_records)
    replay_prefill_url = prefill_url or f"{api_base.rstrip('/')}/prefill"
    backend = HttpReplayBackend(
        model_name=model_name,
        api_base=api_base,
        prefill_url=replay_prefill_url,
        timeout=timeout,
        ignore_eos=True,
    )
    if any((sync_directory, participant, peer)) and not all((sync_directory, participant, peer)):
        raise typer.BadParameter("--sync-directory, --participant, and --peer must be set together")
    barrier = (
        FilePairBarrier(sync_directory, participant=participant, peer=peer) if sync_directory is not None else None
    )
    runner = TraceLabReplayRunner(
        backend,
        algorithm=algorithm,
        time_scale=time_scale,
        measurement_barrier=barrier,
    )
    partial_mode = "a" if resume else "w"
    with partial_results.open(partial_mode, encoding="utf-8", buffering=1) as partial_file:

        def checkpoint_record(record: dict[str, Any]) -> None:
            partial_file.write(json.dumps(record, sort_keys=True) + "\n")
            partial_file.flush()

        resumed_records, invalid = runner.run_workload(
            workload,
            warmup_trials=warmup_trials,
            start_index=start_index,
            max_trial_attempts=max_trial_attempts,
            record_callback=checkpoint_record,
        )
    records = existing_records + resumed_records
    summary = write_run_outputs(
        output,
        workload,
        records,
        invalid,
        manifest_path=manifest,
        algorithm=algorithm,
        time_scale=time_scale,
        warmup_trials=warmup_trials,
        resumed_from_trial=start_index,
        api_base=api_base,
        model_name=model_name,
    )
    partial_results.unlink()
    console.print(
        f"TraceLab replay complete: [bold green]{summary['valid']} valid[/bold green], "
        f"[bold yellow]{summary['skipped']} skipped[/bold yellow]. "
        f"Results saved to [bold green]{output}[/bold green]"
    )


if __name__ == "__main__":
    app()
