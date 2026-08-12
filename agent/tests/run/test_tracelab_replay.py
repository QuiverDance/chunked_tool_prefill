import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from minisweagent.run.extra.tracelab_replay import (
    FilePairBarrier,
    TraceLabReplayRunner,
    _load_resume_records,
    app,
    materialize_trial,
)
from minisweagent.run.replay_backend import cached_tokens_from_chunk
from minisweagent.run.tracelab_workload import (
    load_prepared_workload,
    prepare_tracelab_codex_workload,
    write_prepared_workload,
)


def trace_rounds() -> list[dict]:
    return [
        {
            "provider": "codex",
            "session_id": "codex:session",
            "round_index": 0,
            "trace_key": "current",
            "model": "gpt-test",
            "input_tokens_total": 100,
            "prefix_tokens": 80,
            "newly_append_tokens": 20,
            "output_tokens": 20,
            "current_user_message_count": 1,
            "current_tool_result_count": 0,
            "current_tool_result_chars": 0,
            "timing_events": [],
            "tools": [
                {
                    "tool_index": 0,
                    "tool_name": "slow",
                    "tool_call_id": "call_slow",
                    "emitted_at": "2026-01-01T00:00:00.000Z",
                    "result_at": "2026-01-01T00:00:00.300Z",
                    "result_chars": 3,
                },
                {
                    "tool_index": 1,
                    "tool_name": "fast",
                    "tool_call_id": "call_fast",
                    "emitted_at": "2026-01-01T00:00:00.010Z",
                    "result_at": "2026-01-01T00:00:00.100Z",
                    "result_chars": 7,
                },
            ],
        },
        {
            "provider": "codex",
            "session_id": "codex:session",
            "round_index": 1,
            "trace_key": "following",
            "model": "gpt-test",
            "input_tokens_total": 130,
            "prefix_tokens": 112,
            "newly_append_tokens": 18,
            "output_tokens": 5,
            "current_user_message_count": 0,
            "current_tool_result_count": 2,
            "current_tool_result_chars": 10,
            "timing_events": [
                {
                    "event_type": "tool_result",
                    "tool_call_id": "call_fast",
                    "result_chars": 7,
                },
                {
                    "event_type": "tool_result",
                    "tool_call_id": "call_slow",
                    "result_chars": 3,
                },
            ],
            "tools": [],
        },
    ]


def write_trace(path: Path, rows: list[dict] | None = None) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows or trace_rounds():
            file.write(json.dumps(row) + "\n")


def test_prepare_builds_exact_token_accounting_and_prefix_checkpoints(tmp_path):
    source = tmp_path / "trace.jsonl.gz"
    write_trace(source)

    workload = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )

    assert workload.scan_counts["eligible_trials"] == 1
    trial = workload.trials[0]
    assert trial.prompt_tokens == 130
    assert trial.prefix_tokens == 112
    assert trial.static_suffix_tokens == 8
    assert trial.result_suffix_tokens == 10
    assert [result.tool_call_id for result in trial.tool_results] == ["call_fast", "call_slow"]
    assert [result.result_tokens for result in trial.tool_results] == [7, 3]
    assert [
        (checkpoint.at_s, checkpoint.prompt_tokens, checkpoint.available_result_count)
        for checkpoint in trial.prefill_checkpoints
    ] == [
        (0.0, 120, 0),
        (0.1, 127, 1),
    ]


def test_prepare_waits_for_a_missing_serialized_prefix_result(tmp_path):
    rows = trace_rounds()
    rows[1]["timing_events"].reverse()
    source = tmp_path / "trace.jsonl.gz"
    write_trace(source, rows)

    workload = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )

    assert [(checkpoint.at_s, checkpoint.prompt_tokens) for checkpoint in workload.trials[0].prefill_checkpoints] == [
        (0.0, 120)
    ]


def test_prepare_omits_prefill_checkpoints_when_tool_phase_has_zero_duration(tmp_path):
    rows = trace_rounds()
    timestamp = "2026-01-01T00:00:00.000Z"
    for tool in rows[0]["tools"]:
        tool["emitted_at"] = timestamp
        tool["result_at"] = timestamp
    source = tmp_path / "trace.jsonl.gz"
    manifest = tmp_path / "workload.json"
    write_trace(source, rows)

    workload = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )
    write_prepared_workload(manifest, workload)

    assert workload.trials[0].tool_phase_duration_s == 0
    assert workload.trials[0].prefill_checkpoints == ()
    assert load_prepared_workload(manifest) == workload


def test_manifest_round_trip_preserves_the_prepared_interface(tmp_path):
    source = tmp_path / "trace.jsonl.gz"
    manifest = tmp_path / "workload.json"
    write_trace(source)
    prepared = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )

    write_prepared_workload(manifest, prepared)
    loaded = load_prepared_workload(manifest)

    assert loaded == prepared


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeBackend:
    def __init__(self, clock: FakeClock | None = None):
        self.clock = clock
        self.prefills = []
        self.generations = []
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1
        if self.clock is not None:
            self.clock.sleep(2.0)

    def prefill(self, token_ids, *, cache_salt, step, label, request_id=None):
        self.prefills.append({"token_ids": list(token_ids), "label": label, "request_id": request_id})
        if self.clock is not None:
            self.clock.sleep(3.0)

    def cancel_prefill(self, request_id):
        return None

    def generate_tokens(self, token_ids, *, max_tokens, cache_salt, step, label):
        self.generations.append({"token_ids": list(token_ids), "max_tokens": max_tokens})
        if self.clock is not None:
            self.clock.sleep(0.2)
        return {
            "ttft_s": 0.05,
            "model_total_s": 0.2,
            "decode_s": 0.15,
            "cached_tokens": 112,
            "completion_tokens": max_tokens,
        }


class MissingCachedTokensOnceBackend(FakeBackend):
    def generate_tokens(self, token_ids, *, max_tokens, cache_salt, step, label):
        result = super().generate_tokens(
            token_ids,
            max_tokens=max_tokens,
            cache_salt=cache_salt,
            step=step,
            label=label,
        )
        if len(self.generations) == 1:
            result["cached_tokens"] = None
        return result


def prepared_trial(tmp_path):
    source = tmp_path / "trace.jsonl.gz"
    write_trace(source)
    workload = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )
    return workload.trials[0]


def test_runner_excludes_reset_seed_and_materialization_from_e2e(tmp_path):
    trial = prepared_trial(tmp_path)
    materialized = materialize_trial(trial)
    clock = FakeClock()
    backend = FakeBackend(clock)
    runner = TraceLabReplayRunner(
        backend,
        algorithm="baseline",
        sleep=clock.sleep,
        now=clock,
        run_id="test",
    )

    record = runner.run_trial(materialized, trial_index=0)

    assert record["setup_total_s"] == pytest.approx(5.0)
    assert record["tool_phase_elapsed_s"] == pytest.approx(0.3)
    assert record["problem_e2e_s"] == pytest.approx(0.5)
    assert record["replay_ttft_s"] == pytest.approx(0.05)
    assert record["final_cached_prompt_suffix_tokens"] == 0
    assert record["final_cached_tool_result_tokens"] == 0


def test_baseline_and_incremental_send_the_same_final_prompt(tmp_path):
    trial = prepared_trial(tmp_path)
    materialized = materialize_trial(trial)
    baseline_backend = FakeBackend()
    incremental_backend = FakeBackend()

    TraceLabReplayRunner(
        baseline_backend,
        algorithm="baseline",
        time_scale=0,
        run_id="baseline",
    ).run_trial(materialized, trial_index=0)
    TraceLabReplayRunner(
        incremental_backend,
        algorithm="incremental",
        time_scale=0,
        run_id="incremental",
    ).run_trial(materialized, trial_index=0)

    assert baseline_backend.generations[0]["token_ids"] == incremental_backend.generations[0]["token_ids"]
    assert len(baseline_backend.generations[0]["token_ids"]) == trial.prompt_tokens


def test_paired_runners_retry_together_after_missing_cache_readback(tmp_path):
    source = tmp_path / "trace.jsonl.gz"
    write_trace(source)
    workload = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )
    sync = tmp_path / "sync"
    baseline_backend = MissingCachedTokensOnceBackend()
    incremental_backend = FakeBackend()
    baseline = TraceLabReplayRunner(
        baseline_backend,
        algorithm="baseline",
        time_scale=0,
        run_id="baseline",
        measurement_barrier=FilePairBarrier(
            sync,
            participant="gpu0",
            peer="gpu1",
            timeout_s=1,
            poll_s=0.001,
        ),
    )
    incremental = TraceLabReplayRunner(
        incremental_backend,
        algorithm="incremental",
        time_scale=0,
        run_id="incremental",
        measurement_barrier=FilePairBarrier(
            sync,
            participant="gpu1",
            peer="gpu0",
            timeout_s=1,
            poll_s=0.001,
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline_future = pool.submit(baseline.run_workload, workload, warmup_trials=0)
        incremental_future = pool.submit(incremental.run_workload, workload, warmup_trials=0)
        baseline_records, baseline_invalid = baseline_future.result()
        incremental_records, incremental_invalid = incremental_future.result()

    assert baseline_invalid == incremental_invalid == []
    assert len(baseline_records) == len(incremental_records) == 1
    assert len(baseline_backend.generations) == len(incremental_backend.generations) == 2


def test_streaming_usage_does_not_replace_cached_token_readback_with_null():
    cached_tokens = cached_tokens_from_chunk(
        {"usage": {"prompt_tokens_details": {"cached_tokens": 112}}},
        None,
    )

    cached_tokens = cached_tokens_from_chunk(
        {"usage": {"prompt_tokens_details": {"cached_tokens": None}}},
        cached_tokens,
    )

    assert cached_tokens == 112


def test_baseline_infers_zero_cache_hits_when_zero_prefix_readback_is_omitted(tmp_path):
    trial = replace(prepared_trial(tmp_path), prefix_tokens=0, newly_append_tokens=130)
    backend = MissingCachedTokensOnceBackend()

    record = TraceLabReplayRunner(
        backend,
        algorithm="baseline",
        time_scale=0,
        run_id="zero-prefix",
    ).run_trial(materialize_trial(trial), trial_index=0)

    assert record["cached_tokens"] == 0
    assert record["cached_token_readback_inferred_zero"] is True


def test_resume_checkpoint_preserves_records_and_starts_at_next_trial(tmp_path):
    source = tmp_path / "trace.jsonl.gz"
    checkpoint = tmp_path / "replay_results.partial.jsonl"
    write_trace(source)
    prepared = prepare_tracelab_codex_workload(
        source,
        limit=None,
        max_context_tokens=256,
        cache_block_tokens=16,
    )
    first = prepared.trials[0]
    second = replace(first, trial_id=f"{first.trial_id}:second")
    workload = replace(prepared, trials=(first, second))
    initial_backend = FakeBackend()
    initial_record = TraceLabReplayRunner(
        initial_backend,
        algorithm="baseline",
        time_scale=0,
        run_id="initial",
    ).run_trial(materialize_trial(first), trial_index=0)
    checkpoint.write_text(json.dumps(initial_record) + "\n")

    existing = _load_resume_records(checkpoint, workload, "baseline")
    resumed_backend = FakeBackend()
    resumed, invalid = TraceLabReplayRunner(
        resumed_backend,
        algorithm="baseline",
        time_scale=0,
        run_id="resumed",
    ).run_workload(
        workload,
        warmup_trials=0,
        start_index=len(existing),
    )

    assert invalid == []
    assert [record["trial_index"] for record in existing + resumed] == [0, 1]
    assert len(resumed_backend.generations) == 1


def test_cli_exposes_separate_prepare_and_run_commands():
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "prepare" in result.stdout
    assert "run" in result.stdout


def test_cli_prepare_all_selects_every_eligible_trial(tmp_path):
    from typer.testing import CliRunner

    source = tmp_path / "trace.jsonl.gz"
    manifest = tmp_path / "workload.json"
    write_trace(source)

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            str(source),
            "--output",
            str(manifest),
            "--all",
            "--max-context-tokens",
            "256",
        ],
    )

    assert result.exit_code == 0
    workload = load_prepared_workload(manifest)
    assert workload.requested_limit is None
    assert len(workload.trials) == workload.scan_counts["eligible_trials"] == 1
