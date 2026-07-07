import json
import time

import pytest

from minisweagent.run.replay import (
    LiveReplayRunner,
    ReplayTokenizer,
    collect_trajectory_files,
    main,
)
from minisweagent.run.replay_backend import completion_chunk_has_generated_payload
from minisweagent.run.replay_messages import tokenizer_safe_messages
from minisweagent.run.replay_metrics import (
    live_stream_burst_stats,
    record_for_output,
)
from minisweagent.run.replay_types import (
    LiveCommandResult,
    LiveOutputEvent,
)


def make_trajectory() -> dict:
    return {
        "info": {
            "config": {
                "agent": {"tokenizer_path": "", "tokenizer_local_files_only": True},
                "model": {"model_name": "hosted_vllm/fake-model", "model_kwargs": {"api_base": "http://127.0.0.1:9/v1"}},
            }
        },
        "instance_id": "case-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\": \"printf one\"}"},
                    }
                ],
                "extra": {"actions": [{"command": "printf one", "tool_call_id": "call_1"}]},
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "<returncode>0</returncode>\n<output>one two</output>",
                "extra": {"raw_output": "one two", "returncode": 0},
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\": \"pwd\"}"},
                    }
                ],
                "extra": {
                    "actions": [{"command": "pwd", "tool_call_id": "call_2"}],
                    "token_timing": {"model_call": {"ttft_s": 0.25}},
                },
            },
        ],
    }


def output_first_config(tmp_path, command: str) -> dict:
    data = make_trajectory()
    config = data["info"]["config"]
    config["environment_type"] = "minisweagent.environments.local.LocalEnvironment"
    config["environment"] = {"cwd": str(tmp_path), "timeout": 5}
    config["model"]["observation_template"] = """
<output>
{{ output.output -}}
</output>
<returncode>{{output.returncode}}</returncode>
"""
    data["messages"][2]["extra"]["actions"] = [{"command": command, "tool_call_id": "call_1"}]
    return data


def output_first_trajectory(tmp_path) -> dict:
    return output_first_config(tmp_path, "printf one")


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def sleep(self, seconds):
        self.value += max(0.0, seconds)

    def now(self):
        return self.value


class TinyTemplateTokenizer:
    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"{message.get('role')}:{message.get('content')}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, *, add_special_tokens=False):
        return list((text or "").encode("utf-8"))

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        if return_offsets_mapping:
            return {"offset_mapping": [(index, index + 1) for index, _ in enumerate(text or "")]}
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


class FakeBackend:
    def __init__(self, clock: FakeClock, *, prefill_delay: float = 0.0):
        self.clock = clock
        self.prefill_delay = prefill_delay
        self.prefills = []
        self.prefill_texts = []
        self.measured_token_texts = []
        self.cancelled_prefills = []

    def start_trial(self, trial_name, step):
        return trial_name

    def prefill(self, token_ids, *, cache_salt, step, label, request_id=None):
        self.prefills.append((cache_salt, label, len(token_ids)))
        try:
            self.prefill_texts.append((label, bytes(token_ids).decode("utf-8")))
        except ValueError:
            self.prefill_texts.append((label, ""))
        self.clock.sleep(self.prefill_delay)

    def cancel_prefill(self, request_id):
        self.cancelled_prefills.append(request_id)

    def measure_ttft_tokens(self, token_ids, *, cache_salt, step):
        self.measured_token_texts.append(bytes(token_ids).decode("utf-8"))
        start = self.clock.now()
        ttft = 0.1
        return {
            "ttft_s": ttft,
            "request_start_at": start,
            "first_token_at": start + ttft,
            "cached_tokens": len(token_ids),
            "prompt_tokens": len(token_ids),
        }


class SlowBackend(FakeBackend):
    def __init__(self, clock: FakeClock, *, delay: float):
        super().__init__(clock)
        self.delay = delay

    def prefill(self, token_ids, *, cache_salt, step, label, request_id=None):
        time.sleep(self.delay)
        super().prefill(token_ids, cache_salt=cache_salt, step=step, label=label, request_id=request_id)


class SlowTokenizer(ReplayTokenizer):
    def __init__(self, *, delay: float):
        super().__init__(TinyTemplateTokenizer())
        self.delay = delay

    def encode_messages(self, messages, *, add_generation_prompt):
        if messages and messages[-1].get("role") == "tool":
            time.sleep(self.delay)
        return super().encode_messages(messages, add_generation_prompt=add_generation_prompt)

    def count_text_tokens(self, text):
        time.sleep(self.delay)
        return super().count_text_tokens(text)


def test_tokenizer_safe_messages_parse_tool_call_arguments():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\": \"pwd\"}"},
                }
            ],
            "extra": {"ignored": True},
        }
    ]

    safe = tokenizer_safe_messages(messages)

    assert "extra" not in safe[0]
    assert safe[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{\"command\": \"pwd\"}"


def test_incremental_tokenization_matches_full_suffix_tokenization():
    tokenizer = ReplayTokenizer(TinyTemplateTokenizer())
    first = [{"role": "tool", "content": "<output>hello</output>"}]
    second = [{"role": "tool", "content": "<output>hello world</output>"}]
    first_state = tokenizer.encode_messages_with_state(first, add_generation_prompt=False)
    full_state = tokenizer.encode_messages_with_state(second, add_generation_prompt=False)

    incremental_state = tokenizer.encode_messages_incremental(
        second,
        add_generation_prompt=False,
        previous_state=first_state,
        overlap_chars=4,
    )

    assert incremental_state.token_ids == full_state.token_ids


def test_live_replay_executes_command_and_uses_live_output(tmp_path):
    backend = FakeBackend(FakeClock())
    data = output_first_trajectory(tmp_path)
    data["messages"][2]["extra"]["actions"] = [{"command": "printf live-output", "tool_call_id": "call_1"}]

    runner = LiveReplayRunner(
        backend,
        ReplayTokenizer(TinyTemplateTokenizer()),
        data["info"]["config"],
        prefill_min_new_tokens=999,
        prefill_min_interval_s=0,
        prefill_safety_tail_tokens=0,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == [{"trajectory_path": str(tmp_path / "case.traj.json"), "instance_id": "case-1", "step_index": 1, "valid": False, "skip_reason": "no_next_prompt"}]
    assert len(records) == 1
    record = records[0]
    assert record["valid"] is True
    assert record["live_command_count"] == 1
    assert record["live_stream_output_tokens"] == len("live-output")
    assert record["tool_output_tokens"] > 0
    assert record["pre_end_prefill_count"] == 0
    assert record["unprefilled_tool_output_tokens"] == record["tool_output_tokens"]
    assert "live-output" in backend.measured_token_texts[0]
    assert "one two" not in backend.measured_token_texts[0]


def test_live_stream_burst_stats_show_when_output_arrives():
    result = LiveCommandResult(
        output="abcd",
        returncode=0,
        exception_info="",
        events=[
            LiveOutputEvent(t=0.10, text="a"),
            LiveOutputEvent(t=0.60, text="bc"),
            LiveOutputEvent(t=0.90, text="d"),
        ],
        duration_s=1.0,
    )

    stats = live_stream_burst_stats([result], len)

    assert stats["live_stream_output_tokens"] == 4
    assert stats["live_stream_first_output_fraction"] == pytest.approx(0.10)
    assert stats["live_stream_last_output_fraction"] == pytest.approx(0.90)
    assert stats["live_stream_tokens_before_75pct"] == 3


def test_record_output_keeps_only_core_metrics():
    record = {
        "trajectory_path": "case.traj.json",
        "instance_id": "case",
        "step_index": 0,
        "valid": True,
        "skip_reason": "",
        "baseline_ttft_s": 0.2,
        "stream_prefill_ttft_s": 0.3,
        "internal_debug_counter": 100,
    }

    compact = record_for_output(record)

    assert "baseline_ttft_s" in compact
    assert "internal_debug_counter" not in compact


def test_empty_text_stop_chunk_counts_as_generated_payload():
    chunk = {
        "choices": [
            {
                "text": "",
                "finish_reason": "stop",
                "stop_reason": 248044,
            }
        ]
    }

    assert completion_chunk_has_generated_payload(chunk) is True


def test_usage_only_chunk_is_not_generated_payload():
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
        },
    }

    assert completion_chunk_has_generated_payload(chunk) is False


def test_live_replay_prefills_running_output_when_observation_is_output_first(tmp_path):
    backend = FakeBackend(FakeClock())
    data = output_first_config(tmp_path, "printf '%0400d' 0; sleep 0.2")

    runner = LiveReplayRunner(
        backend,
        ReplayTokenizer(TinyTemplateTokenizer()),
        data["info"]["config"],
        prefill_min_new_tokens=1,
        prefill_min_interval_s=0,
        prefill_safety_tail_tokens=32,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == [{"trajectory_path": str(tmp_path / "case.traj.json"), "instance_id": "case-1", "step_index": 1, "valid": False, "skip_reason": "no_next_prompt"}]
    assert len(records) == 1
    record = records[0]
    assert record["valid"] is True
    assert record["pre_end_prefill_count"] >= 1
    assert record["command_time_prefill_tool_output_tokens"] > 0
    assert record["unprefilled_tool_output_tokens"] < record["tool_output_tokens"]
    tool_output_prefills = [text for label, text in backend.prefill_texts if label == "tool_output"]
    assert tool_output_prefills
    assert all("<output>\n" in text for text in tool_output_prefills)
    assert all("</output>" not in text for text in tool_output_prefills)
    assert all("<returncode>" not in text for text in tool_output_prefills)


def test_live_stream_prefill_does_not_wait_after_final_command(tmp_path):
    backend = SlowBackend(FakeClock(), delay=0.2)
    data = output_first_config(tmp_path, "printf '%0400d' 0; sleep 0.05")

    runner = LiveReplayRunner(
        backend,
        ReplayTokenizer(TinyTemplateTokenizer()),
        data["info"]["config"],
        prefill_min_new_tokens=1,
        prefill_min_interval_s=0,
        prefill_safety_tail_tokens=32,
        cache_block_tokens=1,
    )

    records, _ = runner.run_trajectory(tmp_path / "case.traj.json", data)

    record = records[0]
    assert record["valid"] is True
    assert record["command_time_prefill_tool_output_tokens"] == 0
    assert record["unprefilled_tool_output_tokens"] > 0
    assert record["live_command_duration_s"] < 0.2
    assert record["prefill_active_at_tool_end"] == 1
    assert record["active_prefill_cancel_requested_at_tool_end"] == 1
    assert backend.cancelled_prefills


def test_live_stream_tokenization_does_not_block_command_reader(tmp_path):
    backend = FakeBackend(FakeClock())
    data = output_first_config(tmp_path, "printf '%0400d' 0")

    runner = LiveReplayRunner(
        backend,
        SlowTokenizer(delay=0.2),
        data["info"]["config"],
        prefill_min_new_tokens=1,
        prefill_min_interval_s=0,
        prefill_safety_tail_tokens=32,
        cache_block_tokens=1,
    )

    records, _ = runner.run_trajectory(tmp_path / "case.traj.json", data)

    record = records[0]
    assert record["valid"] is True
    assert record["live_command_duration_s"] < 0.2
    assert record["live_stream_output_tokens"] == 400


def test_live_replay_leaves_final_command_tail_for_ttft_request(tmp_path):
    backend = FakeBackend(FakeClock())
    data = output_first_config(tmp_path, "printf '%01500d' 0")

    runner = LiveReplayRunner(
        backend,
        ReplayTokenizer(TinyTemplateTokenizer()),
        data["info"]["config"],
        prefill_min_new_tokens=1000,
        prefill_min_interval_s=0,
        prefill_safety_tail_tokens=32,
        cache_block_tokens=1,
    )

    records, _ = runner.run_trajectory(tmp_path / "case.traj.json", data)

    record = records[0]
    assert record["valid"] is True
    assert record["unprefilled_tool_output_tokens"] > 0


def test_capacity_skip_uses_live_seed_prompt(tmp_path):
    backend = FakeBackend(FakeClock())
    data = output_first_trajectory(tmp_path)
    runner = LiveReplayRunner(backend, ReplayTokenizer(TinyTemplateTokenizer()), data["info"]["config"], max_context_tokens=3)

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["valid"] is False
    assert records[0]["skip_reason"] == "skipped_capacity"


def test_collect_trajectory_files_accepts_file_and_directory(tmp_path):
    first = tmp_path / "a.traj.json"
    second = tmp_path / "nested" / "b.traj.json"
    second.parent.mkdir()
    first.write_text("{}")
    second.write_text("{}")

    assert collect_trajectory_files(first) == [first]
    assert collect_trajectory_files(tmp_path) == [first, second]


def test_replay_cli_writes_outputs_without_prefill_hook(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "case.traj.json"
    data = output_first_trajectory(tmp_path)
    data["info"]["config"]["agent"]["tokenizer_path"] = "fake-tokenizer"
    trajectory_path.write_text(json.dumps(data))
    output_dir = tmp_path / "out"
    monkeypatch.setattr("minisweagent.run.replay.load_tokenizer", lambda *args, **kwargs: TinyTemplateTokenizer())

    main(path=trajectory_path, output=output_dir, config_spec=[])

    results = (output_dir / "replay_results.jsonl").read_text().splitlines()
    invalid = (output_dir / "invalid_steps.jsonl").read_text().splitlines()
    summary = json.loads((output_dir / "summary.json").read_text())

    assert results
    assert invalid
    assert summary["valid"] == 0
    assert summary["skip_reasons"]["missing_prefill_url"] == 2


def test_replay_cli_flushes_and_continues_after_trajectory_failure(tmp_path, monkeypatch):
    first = tmp_path / "a.traj.json"
    second = tmp_path / "b.traj.json"
    data = output_first_trajectory(tmp_path)
    data["info"]["config"]["agent"]["tokenizer_path"] = "fake-tokenizer"
    first.write_text(json.dumps(data))
    second.write_text(json.dumps(data))
    output_dir = tmp_path / "out"

    def fake_run_trajectory(self, path, data):
        if path == first:
            raise RuntimeError("container start failed")
        return (
            [
                {
                    "trajectory_path": str(path),
                    "instance_id": "case-1",
                    "step_index": 0,
                    "valid": True,
                    "skip_reason": "",
                    "baseline_ttft_s": 0.2,
                    "stream_prefill_ttft_s": 0.1,
                    "delta_ttft_s": 0.1,
                }
            ],
            [],
        )

    monkeypatch.setattr("minisweagent.run.replay.load_tokenizer", lambda *args, **kwargs: TinyTemplateTokenizer())
    monkeypatch.setattr(LiveReplayRunner, "run_trajectory", fake_run_trajectory)

    main(path=tmp_path, output=output_dir, config_spec=[])

    results = (output_dir / "replay_results.jsonl").read_text().splitlines()
    invalid = (output_dir / "invalid_steps.jsonl").read_text().splitlines()
    summary = json.loads((output_dir / "summary.json").read_text())

    assert len(results) == 1
    assert len(invalid) == 1
    assert summary["total"] == 2
    assert summary["valid"] == 1
    assert summary["skipped"] == 1
    assert summary["skip_reasons"]["trajectory_failed:RuntimeError"] == 1
