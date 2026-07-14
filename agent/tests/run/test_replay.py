import json
import os
import threading
import time

from minisweagent.run.replay import (
    ReplayTokenizer,
    TraceReplayRunner,
    collect_trajectory_files,
    iter_visible_checkpoints,
    main,
    mistral_safe_messages,
    prepare_replay_scenario,
    runner_kwargs,
)
from minisweagent.run.replay_backend import completion_chunk_has_generated_payload
from minisweagent.run.replay_messages import tokenizer_safe_messages
from minisweagent.run.replay_metrics import record_for_output


def make_trajectory(raw_output: str = "trace-output") -> dict:
    return {
        "info": {
            "config": {
                "agent": {"tokenizer_path": "", "tokenizer_local_files_only": True},
                "model": {
                    "model_name": "hosted_vllm/fake-model",
                    "model_kwargs": {"api_base": "http://127.0.0.1:9/v1"},
                    "observation_template": "<output>{{ output.output }}</output><returncode>{{ output.returncode }}</returncode>",
                    "stream_observation_template": "<output>{{ output.output }}",
                },
            }
        },
        "instance_id": "case-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            assistant_message(0, "printf SHOULD_NOT_RUN", completion_tokens=5),
            tool_message(raw_output, duration_s=0.02, events=[{"t": 0.005, "output_chars": len(raw_output)}]),
            assistant_message(1, "echo done", completion_tokens=7),
            tool_message("done", duration_s=0.01, events=[{"t": 0.005, "output_chars": 4}]),
            {"role": "exit", "content": "done"},
        ],
    }


def make_candidate_trajectory() -> dict:
    data = make_trajectory()
    data["messages"] = [
        data["messages"][0],
        data["messages"][1],
        assistant_message(0, "pytest tests/a.py -q", completion_tokens=5),
        tool_message("shared-prefix-from-history", duration_s=0.01, events=[]),
        assistant_message(1, "pytest tests/b.py -q", completion_tokens=7),
        tool_message(
            "shared-prefix-from-current",
            duration_s=0.05,
            events=[{"t": 0.04, "output_chars": 26}],
        ),
        assistant_message(2, "echo done", completion_tokens=3),
        tool_message("done", duration_s=0.01, events=[]),
        {"role": "exit", "content": "done"},
    ]
    return data


def assistant_message(index: int, command: str, *, completion_tokens: int) -> dict:
    tool_call_id = f"call_{index}"
    return {
        "role": "assistant",
        "content": f"thought {index}",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        "extra": {
            "actions": [{"command": command, "tool_call_id": tool_call_id}],
            "timestamp": float(index),
            "token_timing": {
                "model_call": {
                    "prompt_tokens": 10 + index,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 10 + index + completion_tokens,
                    "finish_reason": "tool_calls",
                    "ttft_s": 0.1 + index,
                    "model_total_s": 0.2 + index,
                    "decode_s": 0.1,
                }
            },
        },
    }


def tool_message(raw_output: str, *, duration_s: float, events: list[dict]) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": raw_output,
        "extra": {
            "raw_output": raw_output,
            "returncode": 0,
            "exception_info": "",
            "timestamp": duration_s,
            "token_timing": {
                "tool_calls": [
                    {
                        "duration_s": duration_s,
                        "time_to_first_output_s": events[0]["t"] if events else None,
                        "returncode": 0,
                        "raw_output_chars": len(raw_output),
                        "raw_output_bytes": len(raw_output.encode()),
                        "output_events": events,
                    }
                ]
            },
        },
    }


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


class CountingReplayTokenizer(ReplayTokenizer):
    def __init__(self):
        super().__init__(TinyTemplateTokenizer())
        self.encode_calls = 0

    def encode_messages_with_state(self, messages, *, add_generation_prompt):
        self.encode_calls += 1
        return super().encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)


class DelayedSeedTokenizer(ReplayTokenizer):
    def __init__(self, clock, delay):
        super().__init__(TinyTemplateTokenizer())
        self.clock = clock
        self.delay = delay

    def encode_messages_with_state(self, messages, *, add_generation_prompt):
        if messages and messages[-1].get("role") == "assistant" and not add_generation_prompt:
            self.clock.sleep(self.delay)
        return super().encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)


class CommandTemplateTokenizer(TinyTemplateTokenizer):
    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
        lines = []
        for message in messages:
            text = f"{message.get('role')}:{message.get('content') or ''}"
            for tool_call in message.get("tool_calls") or []:
                text += str((tool_call.get("function") or {}).get("arguments") or "")
            lines.append(text)
        text = "\n".join(lines)
        if add_generation_prompt:
            text += "\nassistant:"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class DelayedCommandTokenizer(ReplayTokenizer):
    def __init__(self, clock, delay):
        super().__init__(CommandTemplateTokenizer())
        self.clock = clock
        self.delay = delay

    def encode_messages_with_state(self, messages, *, add_generation_prompt):
        if messages and messages[-1].get("role") == "assistant" and not add_generation_prompt:
            self.clock.sleep(self.delay)
        return super().encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)


class DelayedPartialPromptTokenizer(ReplayTokenizer):
    def __init__(self, clock, delay):
        super().__init__(CommandTemplateTokenizer())
        self.clock = clock
        self.delay = delay

    def encode_messages_with_state(self, messages, *, add_generation_prompt):
        if messages and messages[-1].get("role") == "tool" and add_generation_prompt:
            self.clock.sleep(self.delay)
        return super().encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)


class DivergingGenerationPromptTokenizer(CommandTemplateTokenizer):
    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False):
        text = super().apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if add_generation_prompt:
            text += "\ngeneration-marker-that-is-not-an-assistant-prefix"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def sleep(self, seconds):
        self.value += max(0.0, seconds)
        time.sleep(0.001)

    def now(self):
        return self.value


class FakeBackend:
    def __init__(self):
        self.generations = []
        self.prefills = []
        self.cancelled_prefills = []

    def generate_tokens(self, token_ids, *, max_tokens, cache_salt, step, label):
        text = bytes(token_ids).decode("utf-8")
        self.generations.append({"text": text, "max_tokens": max_tokens, "cache_salt": cache_salt, "label": label})
        return {
            "ttft_s": 0.01,
            "model_total_s": 0.02,
            "decode_s": 0.01,
            "completion_tokens": max_tokens,
            "cached_tokens": len(token_ids),
        }

    def prefill(self, token_ids, *, cache_salt, step, label, request_id=None):
        self.prefills.append(
            {
                "text": bytes(token_ids).decode("utf-8"),
                "tokens": len(token_ids),
                "cache_salt": cache_salt,
                "label": label,
                "request_id": request_id,
                "step_index": step.step_index,
            }
        )

    def cancel_prefill(self, request_id):
        self.cancelled_prefills.append(request_id)


class BlockingPrefillBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def prefill(self, token_ids, *, cache_salt, step, label, request_id=None):
        super().prefill(
            token_ids,
            cache_salt=cache_salt,
            step=step,
            label=label,
            request_id=request_id,
        )
        self.release.wait(timeout=1)
        self.release.clear()

    def cancel_prefill(self, request_id):
        super().cancel_prefill(request_id)
        self.release.set()


def make_runner(backend, data, *, algorithm="baseline", clock=None, **kwargs):
    clock = clock or FakeClock()
    return TraceReplayRunner(
        backend,
        ReplayTokenizer(TinyTemplateTokenizer()),
        data["info"]["config"],
        algorithm=algorithm,
        time_scale=1,
        sleep=clock.sleep,
        now=clock.now,
        **kwargs,
    )


def make_runner_with_tokenizer(backend, data, tokenizer, *, algorithm="baseline", clock=None, **kwargs):
    clock = clock or FakeClock()
    return TraceReplayRunner(
        backend,
        tokenizer,
        data["info"]["config"],
        algorithm=algorithm,
        time_scale=1,
        sleep=clock.sleep,
        now=clock.now,
        **kwargs,
    )


def test_tokenizer_safe_messages_parse_tool_call_arguments():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
                }
            ],
            "extra": {"ignored": True},
        }
    ]

    safe = tokenizer_safe_messages(messages)

    assert "extra" not in safe[0]
    assert safe[0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'


def test_mistral_safe_messages_normalize_tool_call_ids():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1234567890abcdef",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1234567890abcdef", "content": "ok"},
    ]

    safe = mistral_safe_messages(messages)

    assert safe[0]["content"] == ""
    assert safe[0]["tool_calls"][0]["id"] == "tc0000000"
    assert safe[1]["tool_call_id"] == "tc0000000"
    assert messages[0]["tool_calls"][0]["id"] == "call_1234567890abcdef"


def test_prefill_safety_tail_defaults_to_zero():
    kwargs = runner_kwargs({"replay": {"cache_block_tokens": 7}}, algorithm="chunked")

    assert kwargs["cache_block_tokens"] == 7
    assert kwargs["prefill_safety_tail_tokens"] == 0


def test_prefill_safety_tail_can_be_configured():
    kwargs = runner_kwargs(
        {"replay": {"cache_block_tokens": 7, "prefill_safety_tail_tokens": 4}},
        algorithm="chunked",
    )

    assert kwargs["cache_block_tokens"] == 7
    assert kwargs["prefill_safety_tail_tokens"] == 4


def test_prefill_chunk_tokens_can_be_configured():
    kwargs = runner_kwargs(
        {"replay": {"prefill_min_new_tokens": 512, "prefill_chunk_tokens": 128}},
        algorithm="chunked",
    )

    assert kwargs["prefill_chunk_tokens"] == 128


def test_stream_output_char_limit_can_be_configured():
    kwargs = runner_kwargs(
        {"replay": {"stream_output_char_limit": 5000}},
        algorithm="chunked",
    )

    assert kwargs["stream_output_char_limit"] == 5000


def test_candidate_prefill_top_k_can_be_configured():
    kwargs = runner_kwargs(
        {"replay": {"candidate_prefill": {"top_k": 3}}},
        algorithm="candidate",
    )

    assert kwargs["candidate_top_k"] == 3


def test_baseline_replay_uses_trace_output_without_executing_commands(tmp_path):
    data = make_trajectory(raw_output="trace-output")
    backend = FakeBackend()
    runner = make_runner(backend, data, algorithm="baseline")

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert [record["valid"] for record in records] == [True, True]
    assert [generation["max_tokens"] for generation in backend.generations] == [5, 7]
    assert backend.prefills == []
    assert "trace-output" in backend.generations[1]["text"]
    assert "SHOULD_NOT_RUN" not in backend.generations[1]["text"]
    assert records[0]["simulated_tool_duration_s"] == 0.02
    assert abs(records[0]["problem_e2e_s"] - 0.03) < 1e-9
    assert "problem_e2e_s" not in records[1]


def test_prepare_replay_scenario_copies_trace_messages(tmp_path):
    data = make_trajectory(raw_output="trace-output")

    scenario = prepare_replay_scenario(tmp_path / "case.traj.json", data)
    data["messages"][0]["content"] = "mutated"
    data["messages"][2]["extra"]["actions"][0]["command"] = "mutated"

    assert scenario.turns[0].leading_messages[0]["content"] == "system"
    assert scenario.turns[0].assistant["extra"]["actions"][0]["command"] == "printf SHOULD_NOT_RUN"


def test_replay_allows_trace_without_original_ttft(tmp_path):
    data = make_trajectory(raw_output="trace-output")
    for message in data["messages"]:
        model_call = ((message.get("extra") or {}).get("token_timing") or {}).get("model_call") or {}
        model_call["ttft_s"] = None

    scenario = prepare_replay_scenario(tmp_path / "case.traj.json", data)
    records, invalid = make_runner(FakeBackend(), data).run_trajectory(tmp_path / "case.traj.json", data)

    assert scenario.terminal_invalid is None
    assert len(scenario.turns) == 2
    assert invalid == []
    assert [record["valid"] for record in records] == [True, True]
    assert records[0]["trace_ttft_s"] is None
    assert records[0]["replay_ttft_s"] == 0.01


def test_baseline_replay_does_not_tokenize_chunked_prefill_seed(tmp_path):
    data = make_trajectory(raw_output="trace-output")
    tokenizer = CountingReplayTokenizer()
    runner = make_runner_with_tokenizer(FakeBackend(), data, tokenizer, algorithm="baseline")

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert [record["valid"] for record in records] == [True, True]
    assert tokenizer.encode_calls == 2


def test_chunked_replay_prefills_visible_trace_output(tmp_path):
    raw_output = "x" * 200
    data = make_trajectory(raw_output=raw_output)
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="chunked",
        prefill_min_new_tokens=64,
        prefill_check_interval_s=0.001,
        prefill_safety_tail_tokens=32,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] >= 1
    assert records[0]["prefill_started_count"] >= 1
    assert records[0]["prefill_count"] == records[0]["prefill_started_count"]
    assert backend.prefills
    assert "x" * 40 in backend.prefills[0]["text"]
    assert backend.generations[1]["text"].startswith(backend.prefills[0]["text"])
    assert records[0]["prefill_completed_count"] >= 1
    assert records[0]["prefilled_prompt_suffix_tokens"] > 0
    assert 0 < records[0]["prefilled_tool_output_tokens"] < records[0]["prefilled_prompt_suffix_tokens"]


def test_chunked_seed_tokenization_overlaps_tool_time(tmp_path):
    data = make_trajectory(raw_output="trace-output")
    clock = FakeClock()
    tokenizer = DelayedSeedTokenizer(clock, delay=0.01)
    runner = make_runner_with_tokenizer(
        FakeBackend(),
        data,
        tokenizer,
        algorithm="chunked",
        clock=clock,
        prefill_min_new_tokens=10_000,
        prefill_check_interval_s=0.001,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert abs(records[0]["problem_e2e_s"] - 0.03) < 1e-9


def test_chunked_prefill_counts_command_tokens_after_cached_prompt(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 200, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.02, events=[])
    backend = FakeBackend()
    tokenizer = ReplayTokenizer(CommandTemplateTokenizer())
    runner = make_runner_with_tokenizer(
        backend,
        data,
        tokenizer,
        algorithm="chunked",
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.001,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] >= 1
    assert backend.prefills[0]["tokens"] - len(backend.generations[0]["text"].encode()) == 64
    assert records[0]["prefilled_prompt_suffix_tokens"] >= 64
    assert records[0]["prefilled_prompt_suffix_tokens"] % 64 == 0
    assert records[0]["prefilled_tool_output_tokens"] == 0


def test_chunked_does_not_prefill_after_tool_deadline(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 200, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.02, events=[])
    clock = FakeClock()
    tokenizer = DelayedCommandTokenizer(clock, delay=0.03)
    backend = FakeBackend()
    runner = make_runner_with_tokenizer(
        backend,
        data,
        tokenizer,
        algorithm="chunked",
        clock=clock,
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.001,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] == 0
    assert backend.prefills == []


def test_chunked_does_not_submit_when_partial_tokenization_crosses_deadline(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 200, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.02, events=[])
    clock = FakeClock()
    tokenizer = DelayedPartialPromptTokenizer(clock, delay=0.03)
    backend = FakeBackend()
    runner = make_runner_with_tokenizer(
        backend,
        data,
        tokenizer,
        algorithm="chunked",
        clock=clock,
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.001,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] == 0
    assert backend.prefills == []


def test_chunked_drains_command_chunks_while_tool_is_running(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 300, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.2, events=[])
    backend = FakeBackend()
    runner = make_runner_with_tokenizer(
        backend,
        data,
        ReplayTokenizer(CommandTemplateTokenizer()),
        algorithm="chunked",
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.05,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] >= 2
    assert all(
        current["tokens"] - previous["tokens"] == 64
        for previous, current in zip(backend.prefills, backend.prefills[1:])
    )


def test_chunked_keeps_pending_prefill_one_chunk_ahead(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 300, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.2, events=[])
    backend = BlockingPrefillBackend()
    runner = make_runner_with_tokenizer(
        backend,
        data,
        ReplayTokenizer(CommandTemplateTokenizer()),
        algorithm="chunked",
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.01,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_active_at_tool_end"] == 1
    assert records[0]["prefill_pending_at_tool_end"] == 1
    assert (
        records[0]["pending_prefill_prefix_len_at_tool_end"] - records[0]["active_prefill_prefix_len_at_tool_end"] == 64
    )


def test_chunked_cache_frontier_stops_at_prompt_divergence(tmp_path):
    data = make_trajectory(raw_output="")
    data["messages"][2] = assistant_message(0, "x" * 200, completion_tokens=5)
    data["messages"][3] = tool_message("", duration_s=0.02, events=[])
    tokenizer = ReplayTokenizer(DivergingGenerationPromptTokenizer())
    backend = FakeBackend()
    runner = make_runner_with_tokenizer(
        backend,
        data,
        tokenizer,
        algorithm="chunked",
        prefill_chunk_tokens=64,
        prefill_check_interval_s=0.001,
        prefill_safety_tail_tokens=16,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)
    prior_prompt = backend.generations[0]["text"].encode()
    first_prefill = backend.prefills[0]["text"].encode()
    shared_prefix = next(
        index
        for index, (prior_token, prefill_token) in enumerate(zip(prior_prompt, first_prefill))
        if prior_token != prefill_token
    )

    assert invalid == []
    assert backend.prefills[0]["tokens"] - shared_prefix == 64


def test_chunked_replay_advances_prefill_in_fixed_size_chunks(tmp_path):
    data = make_trajectory(raw_output="x" * 240)
    data["messages"][3]["extra"]["timestamp"] = 0.2
    data["messages"][3]["extra"]["token_timing"]["tool_calls"][0]["duration_s"] = 0.2
    data["messages"][3]["extra"]["token_timing"]["tool_calls"][0]["output_events"] = [{"t": 0.005, "output_chars": 240}]
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="chunked",
        prefill_min_new_tokens=64,
        prefill_check_interval_s=0.05,
        prefill_safety_tail_tokens=0,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] >= 2
    assert records[0]["prefill_started_count"] >= 2
    assert records[0]["prefill_count"] == records[0]["prefill_started_count"]
    assert all(
        current["tokens"] - previous["tokens"] == 64
        for previous, current in zip(backend.prefills, backend.prefills[1:])
    )


def test_chunked_replay_reuses_prefill_seed_for_next_prompt(tmp_path):
    data = make_trajectory(raw_output="trace-output")
    tokenizer = CountingReplayTokenizer()
    runner = make_runner_with_tokenizer(
        FakeBackend(),
        data,
        tokenizer,
        algorithm="chunked",
        prefill_min_new_tokens=10_000,
        prefill_check_interval_s=0.001,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert [record["valid"] for record in records] == [True, True]
    assert tokenizer.encode_calls == 5


def test_stream_output_char_limit_avoids_retokenizing_after_limit(tmp_path):
    data = make_trajectory(raw_output="x" * 200)
    data["info"]["config"]["replay"] = {"stream_output_char_limit": 80}
    data["messages"][3]["extra"]["timestamp"] = 0.12
    data["messages"][3]["extra"]["token_timing"]["tool_calls"][0]["duration_s"] = 0.12
    data["messages"][3]["extra"]["token_timing"]["tool_calls"][0]["output_events"] = [
        {"t": 0.005, "output_chars": 120},
        {"t": 0.055, "output_chars": 200},
    ]
    tokenizer = CountingReplayTokenizer()
    runner = make_runner_with_tokenizer(
        FakeBackend(),
        data,
        tokenizer,
        algorithm="chunked",
        prefill_min_new_tokens=64,
        prefill_check_interval_s=0.05,
        stream_output_char_limit=80,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert [record["valid"] for record in records] == [True, True]
    assert tokenizer.encode_calls == 5


def test_chunked_replay_without_next_assistant_does_not_prefill_final_tool(tmp_path):
    data = make_trajectory(raw_output="x" * 200)
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="chunked",
        prefill_min_new_tokens=64,
        prefill_check_interval_s=0.001,
        cache_block_tokens=1,
    )

    records, _ = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert records[-1]["prefill_count"] == 0


def test_candidate_replay_prefills_historical_output_for_a_similar_command(tmp_path):
    data = make_candidate_trajectory()
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=4,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    candidate_prefills = [prefill for prefill in backend.prefills if prefill["label"] == "candidate_tool_output"]
    assert invalid == []
    assert records[0]["candidate_selected_count"] == 0
    assert records[1]["candidate_selected_count"] == 1
    assert records[1]["candidate_completed_count"] == 1
    assert len(candidate_prefills) == 1
    assert "shared-prefix-from-history" in candidate_prefills[0]["text"]
    assert candidate_prefills[0]["cache_salt"] == backend.generations[1]["cache_salt"]
    expected_verified_prefix = len(
        os.path.commonprefix([candidate_prefills[0]["text"], backend.generations[2]["text"]]).encode()
    )
    assert records[1]["candidate_verified_prefix_tokens"] == expected_verified_prefix
    assert records[1]["prefill_completed_prompt_tokens"] == expected_verified_prefix


def test_candidate_planning_overlaps_tool_time(tmp_path):
    data = make_candidate_trajectory()
    clock = FakeClock()
    tokenizer = DelayedSeedTokenizer(clock, delay=0.005)
    runner = make_runner_with_tokenizer(
        FakeBackend(),
        data,
        tokenizer,
        algorithm="candidate",
        clock=clock,
        candidate_top_k=4,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert abs(records[0]["problem_e2e_s"] - 0.07) < 1e-9


def test_candidate_replay_prefills_all_branches_in_shared_subtree_order(tmp_path):
    data = make_trajectory()
    messages = [data["messages"][0], data["messages"][1]]
    history = [
        ("pytest tests/a.py -q", "shared/a-two"),
        ("cat README.md", "zzz"),
        ("pytest tests/c.py -q", "different"),
        ("pytest tests/d.py -q", "shared/a-one"),
    ]
    for index, (command, output) in enumerate(history):
        messages.extend(
            [
                assistant_message(index, command, completion_tokens=3),
                tool_message(output, duration_s=0.01, events=[]),
            ]
        )
    messages.extend(
        [
            assistant_message(4, "pytest tests/e.py -q", completion_tokens=3),
            tool_message("shared/actual", duration_s=0.2, events=[]),
            assistant_message(5, "echo done", completion_tokens=3),
            tool_message("done", duration_s=0.01, events=[]),
            {"role": "exit", "content": "done"},
        ]
    )
    data["messages"] = messages
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=4,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    target_prefills = [
        prefill
        for prefill in backend.prefills
        if prefill["label"] == "candidate_tool_output" and prefill["step_index"] == 4
    ]
    assert invalid == []
    assert records[4]["candidate_selected_count"] == 4
    assert records[4]["candidate_completed_count"] == 4
    assert len(target_prefills) == 4
    candidate_suffixes = [item["text"].rsplit("tool:", 1)[-1] for item in target_prefills]
    assert [
        next(output for output in ("shared/a-one", "shared/a-two", "different", "zzz") if output in suffix)
        for suffix in candidate_suffixes
    ] == [
        "shared/a-one",
        "shared/a-two",
        "different",
        "zzz",
    ]


def test_candidate_replay_falls_back_to_actual_chunks_when_every_candidate_misses(tmp_path):
    data = make_candidate_trajectory()
    current_tool = data["messages"][5]
    current_tool["extra"]["raw_output"] = "totally-new-output"
    current_tool["extra"]["timestamp"] = 0.2
    timing = current_tool["extra"]["token_timing"]["tool_calls"][0]
    timing["duration_s"] = 0.2
    timing["output_events"] = [{"t": 0.02, "output_chars": len("totally-new-output")}]
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=4,
        prefill_chunk_tokens=4,
        prefill_check_interval_s=0.01,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    target_prefills = [prefill for prefill in backend.prefills if prefill["step_index"] == 1]
    assert invalid == []
    assert records[1]["candidate_selected_count"] == 1
    assert records[1]["candidate_fallback_to_chunked"] == 1
    assert records[1]["candidate_surviving_count"] == 0
    assert [prefill["label"] for prefill in target_prefills][0] == "candidate_tool_output"
    assert "tool_output" in [prefill["label"] for prefill in target_prefills]


def test_candidate_fallback_continues_after_a_completed_matching_prefix(tmp_path):
    data = make_candidate_trajectory()
    data["messages"][3] = tool_message("long-shared-prefix-HISTORY", duration_s=0.01, events=[])
    data["messages"][5] = tool_message(
        "long-shared-prefix-CURRENT",
        duration_s=0.2,
        events=[{"t": 0.02, "output_chars": len("long-shared-prefix-CURRENT")}],
    )
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
        prefill_chunk_tokens=4,
        prefill_check_interval_s=0.01,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    target_prefills = [prefill for prefill in backend.prefills if prefill["step_index"] == 1]
    candidate = next(prefill for prefill in target_prefills if prefill["label"] == "candidate_tool_output")
    actual = next(prefill for prefill in target_prefills if prefill["label"] == "tool_output")
    final_actual_prompt = backend.generations[2]["text"]
    matching_prefix = len(os.path.commonprefix([candidate["text"], final_actual_prompt]).encode())
    assert invalid == []
    assert records[1]["candidate_fallback_to_chunked"] == 1
    assert actual["tokens"] == matching_prefix + 4


def test_candidate_prefill_applies_the_stream_output_character_limit(tmp_path):
    data = make_candidate_trajectory()
    data["messages"][3] = tool_message("abcdefghijk", duration_s=0.01, events=[])
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
        stream_output_char_limit=5,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    candidate = next(
        prefill
        for prefill in backend.prefills
        if prefill["step_index"] == 1 and prefill["label"] == "candidate_tool_output"
    )
    candidate_observation = candidate["text"].rsplit("tool:", 1)[-1]
    assert invalid == []
    assert records[1]["candidate_selected_count"] == 1
    assert "<output>abcde" in candidate_observation
    assert "fghijk" not in candidate_observation


def test_candidate_prefill_skips_a_branch_that_exceeds_the_context_limit(tmp_path):
    data = make_candidate_trajectory()
    data["messages"][3] = tool_message("x" * 200, duration_s=0.01, events=[])
    data["messages"][5] = tool_message("different", duration_s=0.05, events=[])

    probe_backend = FakeBackend()
    probe = make_runner(
        probe_backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
        cache_block_tokens=1,
    )
    probe.run_trajectory(tmp_path / "probe.traj.json", data)
    candidate_tokens = next(
        prefill["tokens"]
        for prefill in probe_backend.prefills
        if prefill["step_index"] == 1 and prefill["label"] == "candidate_tool_output"
    )
    max_actual_prompt_tokens = max(len(generation["text"].encode()) for generation in probe_backend.generations)
    assert candidate_tokens > max_actual_prompt_tokens

    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
        max_context_tokens=max_actual_prompt_tokens,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    target_prefills = [prefill for prefill in backend.prefills if prefill["step_index"] == 1]
    assert invalid == []
    assert records[1]["candidate_selected_count"] == 0
    assert records[1]["candidate_skipped_capacity_count"] == 1
    assert target_prefills == []


def test_candidate_replay_aborts_a_wrong_branch_and_keeps_a_matching_branch(tmp_path):
    data = make_trajectory()
    data["messages"] = [
        data["messages"][0],
        data["messages"][1],
        assistant_message(0, "pytest tests/a.py -q", completion_tokens=3),
        tool_message("matching-prefix-from-history", duration_s=0.01, events=[]),
        assistant_message(1, "pytest tests/b.py -q", completion_tokens=3),
        tool_message("wrong-prefix-from-history", duration_s=0.01, events=[]),
        assistant_message(2, "pytest tests/c.py -q", completion_tokens=3),
        tool_message(
            "matching-prefix-from-current",
            duration_s=0.2,
            events=[{"t": 0.02, "output_chars": len("matching-prefix-from-")}],
        ),
        assistant_message(3, "echo done", completion_tokens=3),
        tool_message("done", duration_s=0.01, events=[]),
        {"role": "exit", "content": "done"},
    ]
    backend = BlockingPrefillBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=2,
        prefill_check_interval_s=0.01,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    target_prefills = [prefill for prefill in backend.prefills if prefill["step_index"] == 2]
    assert invalid == []
    assert records[2]["candidate_selected_count"] == 2
    assert records[2]["candidate_pruned_count"] == 1
    assert records[2]["candidate_surviving_count"] == 1
    assert records[2]["candidate_fallback_to_chunked"] == 0
    assert records[2]["candidate_cancelled_count"] == 1
    assert [prefill["label"] for prefill in target_prefills] == [
        "candidate_tool_output",
        "candidate_tool_output",
    ]
    assert backend.cancelled_prefills


def test_scheduled_checks_are_on_interval_after_output_events():
    events = [{"t": 0.011, "output_chars": 10}, {"t": 0.037, "output_chars": 20}]

    assert list(iter_visible_checkpoints(events, duration_s=0.08, output_chars=20, interval_s=0.025)) == [
        (0.025, 10),
        (0.05, 20),
        (0.075, 20),
    ]


def test_scheduled_checks_continue_without_output_events():
    assert list(iter_visible_checkpoints([], duration_s=0.16, output_chars=0, interval_s=0.05)) == [
        (0.05, 0),
        (0.1, 0),
        (0.15, 0),
    ]


def test_event_time_checkpoints_include_visible_chars():
    events = [{"t": 0.011, "output_chars": 10}, {"t": 0.037, "output_chars": 20}]

    assert list(iter_visible_checkpoints(events, duration_s=0.08, output_chars=20, interval_s=0)) == [
        (0.011, 10),
        (0.037, 20),
    ]


def test_record_output_keeps_only_core_metrics():
    record = {
        "trajectory_path": "case.traj.json",
        "instance_id": "case",
        "algorithm": "baseline",
        "step_index": 0,
        "valid": True,
        "skip_reason": "",
        "trace_ttft_s": 0.2,
        "replay_ttft_s": 0.3,
        "candidate_selected_count": 4,
        "internal_debug_counter": 100,
    }

    compact = record_for_output(record)

    assert "trace_ttft_s" in compact
    assert compact["candidate_selected_count"] == 4
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


def test_collect_trajectory_files_accepts_file_and_directory(tmp_path):
    first = tmp_path / "a.traj.json"
    second = tmp_path / "nested" / "b.traj.json"
    second.parent.mkdir()
    first.write_text("{}")
    second.write_text("{}")

    assert collect_trajectory_files(first) == [first]
    assert collect_trajectory_files(tmp_path) == [first, second]


def test_replay_cli_writes_outputs(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "case.traj.json"
    data = make_trajectory(raw_output="cli-output")
    data["info"]["config"]["agent"]["tokenizer_path"] = "fake-tokenizer"
    trajectory_path.write_text(json.dumps(data))
    output_dir = tmp_path / "out"
    backend = FakeBackend()

    monkeypatch.setattr("minisweagent.run.replay.load_tokenizer", lambda *args, **kwargs: TinyTemplateTokenizer())
    monkeypatch.setattr("minisweagent.run.replay.backend_from_config", lambda config: backend)

    main(path=trajectory_path, output=output_dir, algorithm="baseline", limit=None, config_spec=[])

    results = (output_dir / "replay_results.jsonl").read_text().splitlines()
    invalid = (output_dir / "invalid_steps.jsonl").read_text().splitlines()
    summary = json.loads((output_dir / "summary.json").read_text())

    assert len(results) == 2
    assert invalid == []
    assert summary["valid"] == 2
