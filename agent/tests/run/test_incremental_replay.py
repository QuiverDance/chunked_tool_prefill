import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from minisweagent.run.extra.incremental_replay import (
    IncrementalReplayRunner,
    completed_tool_calls,
    main,
)
from minisweagent.run.replay import ReplayTokenizer, load_trajectory, prepare_replay_scenario
from minisweagent.run.replay_sources import collect_replay_sources, load_replay_source
from tests.run.test_replay import (
    BlockingPrefillBackend,
    CommandTemplateTokenizer,
    FakeBackend,
    FakeClock,
    assistant_message,
    make_trajectory,
    tool_message,
)


@dataclass(frozen=True)
class ToolFixture:
    command: str
    output: str
    duration_s: float
    call_id: str
    completion_offset_s: float | None = None


class DelayedIncrementalTokenizer(ReplayTokenizer):
    def __init__(self, clock: FakeClock, delay_s: float):
        super().__init__(CommandTemplateTokenizer())
        self.clock = clock
        self.delay_s = delay_s

    def encode_messages_with_state(self, messages, *, add_generation_prompt):
        if messages and messages[-1].get("role") == "tool" and not add_generation_prompt:
            self.clock.sleep(self.delay_s)
        return super().encode_messages_with_state(messages, add_generation_prompt=add_generation_prompt)


def multi_tool_trajectory(*, completion_offsets: list[float] | None) -> dict:
    offsets = completion_offsets or [None, None, None]
    return tool_group_trajectory(
        [
            ToolFixture("slow", "slow output", 0.3, "call_slow", offsets[0]),
            ToolFixture("middle", "middle output", 0.2, "call_middle", offsets[1]),
            ToolFixture("fast", "fast output", 0.1, "call_fast", offsets[2]),
        ]
    )


def tool_group_trajectory(tools: list[ToolFixture]) -> dict:
    data = make_trajectory()
    assistant = assistant_message(0, tools[0].command, completion_tokens=5)
    actions = [{"command": tool.command, "tool_call_id": tool.call_id} for tool in tools]
    assistant["extra"]["actions"] = actions
    assistant["tool_calls"] = [
        {
            "id": action["tool_call_id"],
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": action["command"]}),
            },
        }
        for action in actions
    ]

    tool_messages = []
    for action, tool in zip(actions, tools, strict=True):
        message = tool_message(
            tool.output,
            duration_s=tool.duration_s,
            events=[{"t": tool.duration_s, "output_chars": len(tool.output)}],
        )
        message["tool_call_id"] = action["tool_call_id"]
        if tool.completion_offset_s is not None:
            metric = message["extra"]["token_timing"]["tool_calls"][0]
            metric["completion_offset_s"] = tool.completion_offset_s
        tool_messages.append(message)

    data["messages"] = [
        *data["messages"][:2],
        assistant,
        *tool_messages,
        *data["messages"][4:],
    ]
    return data


def open_code_trace() -> dict:
    return {
        "info": {"id": "ses_test"},
        "messages": [
            {
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "Inspect the repository"}],
            },
            {
                "info": {
                    "role": "assistant",
                    "tokens": {"input": 20, "output": 8, "total": 28},
                    "time": {"created": 1_000, "completed": 1_050},
                    "finish": "tool-calls",
                },
                "parts": [
                    {"type": "text", "text": "I will inspect both."},
                    {
                        "type": "tool",
                        "tool": "read",
                        "callID": "call_read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "README.md"},
                            "output": "read output",
                            "time": {"start": 1_100, "end": 1_400},
                        },
                    },
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call_bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "git status --short"},
                            "output": "bash output",
                            "metadata": {"exit": 0},
                            "time": {"start": 1_100, "end": 1_200},
                        },
                    },
                ],
            },
            {
                "info": {
                    "role": "assistant",
                    "tokens": {"input": 30, "output": 4, "total": 34},
                    "time": {"created": 1_500, "completed": 1_550},
                    "finish": "stop",
                },
                "parts": [{"type": "text", "text": "Inspection complete."}],
            },
        ],
    }


def codex_trace() -> list[dict]:
    return [
        {
            "timestamp": "2026-04-01T12:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "codex_test"},
        },
        {
            "timestamp": "2026-04-01T12:00:00.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Follow repository instructions."}],
            },
        },
        {
            "timestamp": "2026-04-01T12:00:00.001Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect both files."}],
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect both."}],
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.100Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "slow command"}),
                "call_id": "call_slow",
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.200Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "fast command"}),
                "call_id": "call_fast",
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.300Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "output_tokens": 12,
                        "reasoning_output_tokens": 3,
                        "total_tokens": 112,
                    }
                },
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.500Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_fast",
                "output": "fast output",
            },
        },
        {
            "timestamp": "2026-04-01T12:00:01.900Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_slow",
                "output": "slow output",
            },
        },
        {
            "timestamp": "2026-04-01T12:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Inspection complete."}],
            },
        },
        {
            "timestamp": "2026-04-01T12:00:02.100Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 120,
                        "cached_input_tokens": 100,
                        "output_tokens": 4,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 124,
                    }
                },
            },
        },
    ]


def write_codex_trace(path: Path, records: list[dict] | None = None) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records or codex_trace()))


def make_runner(
    backend: FakeBackend,
    data: dict,
    clock: FakeClock,
    *,
    algorithm: str = "incremental",
) -> IncrementalReplayRunner:
    return IncrementalReplayRunner(
        backend,
        ReplayTokenizer(CommandTemplateTokenizer()),
        data.get("info", {}).get("config", {}),
        algorithm=algorithm,
        cache_block_tokens=1,
        time_scale=1,
        sleep=clock.sleep,
        now=clock.now,
    )


def test_generic_trajectory_parser_reads_tool_completion_offsets(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    data["messages"][3:6] = reversed(data["messages"][3:6])

    scenario = prepare_replay_scenario(tmp_path / "case.traj.json", data)

    first_turn = scenario.turns[0]
    assert [tool.raw_output for tool in first_turn.trace_tools] == [
        "slow output",
        "middle output",
        "fast output",
    ]
    assert [tool.completion_offset_s for tool in first_turn.trace_tools] == [0.3, 0.2, 0.1]


def test_incremental_replay_appends_parallel_outputs_in_completion_order(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    backend = FakeBackend()
    runner = make_runner(backend, data, FakeClock())

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    prefills = [prefill["text"] for prefill in backend.prefills]
    assert invalid == []
    assert records[0]["algorithm"] == "incremental"
    assert records[0]["prefill_submitted_count"] == 2
    assert records[0]["prefill_completed_count"] == 2
    assert "fast output" in prefills[0]
    assert "middle output" not in prefills[0]
    assert prefills[1].index("fast output") < prefills[1].index("middle output")
    assert "slow output" not in prefills[1]
    assert backend.generations[1]["text"].startswith(prefills[1])
    assert backend.generations[1]["text"].index("middle output") < backend.generations[1]["text"].index("slow output")
    assert all(command in prefills[0] for command in ("slow", "middle", "fast"))
    assert prefills[0].index('"command": "slow"') < prefills[0].index('"command": "middle"')
    assert prefills[0].index('"command": "middle"') < prefills[0].index('"command": "fast"')


def test_incremental_prefill_sends_the_entire_partial_prompt(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    backend = FakeBackend()
    clock = FakeClock()
    tokenizer = ReplayTokenizer(CommandTemplateTokenizer())
    runner = IncrementalReplayRunner(
        backend,
        tokenizer,
        data["info"]["config"],
        algorithm="incremental",
        cache_block_tokens=64,
        time_scale=1,
        sleep=clock.sleep,
        now=clock.now,
    )
    scenario = prepare_replay_scenario(tmp_path / "case.traj.json", data)
    turn = scenario.turns[0]
    first_completed = completed_tool_calls(turn.actions, turn.trace_tools)[:1]
    partial_messages = turn.leading_messages + [turn.assistant] + runner.observation_messages_for(first_completed)
    expected_tokens = runner.encode_messages_with_state(
        partial_messages,
        add_generation_prompt=False,
    ).token_ids

    runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert backend.prefills[0]["tokens"] == len(expected_tokens)


def test_incremental_replay_uses_call_order_for_sequential_tools(tmp_path):
    data = multi_tool_trajectory(completion_offsets=None)
    backend = FakeBackend()
    runner = make_runner(backend, data, FakeClock())

    runner.run_trajectory(tmp_path / "case.traj.json", data)

    prefills = [prefill["text"] for prefill in backend.prefills]
    assert len(prefills) == 2
    assert "slow output" in prefills[0]
    assert "middle output" not in prefills[0]
    assert prefills[1].index("slow output") < prefills[1].index("middle output")
    assert "fast output" not in prefills[1]


def test_incremental_replay_cancels_prefill_when_all_tools_finish(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.1, 0.2, 0.3])
    backend = BlockingPrefillBackend()
    runner = make_runner(backend, data, FakeClock())

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    first_turn = records[0]
    next_prompt = backend.generations[1]["text"]
    assert invalid == []
    assert first_turn["prefill_submitted_count"] == 2
    assert first_turn["prefill_active_at_tool_end"] == 1
    assert first_turn["prefill_pending_at_tool_end"] == 1
    assert first_turn["active_prefill_cancel_requested_at_tool_end"] == 1
    assert backend.cancelled_prefills
    assert all(output in next_prompt for output in ("slow output", "middle output", "fast output"))


def test_incremental_replay_keeps_only_the_latest_pending_prefix(tmp_path):
    data = tool_group_trajectory(
        [
            ToolFixture("a", "output a", 0.1, "call_a", 0.1),
            ToolFixture("b", "output b", 0.2, "call_b", 0.2),
            ToolFixture("c", "output c", 0.3, "call_c", 0.3),
            ToolFixture("d", "output d", 0.4, "call_d", 0.4),
        ]
    )
    backend = BlockingPrefillBackend()
    clock = FakeClock()
    runner = make_runner(backend, data, clock)
    turn = prepare_replay_scenario(tmp_path / "case.traj.json", data).turns[0]
    completed = completed_tool_calls(turn.actions, turn.trace_tools)
    expected_pending_tokens = runner.encode_messages_with_state(
        turn.leading_messages + [turn.assistant] + runner.observation_messages_for(completed[:3]),
        add_generation_prompt=False,
    ).token_ids

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    first_turn = records[0]
    assert invalid == []
    assert first_turn["prefill_submitted_count"] == 3
    assert first_turn["prefill_started_count"] == 1
    assert first_turn["prefill_coalesced_count"] == 1
    assert first_turn["pending_prefill_prefix_len_at_tool_end"] == len(expected_pending_tokens)
    assert len(backend.prefills) == 1
    assert "output a" in backend.prefills[0]["text"]
    assert "output b" not in backend.prefills[0]["text"]


def test_simultaneous_tool_results_submit_one_combined_prefill(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.1, 0.1, 0.3])
    backend = FakeBackend()
    runner = make_runner(backend, data, FakeClock())

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] == 1
    assert len(backend.prefills) == 1
    assert "slow output" in backend.prefills[0]["text"]
    assert "middle output" in backend.prefills[0]["text"]
    assert "fast output" not in backend.prefills[0]["text"]


def test_incremental_replay_does_not_submit_after_the_tool_deadline(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.1, 0.2, 0.3])
    backend = FakeBackend()
    clock = FakeClock()
    runner = IncrementalReplayRunner(
        backend,
        DelayedIncrementalTokenizer(clock, delay_s=0.25),
        data["info"]["config"],
        algorithm="incremental",
        cache_block_tokens=1,
        time_scale=1,
        sleep=clock.sleep,
        now=clock.now,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] == 0
    assert backend.prefills == []
    assert all(output in backend.generations[1]["text"] for output in ("slow output", "middle output", "fast output"))


def test_single_tool_turn_uses_the_full_generation_prompt_without_prefill(tmp_path):
    data = make_trajectory(raw_output="only output")
    backend = FakeBackend()
    runner = make_runner(backend, data, FakeClock())

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[0]["prefill_submitted_count"] == 0
    assert backend.prefills == []
    assert "only output" in backend.generations[1]["text"]


def test_multi_tool_prompt_has_one_result_message_per_call(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    backend = FakeBackend()
    runner = make_runner(backend, data, FakeClock())

    runner.run_trajectory(tmp_path / "case.traj.json", data)

    next_prompt = backend.generations[1]["text"]
    assert next_prompt.count("<output>") == 3
    assert next_prompt.count("</output>") == 3
    assert next_prompt.count("<returncode>") == 3
    assert all(f'"command": "{command}"' in next_prompt for command in ("slow", "middle", "fast"))


def test_baseline_and_incremental_use_the_same_full_prompt(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    baseline_backend = FakeBackend()
    incremental_backend = FakeBackend()

    make_runner(
        baseline_backend,
        data,
        FakeClock(),
        algorithm="baseline",
    ).run_trajectory(tmp_path / "case.traj.json", data)
    make_runner(
        incremental_backend,
        data,
        FakeClock(),
    ).run_trajectory(tmp_path / "case.traj.json", data)

    assert baseline_backend.prefills == []
    assert baseline_backend.generations[1]["text"] == incremental_backend.generations[1]["text"]


MISTRAL_TOKENIZER = Path("/home/pjw7200/models/Mistral-Small-3.2-24B-Instruct-2506")


@pytest.mark.skipif(not MISTRAL_TOKENIZER.exists(), reason="Mistral tokenizer is not installed")
def test_mistral_partial_prompts_append_in_completion_order(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    tokenizer = ReplayTokenizer.from_path(str(MISTRAL_TOKENIZER))
    runner = IncrementalReplayRunner(
        FakeBackend(),
        tokenizer,
        data["info"]["config"],
        algorithm="incremental",
    )
    turn = prepare_replay_scenario(tmp_path / "case.traj.json", data).turns[0]
    completed = completed_tool_calls(turn.actions, turn.trace_tools)
    history = turn.leading_messages + [turn.assistant]
    first_prompt = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed[:1]),
        add_generation_prompt=False,
    ).token_ids
    second_prompt = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed[:2]),
        add_generation_prompt=False,
    ).token_ids
    full_prompt = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed),
        add_generation_prompt=True,
    ).token_ids

    assert second_prompt[: len(first_prompt)] == first_prompt
    assert full_prompt[: len(second_prompt)] == second_prompt
    rendered = tokenizer.tokenizer.decode(full_prompt)
    assert rendered.index('{"command": "slow"}') < rendered.index('{"command": "middle"}')
    assert rendered.index('{"command": "middle"}') < rendered.index('{"command": "fast"}')
    assert rendered.index("fast output") < rendered.index("middle output")
    assert rendered.index("middle output") < rendered.index("slow output")


@pytest.mark.skipif(not MISTRAL_TOKENIZER.exists(), reason="Mistral tokenizer is not installed")
def test_mistral_open_code_prompt_keeps_call_order_and_appends_result_order(tmp_path):
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(open_code_trace()))
    data = load_replay_source(path)
    tokenizer = ReplayTokenizer.from_path(str(MISTRAL_TOKENIZER))
    runner = IncrementalReplayRunner(
        FakeBackend(),
        tokenizer,
        {},
        algorithm="incremental",
    )
    turn = prepare_replay_scenario(path, data).turns[0]
    completed = completed_tool_calls(turn.actions, turn.trace_tools)
    history = turn.leading_messages + [turn.assistant]
    partial = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed[:1]),
        add_generation_prompt=False,
    ).token_ids
    full = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed),
        add_generation_prompt=True,
    ).token_ids

    assert full[: len(partial)] == partial
    rendered = tokenizer.tokenizer.decode(full)
    assert rendered.index('"filePath": "README.md"') < rendered.index('"command": "git status --short"')
    assert rendered.index("bash output") < rendered.index("read output")


def test_completed_tool_calls_keep_equal_completion_times_stable(tmp_path):
    data = multi_tool_trajectory(completion_offsets=[0.1, 0.1, 0.1])
    scenario = prepare_replay_scenario(tmp_path / "case.traj.json", data)
    turn = scenario.turns[0]

    completed = completed_tool_calls(turn.actions, turn.trace_tools)

    assert [call.action["tool_call_id"] for call in completed] == [
        "call_slow",
        "call_middle",
        "call_fast",
    ]


REPOSITORY_TRACE = (
    Path(__file__).resolve().parents[3]
    / "traces"
    / "analysisbench_minisweagent_toolcall_full_20260709T131115Z"
    / "gpu0"
    / "AFLplusplus_fastfetch"
    / "AFLplusplus_fastfetch.traj.json"
)


@pytest.mark.skipif(not REPOSITORY_TRACE.exists(), reason="Repository trace archive is not installed")
def test_repository_trace_uses_the_regular_replay_input_contract():
    data = load_trajectory(REPOSITORY_TRACE)

    scenario = prepare_replay_scenario(REPOSITORY_TRACE, data)
    turn = next(turn for turn in scenario.turns if len(turn.trace_tools) > 1)
    completed = completed_tool_calls(turn.actions, turn.trace_tools)

    assert len(completed) == len(turn.actions)
    assert [call.action["tool_call_id"] for call in completed] == [action["tool_call_id"] for action in turn.actions]


SWE_CHAT_OPENCODE_TRACE = Path("/home/pjw7200/traces/swe-chat/ses_277daa810ffePOBsdfS6gm8uA2/raw/full.jsonl")


@pytest.mark.skipif(not SWE_CHAT_OPENCODE_TRACE.exists(), reason="SWE-chat OpenCode trace is not installed")
def test_swe_chat_completion_times_drive_incremental_result_order():
    data = load_replay_source(SWE_CHAT_OPENCODE_TRACE)
    scenario = prepare_replay_scenario(SWE_CHAT_OPENCODE_TRACE, data)
    turn = next(turn for turn in scenario.turns if len(turn.trace_tools) > 1)
    completion_order = completed_tool_calls(turn.actions, turn.trace_tools)

    assert [call.action["command"].split(" ", 1)[0] for call in completion_order] == ["read", "webfetch"]

    backend = FakeBackend()
    make_runner(backend, data, FakeClock()).run_trajectory(SWE_CHAT_OPENCODE_TRACE, data)

    first_result = completion_order[0].trace.raw_output
    final_result = completion_order[1].trace.raw_output
    partial_prompt = backend.prefills[0]["text"]
    next_prompt = backend.generations[turn.step_index + 1]["text"]
    assert partial_prompt.index('"url": "https://docs.entire.io/cli/commands.md"') < partial_prompt.index(
        '"filePath": "/Users/savekirk/dev/ai/entire/session-bridge"'
    )
    assert first_result in partial_prompt
    assert final_result not in partial_prompt
    assert next_prompt.startswith(partial_prompt)
    assert next_prompt.index(first_result) < next_prompt.index(final_result)


def test_open_code_source_loads_the_full_session(tmp_path):
    path = tmp_path / "raw" / "full.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(open_code_trace()))

    data = load_replay_source(path)
    scenario = prepare_replay_scenario(path, data)

    assert data["trajectory_format"] == "swe-chat-opencode"
    assert scenario.instance_id == "ses_test"
    assert len(scenario.turns) == 2
    first_turn = scenario.turns[0]
    assert [call["function"]["name"] for call in first_turn.assistant["tool_calls"]] == ["read", "bash"]
    assert [action["command"] for action in first_turn.actions] == [
        'read {"filePath": "README.md"}',
        "git status --short",
    ]
    assert [tool.raw_output for tool in first_turn.trace_tools] == ["read output", "bash output"]
    assert [tool.completion_offset_s for tool in first_turn.trace_tools] == [0.3, 0.1]
    assert [
        call.action["tool_call_id"] for call in completed_tool_calls(first_turn.actions, first_turn.trace_tools)
    ] == [
        "call_bash",
        "call_read",
    ]


def test_codex_source_loads_completion_order_and_commands(tmp_path):
    path = tmp_path / "raw" / "full.jsonl"
    path.parent.mkdir()
    write_codex_trace(path)

    data = load_replay_source(path)
    scenario = prepare_replay_scenario(path, data)

    assert data["trajectory_format"] == "swe-chat-codex"
    assert scenario.instance_id == "codex_test"
    assert len(scenario.turns) == 2
    first_turn = scenario.turns[0]
    assert first_turn.leading_messages == [
        {"role": "system", "content": "Follow repository instructions."},
        {"role": "user", "content": "Inspect both files."},
    ]
    assert [action["command"] for action in first_turn.actions] == ["slow command", "fast command"]
    assert [tool.raw_output for tool in first_turn.trace_tools] == ["slow output", "fast output"]
    assert [tool.completion_offset_s for tool in first_turn.trace_tools] == pytest.approx([0.8, 0.4])
    assert first_turn.model_call["prompt_tokens"] == 100
    assert first_turn.trace_completion_tokens == 12
    assert first_turn.model_call["reasoning_tokens"] == 3
    assert [call.action["tool_call_id"] for call in completed_tool_calls(first_turn.actions, first_turn.trace_tools)] == [
        "call_fast",
        "call_slow",
    ]


def test_codex_history_message_during_tool_execution_follows_the_results(tmp_path):
    records = codex_trace()
    records.insert(
        7,
        {
            "timestamp": "2026-04-01T12:00:01.400Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "New permissions."}],
            },
        },
    )
    path = tmp_path / "full.jsonl"
    write_codex_trace(path, records)

    data = load_replay_source(path)
    first_assistant = next(index for index, message in enumerate(data["messages"]) if message["role"] == "assistant")

    assert [message["role"] for message in data["messages"][first_assistant : first_assistant + 4]] == [
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert data["messages"][first_assistant + 3]["content"] == "[Developer instructions]\nNew permissions."


@pytest.mark.skipif(not MISTRAL_TOKENIZER.exists(), reason="Mistral tokenizer is not installed")
def test_mistral_codex_prompt_appends_results_in_completion_order(tmp_path):
    path = tmp_path / "full.jsonl"
    write_codex_trace(path)
    data = load_replay_source(path)
    tokenizer = ReplayTokenizer.from_path(str(MISTRAL_TOKENIZER))
    runner = IncrementalReplayRunner(FakeBackend(), tokenizer, {}, algorithm="incremental")
    turn = prepare_replay_scenario(path, data).turns[0]
    completed = completed_tool_calls(turn.actions, turn.trace_tools)
    history = turn.leading_messages + [turn.assistant]
    partial = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed[:1]),
        add_generation_prompt=False,
    ).token_ids
    full = runner.encode_messages_with_state(
        history + runner.observation_messages_for(completed),
        add_generation_prompt=True,
    ).token_ids

    assert full[: len(partial)] == partial
    rendered = tokenizer.tokenizer.decode(full)
    assert rendered.index('"cmd": "slow command"') < rendered.index('"cmd": "fast command"')
    assert rendered.index("fast output") < rendered.index("slow output")


def test_open_code_usage_handles_separately_counted_reasoning_tokens(tmp_path):
    source = open_code_trace()
    source["messages"][1]["info"]["tokens"] = {
        "input": 5,
        "output": 3,
        "reasoning": 2,
        "cache": {"read": 7, "write": 0},
        "total": 17,
    }
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(source))

    turn = prepare_replay_scenario(path, load_replay_source(path)).turns[0]

    assert turn.model_call["prompt_tokens"] == 12
    assert turn.trace_completion_tokens == 5
    assert turn.model_call["reasoning_tokens"] == 2


def test_open_code_zero_token_message_stays_in_history_without_becoming_a_replay_turn(tmp_path):
    source = open_code_trace()
    source["messages"].insert(
        2,
        {
            "info": {
                "role": "assistant",
                "tokens": {"input": 0, "output": 0},
                "finish": "tool-calls",
            },
            "parts": [
                {
                    "type": "tool",
                    "tool": "task",
                    "callID": "call_task",
                    "state": {
                        "status": "completed",
                        "input": {"description": "background work"},
                        "output": "task output",
                        "time": {"start": 1_410, "end": 1_490},
                    },
                }
            ],
        },
    )
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(source))

    scenario = prepare_replay_scenario(path, load_replay_source(path))

    assert len(scenario.turns) == 2
    assert any(message.get("role") == "assistant" for message in scenario.turns[1].leading_messages)
    assert any(message.get("content") == "task output" for message in scenario.turns[1].leading_messages)


def test_open_code_incomplete_tool_stops_the_trajectory(tmp_path):
    source = open_code_trace()
    source["messages"][1]["parts"][1]["state"] = {
        "status": "running",
        "input": {"filePath": "README.md"},
        "time": {"start": 1_100},
    }
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(source))

    scenario = prepare_replay_scenario(path, load_replay_source(path))

    assert scenario.turns == []
    assert scenario.terminal_invalid is not None
    assert scenario.terminal_invalid["skip_reason"] == "incomplete_open_code_tool"


def test_open_code_reused_redacted_call_ids_are_made_unique(tmp_path):
    source = open_code_trace()
    source["messages"][1]["parts"][1]["callID"] = "REDACTED"
    source["messages"][1]["parts"][2]["callID"] = "REDACTED"
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(source))

    turn = prepare_replay_scenario(path, load_replay_source(path)).turns[0]

    assert [action["source_tool_call_id"] for action in turn.actions] == ["REDACTED", "REDACTED"]
    assert len({action["tool_call_id"] for action in turn.actions}) == 2
    assert [tool.raw_output for tool in turn.trace_tools] == ["read output", "bash output"]


def test_open_code_compaction_replaces_old_history_after_the_summary_turn(tmp_path):
    source = open_code_trace()
    source["messages"].insert(
        2,
        {
            "info": {"role": "user"},
            "parts": [{"type": "compaction", "auto": True}],
        },
    )
    source["messages"].extend(
        [
            {
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "Continue from the summary."}],
            },
            {
                "info": {
                    "role": "assistant",
                    "tokens": {"input": 12, "output": 3, "total": 15},
                    "finish": "stop",
                },
                "parts": [{"type": "text", "text": "Continued."}],
            },
        ]
    )
    path = tmp_path / "full.jsonl"
    path.write_text(json.dumps(source))
    data = load_replay_source(path)
    scenario = prepare_replay_scenario(path, data)

    assert scenario.turns[1].replay_history_after == [
        {
            "role": "system",
            "content": "Conversation summary:\nInspection complete.",
        }
    ]

    backend = FakeBackend()
    make_runner(backend, data, FakeClock()).run_trajectory(path, data)
    prompt_after_compaction = backend.generations[2]["text"]
    assert "Conversation summary:" in prompt_after_compaction
    assert "Continue from the summary." in prompt_after_compaction
    assert "Inspect the repository" not in prompt_after_compaction


def test_swe_chat_root_discovers_only_indexed_open_code_sources(tmp_path):
    open_code_path = tmp_path / "ses_open" / "raw" / "full.jsonl"
    claude_path = tmp_path / "ses_claude" / "raw" / "full.jsonl"
    open_code_path.parent.mkdir(parents=True)
    claude_path.parent.mkdir(parents=True)
    open_code_path.write_text(json.dumps(open_code_trace()))
    claude_path.write_text("{}")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "session-summary.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"session_id": "ses_claude", "format": "claude-jsonl"}),
                json.dumps({"session_id": "ses_open", "format": "opencode-json"}),
            ]
        )
    )

    assert collect_replay_sources(tmp_path) == [open_code_path]
    assert collect_replay_sources(tmp_path, swe_chat_format="codex-jsonl") == []


def test_incremental_replay_cli_uses_regular_trajectory_loading(tmp_path, monkeypatch):
    trajectory_path = tmp_path / "case.traj.json"
    data = multi_tool_trajectory(completion_offsets=[0.3, 0.2, 0.1])
    data["info"]["config"]["agent"]["tokenizer_path"] = "fake-tokenizer"
    trajectory_path.write_text(json.dumps(data))
    output = tmp_path / "replay-output"
    backend = FakeBackend()

    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.tokenizer_from_config",
        lambda config: ReplayTokenizer(CommandTemplateTokenizer()),
    )
    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.backend_from_config",
        lambda config: backend,
    )

    main(
        path=trajectory_path,
        output=output,
        algorithm="incremental",
        limit=None,
        swe_chat_format="opencode-json",
        config_spec=[],
    )

    results = (output / "replay_results.jsonl").read_text().splitlines()
    summary = json.loads((output / "summary.json").read_text())
    assert len(results) == 2
    assert summary["valid"] == 2


def test_incremental_replay_cli_resets_prefix_cache_before_each_trajectory(tmp_path, monkeypatch):
    data = make_trajectory()
    data["info"]["config"]["agent"]["tokenizer_path"] = "fake-tokenizer"
    for name in ("first.traj.json", "second.traj.json"):
        (tmp_path / name).write_text(json.dumps(data))
    output = tmp_path / "replay-output"
    backend = FakeBackend()

    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.tokenizer_from_config",
        lambda config: ReplayTokenizer(CommandTemplateTokenizer()),
    )
    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.backend_from_config",
        lambda config: backend,
    )

    main(
        path=tmp_path,
        output=output,
        algorithm="baseline",
        limit=None,
        swe_chat_format="opencode-json",
        config_spec=[],
    )

    assert backend.prefix_cache_resets == 2


def test_incremental_replay_cli_loads_open_code_directly(tmp_path, monkeypatch):
    trace_path = tmp_path / "raw" / "full.jsonl"
    trace_path.parent.mkdir()
    trace_path.write_text(json.dumps(open_code_trace()))
    output = tmp_path / "replay-output"
    backend = FakeBackend()

    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.tokenizer_from_config",
        lambda config: ReplayTokenizer(CommandTemplateTokenizer()),
    )
    monkeypatch.setattr(
        "minisweagent.run.extra.incremental_replay.backend_from_config",
        lambda config: backend,
    )

    main(
        path=trace_path,
        output=output,
        algorithm="incremental",
        limit=None,
        swe_chat_format="opencode-json",
        config_spec=[],
    )

    results = (output / "replay_results.jsonl").read_text().splitlines()
    summary = json.loads((output / "summary.json").read_text())
    assert len(results) == 2
    assert summary["valid"] == 2
    assert "bash output" in backend.prefills[0]["text"]
    assert "read output" not in backend.prefills[0]["text"]
