import os

from tests.run.test_replay import (
    BlockingPrefillBackend,
    DelayedSeedTokenizer,
    FakeBackend,
    FakeClock,
    assistant_message,
    make_runner,
    make_runner_with_tokenizer,
    make_trajectory,
    tool_message,
)


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


def test_candidate_replay_uses_chunked_prefill_when_history_is_empty(tmp_path):
    data = make_trajectory(raw_output="first-output")
    data["info"]["config"]["model"]["stream_observation_template"] = (
        "<output>{{ output.output }}|rc={{ output.returncode }}|exc={{ output.exception_info }}"
    )
    data["messages"][3]["extra"]["returncode"] = 88
    data["messages"][3]["extra"]["exception_info"] = "not-visible-yet"
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=4,
        prefill_chunk_tokens=4,
        prefill_check_interval_s=0.001,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    first_tool_prefills = [prefill for prefill in backend.prefills if prefill["step_index"] == 0]
    assert invalid == []
    assert records[0]["candidate_selected_count"] == 0
    assert records[0]["candidate_fallback_to_chunked"] == 1
    assert "tool_output" in [prefill["label"] for prefill in first_tool_prefills]
    for prefill in first_tool_prefills:
        streamed_observation = prefill["text"].rsplit("tool:", 1)[-1]
        assert "rc=88" not in streamed_observation
        assert "not-visible-yet" not in streamed_observation


def test_candidate_prompt_does_not_copy_historical_result_metadata(tmp_path):
    data = make_candidate_trajectory()
    data["info"]["config"]["model"]["stream_observation_template"] = (
        "<output>{{ output.output }}|rc={{ output.returncode }}|exc={{ output.exception_info }}"
    )
    historical_tool = data["messages"][3]
    historical_tool["extra"]["returncode"] = 73
    historical_tool["extra"]["exception_info"] = "historical-secret"
    backend = FakeBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
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
    assert "rc=73" not in candidate_observation
    assert "historical-secret" not in candidate_observation


def test_candidate_prefill_is_cancelled_at_the_tool_deadline(tmp_path):
    data = make_candidate_trajectory()
    data["messages"][5] = tool_message(
        "shared-prefix-from-history",
        duration_s=0.05,
        events=[],
    )
    backend = BlockingPrefillBackend()
    runner = make_runner(
        backend,
        data,
        algorithm="candidate",
        candidate_top_k=1,
        cache_block_tokens=1,
    )

    records, invalid = runner.run_trajectory(tmp_path / "case.traj.json", data)

    assert invalid == []
    assert records[1]["candidate_completed_count"] == 0
    assert records[1]["active_prefill_cancel_requested_at_tool_end"] == 1


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
