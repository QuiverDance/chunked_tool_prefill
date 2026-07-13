import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from minisweagent.run.extra.branchahead_opportunity import (
    analyze_trajectory,
    app,
    fallback_completion_tokens,
    run_analysis,
    tool_duration,
)


class TemplateByteTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        token_ids = self.encode(text)
        if not return_offsets_mapping:
            return {"input_ids": token_ids}
        return {"input_ids": token_ids, "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        prefix = "<assistant><think>\n"
        if add_generation_prompt:
            return prefix

        assistant = messages[-1]
        content = str(assistant.get("content") or "")
        commands = [json.loads(call["function"]["arguments"])["command"] for call in assistant.get("tool_calls") or []]
        return prefix + "\n</think>\n" + content + "|" + "|".join(commands) + "<end>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return bytes(token_ids).decode()


def test_analysis_recovers_response_drafts_when_observation_prefixes_diverge() -> None:
    rows = analyze_trajectory(example_trajectory(), TemplateByteTokenizer(), trajectory_path="case.traj.json")

    partial_observation = rows[1]
    assert partial_observation["policy_command_similarity_k1_observation_lcp_tokens"] == len("error ")
    assert partial_observation["policy_command_similarity_k1_response_lcp_tokens"] == len("plan shared ")
    assert partial_observation["policy_command_similarity_k1_salvage_group"] == "partial_observation_response_hit"

    failed_observation = rows[2]
    assert failed_observation["policy_command_similarity_k1_observation_lcp_tokens"] == 0
    assert failed_observation["policy_command_similarity_k1_response_lcp_tokens"] == len("plan shared B")
    assert failed_observation["policy_command_similarity_k1_salvage_group"] == "zero_observation_response_hit"
    assert failed_observation["policy_command_similarity_k1_next_tool_exact"] is True
    assert failed_observation["policy_command_similarity_k1_next_tool_match_commands"] == ["cat parser.py"]
    assert failed_observation["policy_command_similarity_k1_joint_opportunity_s"] > 0
    assert failed_observation["next_tool_duration_s"] == 12.0

    assert rows[0]["policy_any_prior_k0_selected_count"] == 0
    assert rows[1]["policy_any_prior_k0_selected_count"] == 1


def test_run_analysis_reports_response_and_next_tool_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "traces"
    trajectory_path = run_dir / "case.traj.json"
    run_dir.mkdir()
    trajectory_path.write_text(json.dumps(example_trajectory()))

    summary = run_analysis(
        run_dir,
        tmp_path / "report",
        TemplateByteTokenizer(),
        tokenizer_path="test-tokenizer",
    )

    policy = summary["policy_frontier"]["command_similarity"]["1"]
    assert policy["response_lcp_tokens"] > 0
    assert policy["response_draft_coverage"] > 0
    assert policy["salvage_groups"]["zero_observation_response_hit"]["calls"] >= 1
    assert policy["next_tool_exact_calls"] >= 1
    assert policy["next_tool_time_coverage"] > 0
    assert policy["joint_opportunity_s"] > 0
    assert summary["tokenizer_path"] == "test-tokenizer"
    assert (tmp_path / "report" / "per_call.jsonl.gz").is_file()
    assert (tmp_path / "report" / "top_response_matches.json").is_file()
    assert (tmp_path / "report" / "top_tool_time_hits.json").is_file()
    assert "BranchAhead Offline Opportunity" in (tmp_path / "report" / "report.md").read_text()


def test_cli_exposes_trace_and_output_directory_arguments() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "RUN_DIR" in result.stdout
    assert "--output-dir" in result.stdout


def test_fallback_serializes_tool_calls_and_idless_duration_requires_one_metric() -> None:
    message = assistant_message("call-1", "pytest", "reason")
    tokens = fallback_completion_tokens(
        message,
        TemplateByteTokenizer(),
        use_recorded_completion_length=False,
    )
    decoded = TemplateByteTokenizer().decode(tokens)
    assert '"name": "bash"' in decoded
    assert '"command": "pytest"' in decoded

    one_metric = {"extra": {"token_timing": {"tool_calls": [{"duration_s": 3.0}]}}}
    ambiguous_metrics = {
        "extra": {"token_timing": {"tool_calls": [{"duration_s": 3.0}, {"duration_s": 4.0}]}}
    }
    assert tool_duration(one_metric, "call-1") == 3.0
    assert tool_duration(ambiguous_metrics, "call-1") is None


def test_full_observation_match_uses_truncated_model_visible_head_and_tail() -> None:
    first = "h" * 5000 + "a" * 12 + "t" * 5000
    second = "h" * 5000 + "b" * 12 + "t" * 5000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        assistant_message("call-1", "build", "candidate"),
        truncated_tool_message("call-1", first),
        assistant_message("call-2", "build again", "shared response"),
        truncated_tool_message("call-2", second),
        final_message("shared response continued"),
    ]
    tokenizer = TemplateByteTokenizer()
    for message in messages:
        if message.get("role") == "assistant":
            message["extra"]["token_timing"] = {
                "model_call": {"completion_tokens": len(tokenizer.encode(completion_text(message)))}
            }

    rows = analyze_trajectory({"instance_id": "truncated", "messages": messages}, tokenizer)

    assert rows[1]["policy_any_prior_k0_full_observation_match"] is True


def example_trajectory() -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        assistant_message("call-1", "pytest a.py", "initial"),
        tool_message("call-1", "error one", 0.2),
        assistant_message("call-2", "sed parser.py", "plan shared A"),
        tool_message("call-2", "error two", 0.3),
        assistant_message("call-3", "cat parser.py", "plan shared B"),
        tool_message("call-3", "totally different", 0.4),
        assistant_message("call-4", "cat parser.py", "plan shared B more"),
        tool_message("call-4", "done", 12.0),
        final_message("finished"),
    ]
    tokenizer = TemplateByteTokenizer()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        completion = completion_text(message)
        message["extra"]["token_timing"] = {
            "model_call": {"completion_tokens": len(tokenizer.encode(completion)), "decode_s": 1.0}
        }
    return {
        "instance_id": "case",
        "info": {
            "config": {"model": {"model_name": "test-model"}},
            "token_timing": {"problem": {"e2e_s": 30.0}},
        },
        "messages": messages,
    }


def assistant_message(call_id: str, command: str, reasoning: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        "extra": {"actions": [{"command": command, "tool_call_id": call_id}]},
    }


def final_message(reasoning: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [],
        "extra": {"actions": []},
    }


def completion_text(message: dict[str, Any]) -> str:
    commands = [json.loads(call["function"]["arguments"])["command"] for call in message.get("tool_calls") or []]
    return str(message.get("reasoning_content") or "") + "\n</think>\n|" + "|".join(commands)


def tool_message(call_id: str, output: str, duration_s: float) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": f"<returncode>0</returncode>\n<output>\n{output}\n</output>",
        "extra": {
            "raw_output": output,
            "returncode": 0,
            "token_timing": {
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "command_category": "test",
                        "duration_s": duration_s,
                        "returncode": 0,
                    }
                ]
            },
        },
    }


def truncated_tool_message(call_id: str, output: str) -> dict[str, Any]:
    message = tool_message(call_id, output, 1.0)
    message["content"] = (
        f"<output_head>\n{output[:5000]}\n</output_head>"
        f"<elided_chars>{len(output) - 10000}</elided_chars>"
        f"<output_tail>\n{output[-5000:]}\n</output_tail>"
    )
    return message
