import gzip
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from minisweagent.run.extra.branchfill_prefix_opportunity import analyze_trajectory, app, run_analysis


class ByteTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        encoded = list(text.encode("utf-8"))
        if not return_offsets_mapping:
            return {"input_ids": encoded}

        offsets = []
        byte_offset = 0
        for character in text:
            width = len(character.encode("utf-8"))
            offsets.extend([(byte_offset, byte_offset + width)] * width)
            byte_offset += width
        return {"input_ids": encoded, "offset_mapping": offsets}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return bytes(token_ids).decode("utf-8")


def assistant_message(call_id: str, command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
        "extra": {"actions": [{"command": command, "tool_call_id": call_id}]},
    }


def tool_message(call_id: str, output: str, category: str) -> dict[str, Any]:
    rendered = (
        f"<returncode>0</returncode>\n<output>\n{output}\n</output>"
        if len(output) < 10000
        else (
            "<returncode>0</returncode>\n<warning>output truncated</warning>\n"
            f"<output_head>\n{output[:5000]}\n</output_head>\n"
            f"<elided_chars>{len(output) - 10000}</elided_chars>\n"
            f"<output_tail>\n{output[-5000:]}\n</output_tail>"
        )
    )
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": rendered,
        "extra": {
            "raw_output": output,
            "returncode": 0,
            "token_timing": {
                "tool_calls": [
                    {
                        "tool_call_id": call_id,
                        "command_category": category,
                        "returncode": 0,
                    }
                ]
            },
        },
    }


def test_analysis_uses_only_earlier_outputs_and_keeps_candidate_pools_separate() -> None:
    trajectory = example_trajectory()

    rows = analyze_trajectory(trajectory, ByteTokenizer(), trajectory_path="case-1.traj.json")

    assert [row["any_prior_candidate_count"] for row in rows] == [0, 1, 2]
    assert [row["exact_args_candidate_count"] for row in rows] == [0, 0, 1]
    assert rows[1]["model_visible_any_prior_lcp_tokens"] == 2
    assert rows[1]["model_visible_exact_args_lcp_tokens"] == 0
    assert rows[2]["model_visible_any_prior_lcp_tokens"] == 3
    assert rows[2]["model_visible_exact_args_lcp_tokens"] == 3
    assert rows[2]["model_visible_any_prior_match_call_index"] == 0
    assert rows[2]["model_visible_any_prior_prefix_preview"] == "abc"


def test_analysis_exposes_effective_signature_and_resources_after_shell_setup() -> None:
    trajectory = {
        "instance_id": "signature-case",
        "messages": [
            assistant_message("call-1", "cd /testbed && pytest tests/test_a.py -q"),
            tool_message("call-1", "passed", "cd"),
        ],
    }

    row = analyze_trajectory(trajectory, ByteTokenizer())[0]

    assert row["recorded_command_category"] == "cd"
    assert row["effective_command"] == "pytest tests/test_a.py -q"
    assert row["command_category"] == "pytest"
    assert row["command_signature"] == "pytest -q"
    assert row["resource_keys"] == ["tests/test_a.py"]


def test_signatures_keep_subcommands_and_python_modules() -> None:
    trajectory = {
        "instance_id": "subcommand-case",
        "messages": [
            assistant_message("call-1", "git -C /testbed diff src/a.py --stat"),
            tool_message("call-1", "diff", "git"),
            assistant_message("call-2", "python -m pytest ./tests/test_a.py -q"),
            tool_message("call-2", "passed", "python"),
        ],
    }

    git_row, python_row = analyze_trajectory(trajectory, ByteTokenizer())

    assert git_row["command_signature"] == "git diff --stat -C"
    assert git_row["resource_keys"] == ["src/a.py"]
    assert python_row["command_signature"] == "python -m pytest -q"
    assert python_row["resource_keys"] == ["tests/test_a.py"]


def test_causal_policies_rank_recent_signature_and_matching_resources_without_seeing_output() -> None:
    trajectory = {
        "instance_id": "policy-case",
        "messages": [
            assistant_message("call-1", "cd /testbed && pytest tests/a.py -q"),
            tool_message("call-1", "abcdefOLD", "cd"),
            assistant_message("call-2", "pytest tests/b.py -q"),
            tool_message("call-2", "abcXYZ", "pytest"),
            assistant_message("call-3", "cat tests/a.py"),
            tool_message("call-3", "zzz", "cat"),
            assistant_message("call-4", "pytest tests/a.py -q"),
            tool_message("call-4", "abcdefNEW", "pytest"),
        ],
    }

    row = analyze_trajectory(trajectory, ByteTokenizer())[-1]

    assert row["model_visible_same_signature_lcp_tokens"] == 6
    assert row["policy_exact_args_recent_k1_lcp_tokens"] == 0
    assert row["policy_signature_recent_k1_lcp_tokens"] == 3
    assert row["policy_signature_recent_k2_lcp_tokens"] == 6
    assert row["policy_resource_aware_recent_k1_lcp_tokens"] == 6
    assert row["policy_command_similarity_k1_lcp_tokens"] == 6
    assert row["policy_combined_k1_lcp_tokens"] == 6


def test_run_analysis_writes_auditable_records_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "traces"
    trajectory_path = run_dir / "gpu0" / "case-1" / "case-1.traj.json"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(json.dumps(example_trajectory()))
    output_dir = tmp_path / "report"

    summary = run_analysis(run_dir, output_dir, ByteTokenizer())

    assert summary["trajectory_count"] == 1
    assert summary["tool_call_count"] == 3
    assert summary["model_visible"]["any_prior"]["output_tokens"] == 15
    assert summary["model_visible"]["any_prior"]["reusable_tokens"] == 5
    assert summary["model_visible"]["exact_args"]["reusable_tokens"] == 3
    exact_policy = summary["policy_frontier"]["exact_args_recent"]["1"]
    assert exact_policy["reusable_tokens"] == 3
    assert exact_policy["reuse_ratio"] == 3 / 15
    assert exact_policy["any_prior_oracle_capture"] == 3 / 5
    assert exact_policy["same_signature_oracle_capture"] == 1.0
    assert summary["policy_categories"]["pytest"]["combined"]["1"]["reusable_tokens"] == 3
    assert summary["model_visible"]["any_prior"]["lcp_tokens"]["max"] == 3
    assert summary["model_visible"]["any_prior"]["thresholds"]["1"]["calls"] == 2
    assert summary["model_visible"]["any_prior"]["trajectory_reuse_ratio"]["p25"] == 5 / 15
    assert summary["model_visible"]["any_prior"]["trajectory_reuse_ratio"]["median"] == 5 / 15
    assert summary["model_visible"]["any_prior"]["trajectory_reuse_ratio"]["p75"] == 5 / 15
    assert summary["model_visible"]["any_prior"]["reuse_ratio_ci95"] == {"low": 5 / 15, "high": 5 / 15}
    assert summary["rendering"]["full_calls"] == 3
    assert summary["rendering"]["truncated_calls"] == 0
    assert summary["command_categories"]["pytest"]["model_visible"]["any_prior"]["reusable_tokens"] == 3
    assert summary["output_length_buckets"]["1-31"]["tool_call_count"] == 3
    with gzip.open(output_dir / "per_call.jsonl.gz", "rt") as file:
        assert len(file.read().splitlines()) == 3
    with gzip.open(output_dir / "policy_per_call.jsonl.gz", "rt") as file:
        assert len(file.read().splitlines()) == 3
    assert json.loads((output_dir / "policy_frontier.json").read_text()) == summary["policy_frontier"]
    assert json.loads((output_dir / "policy_categories.json").read_text()) == summary["policy_categories"]
    top_matches = json.loads((output_dir / "top_matches.json").read_text())
    assert top_matches[0]["call_index"] == 2
    assert top_matches[0]["model_visible_any_prior_prefix_preview"] == "abc"
    assert top_matches[0]["model_visible_any_prior_match_command"] == "pytest"
    top_policy_matches = json.loads((output_dir / "top_policy_matches.json").read_text())
    assert top_policy_matches[0]["call_index"] == 2
    assert top_policy_matches[0]["policy_combined_k4_match_command"] == "pytest"
    assert json.loads((output_dir / "summary.json").read_text()) == summary
    report = (output_dir / "report.md").read_text()
    assert "BranchFill Prefix Opportunity" in report
    assert "Per-trajectory reuse ratio" in report
    assert "Output-length breakdown" in report
    assert "Causal policy frontier" in report


def test_truncated_outputs_reuse_only_the_visible_head_and_do_not_mix_render_kinds() -> None:
    first = "a" * 5000 + "X" + "z" * 5000
    second = "a" * 5000 + "Y" + "z" * 5000
    full = "a" * 6000
    trajectory = {
        "instance_id": "truncation-case",
        "messages": [
            assistant_message("call-1", "pytest"),
            tool_message("call-1", first, "pytest"),
            assistant_message("call-2", "pytest"),
            tool_message("call-2", second, "pytest"),
            assistant_message("call-3", "pytest"),
            tool_message("call-3", full, "pytest"),
        ],
    }

    rows = analyze_trajectory(trajectory, ByteTokenizer())

    assert rows[1]["render_kind"] == "truncated"
    assert rows[1]["model_visible_output_tokens"] == 10000
    assert rows[1]["model_visible_any_prior_lcp_tokens"] == 5000
    assert rows[2]["render_kind"] == "full"
    assert rows[2]["any_prior_candidate_count"] == 2
    assert rows[2]["model_visible_any_prior_candidate_count"] == 0
    assert rows[2]["model_visible_any_prior_lcp_tokens"] == 0


def test_cli_exposes_trace_and_output_directory_arguments() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "RUN_DIR" in result.stdout
    assert "--output-dir" in result.stdout


def example_trajectory() -> dict[str, Any]:
    return {
        "instance_id": "case-1",
        "messages": [
            assistant_message("call-1", "pytest"),
            tool_message("call-1", "abcXYZ", "pytest"),
            assistant_message("call-2", "compile"),
            tool_message("call-2", "abd", "compile"),
            assistant_message("call-3", "pytest"),
            tool_message("call-3", "abc123", "pytest"),
        ],
    }
