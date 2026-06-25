import subprocess

from minisweagent.run.benchmarks.utils.token_timing import (
    SETUP_COMMANDS,
    CommandSegment,
    TokenTimingProgressAgent,
    instrumented_command,
    is_setup_segment,
    parse_instrumented_output,
    pipeline_category,
    shell_tokens,
)


class FakeProgress:
    def update_instance_status(self, instance_id, status):
        pass


class FakeModel:
    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {
                "role": "tool",
                "content": f"<returncode>{output['returncode']}</returncode>\n<output>{output['output']}</output>",
                "extra": output.get("extra", {}),
            }
            for output in outputs
        ]

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {"info": {"config": {"model": {}}}}


class FakeEnv:
    def __init__(self):
        self.commands = []

    def execute(self, action, cwd="", timeout=None):
        self.commands.append(action["command"])
        return {"output": "done\n", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {"info": {"config": {"environment": {}}}}


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text or "").split()


def make_agent(tmp_path):
    env = FakeEnv()
    agent = TokenTimingProgressAgent(
        FakeModel(),
        env,
        progress_manager=FakeProgress(),
        instance_id="case",
        system_template="",
        instance_template="",
        tokenizer_path="",
        output_path=tmp_path / "case.traj.json",
    )
    agent.tokenizer = WordTokenizer()
    return agent, env


def test_unclosed_quotes_do_not_break_command_categorization():
    command = 'python -c "print(\'unterminated)'

    assert shell_tokens(command)
    assert pipeline_category(command) == "python"
    assert not is_setup_segment(CommandSegment(command, "start"), SETUP_COMMANDS)


def test_submission_command_is_not_instrumented(tmp_path):
    agent, env = make_agent(tmp_path)
    command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"

    agent.execute_timed_action({"command": command})

    assert env.commands == [command]
    assert "__MSWEA_TOKEN_TIMING" not in env.commands[0]
    assert agent.tool_metrics == []


def test_instrumented_output_records_stderr_and_stream_events():
    marker = "__TEST_MARKER__"
    segments = [CommandSegment("printf 'out\\n'; printf 'err\\n' >&2", "start")]
    command = instrumented_command(segments[0].command, segments, marker)

    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, check=False)
    clean_output, records = parse_instrumented_output(result.stdout, marker=marker, segments=segments)

    assert "out" in clean_output
    assert "err" in clean_output
    record = records[0]
    assert record["first_stdout_ts"] is not None
    assert record["first_stderr_ts"] is not None
    assert record["first_output_ts"] == min(record["first_stdout_ts"], record["first_stderr_ts"])
    assert len(record["stdout_events"]) == 1
    assert len(record["stderr_events"]) == 1


def test_rendered_observation_metrics_are_owned_once(tmp_path):
    agent, _ = make_agent(tmp_path)
    output = {
        "output": "raw output",
        "returncode": 0,
        "exception_info": "",
        "extra": {
            "token_timing": {
                "tool_calls": [
                    {"command": "echo one"},
                    {"command": "echo two"},
                ]
            }
        },
    }
    messages = [{"role": "tool", "content": "rendered observation words"}]

    agent.attach_rendered_observation_metrics([output], messages)

    first, second = output["extra"]["token_timing"]["tool_calls"]
    assert first["rendered_observation_owner"] is True
    assert first["rendered_observation_tokens"] == 3
    assert first["rendered_chars"] == len("rendered observation words")
    assert second["rendered_observation_owner"] is False
    assert "rendered_observation_tokens" not in second
