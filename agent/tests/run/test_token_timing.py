import sys

from minisweagent.run.benchmarks.utils.token_timing import (
    SETUP_COMMANDS,
    STREAM_READ_CHUNK_BYTES,
    TokenTimingProgressAgent,
    is_setup_command,
    pipeline_category,
    run_streaming_command,
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
    return agent, env


def test_unclosed_quotes_do_not_break_command_categorization():
    command = 'python -c "print(\'unterminated)'

    assert shell_tokens(command)
    assert pipeline_category(command) == "python"
    assert not is_setup_command(command, SETUP_COMMANDS)


def test_submission_command_is_not_instrumented(tmp_path):
    agent, env = make_agent(tmp_path)
    command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"

    agent.execute_timed_action({"command": command})

    assert env.commands == [command]
    assert "__MSWEA_TOKEN_TIMING" not in env.commands[0]
    assert agent.tool_metrics == []


def test_setup_command_is_executed_but_not_recorded_as_tool_metric(tmp_path):
    agent, env = make_agent(tmp_path)

    output = agent.execute_timed_action({"command": "cd /tmp"})

    assert env.commands == ["cd /tmp"]
    assert agent.tool_metrics == []
    assert output["extra"]["token_timing"]["tool_calls"] == []


def test_empty_command_is_executed_but_not_recorded_as_tool_metric(tmp_path):
    agent, env = make_agent(tmp_path)

    output = agent.execute_timed_action({"command": ""})

    assert env.commands == [""]
    assert agent.tool_metrics == []
    assert output["extra"]["token_timing"]["tool_calls"] == []


def test_tool_metric_keeps_runtime_tokenization_off_critical_path(tmp_path):
    agent, env = make_agent(tmp_path)

    output = agent.execute_timed_action({"command": "echo hello"})

    assert env.commands == ["echo hello"]
    assert output["output"] == "done\n"
    metric = agent.tool_metrics[0]
    assert "output_tokens" not in metric
    assert "stream_token_sample_count" not in metric
    assert "raw_chars" not in metric
    assert metric["raw_output_chars"] == len("done\n")
    assert metric["output_events"] == [
        {
            "t": metric["output_events"][0]["t"],
            "output_chars": len("done\n"),
            "output_bytes": len("done\n"),
        }
    ]


def test_streaming_command_records_output_as_the_runner_receives_it():
    code = "import time; print('one ', end='', flush=True); time.sleep(0.12); print('two ', end='', flush=True)"

    output, record = run_streaming_command(cmd=[sys.executable, "-c", code], timeout=5)

    assert output["output"] == "one two "
    assert record["first_output_ts"] < 0.08
    assert len(record["output_events"]) >= 2
    assert record["output_events"][-1]["output_chars"] == len("one two ")


def test_streaming_command_does_not_split_small_bursts_at_4k():
    burst_bytes = 5000
    code = f"import sys; sys.stdout.buffer.write(b'x' * {burst_bytes}); sys.stdout.flush()"

    output, record = run_streaming_command(cmd=[sys.executable, "-c", code], timeout=5)

    assert STREAM_READ_CHUNK_BYTES > burst_bytes > 4096
    assert output["output"] == "x" * burst_bytes
    assert len(record["output_events"]) == 1
    assert record["output_events"][0]["output_chars"] == burst_bytes
    assert record["output_events"][0]["output_bytes"] == burst_bytes


def test_unflushed_program_output_is_only_visible_after_flush():
    code = "import time; print('one ', end=''); time.sleep(0.12); print('two ', end='')"

    _, record = run_streaming_command(cmd=[sys.executable, "-c", code], timeout=5)

    assert record["first_output_ts"] >= 0.12
