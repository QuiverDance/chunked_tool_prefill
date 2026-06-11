from minisweagent.run.benchmarks.utils.token_timing import (
    SETUP_COMMANDS,
    CommandSegment,
    is_setup_segment,
    pipeline_category,
    shell_tokens,
)


def test_unclosed_quotes_do_not_break_command_categorization():
    command = 'python -c "print(\'unterminated)'

    assert shell_tokens(command)
    assert pipeline_category(command) == "python"
    assert not is_setup_segment(CommandSegment(command, "start"), SETUP_COMMANDS)
