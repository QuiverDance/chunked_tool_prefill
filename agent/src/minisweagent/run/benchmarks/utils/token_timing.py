"""Token and tool timing hooks for SWE-bench runs."""

from __future__ import annotations

import codecs
import os
import re
import selectors
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from minisweagent.agents.default import AgentConfig
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

SETUP_COMMANDS = ["cd", "export", "source", ".", "alias", "unalias", "set", "unset"]
SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
STREAM_SELECT_TIMEOUT_S = 0.01
STREAM_READ_CHUNK_BYTES = 64 * 1024
COMPOUND_KEYWORDS = {
    "case",
    "do",
    "done",
    "elif",
    "else",
    "esac",
    "fi",
    "for",
    "function",
    "if",
    "select",
    "then",
    "until",
    "while",
}

TOKENIZER_CACHE: dict[tuple[str, bool], Any] = {}
TOKENIZER_LOCK = threading.Lock()


class TokenTimingAgentConfig(AgentConfig):
    tokenizer_path: str = ""
    tokenizer_local_files_only: bool = True
    setup_command_categories: list[str] = Field(default_factory=lambda: list(SETUP_COMMANDS))


class TokenTimingProgressAgent(ProgressTrackingAgent):
    """Progress-tracking SWE-bench agent with model token and tool timing metrics."""

    def __init__(self, *args, progress_manager, instance_id: str = "", **kwargs):
        super().__init__(
            *args,
            progress_manager=progress_manager,
            instance_id=instance_id,
            config_class=TokenTimingAgentConfig,
            **kwargs,
        )
        self.model_metrics: list[dict[str, Any]] = []
        self.tool_metrics: list[dict[str, Any]] = []
        self.problem_timing: dict[str, Any] = {}

    def run(self, *args, **kwargs) -> dict:
        start_wall = time.time()
        start_perf = time.perf_counter()
        self.problem_timing = {
            "start_wall_s": start_wall,
            "end_wall_s": None,
            "e2e_s": None,
        }
        try:
            return super().run(*args, **kwargs)
        finally:
            end_wall = time.time()
            self.problem_timing = {
                "start_wall_s": start_wall,
                "end_wall_s": end_wall,
                "e2e_s": time.perf_counter() - start_perf,
            }
            self.save(self.config.output_path)

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"token_timing": {"problem": self.problem_timing}}}, *extra_dicts)

    def add_messages(self, *messages: dict) -> list[dict]:
        for message in messages:
            self.annotate_model_usage(message)
        return super().add_messages(*messages)

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = [self.execute_timed_action(action) for action in actions]
        observation_messages = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        return self.add_messages(*observation_messages)

    def annotate_model_usage(self, message: dict) -> None:
        extra = message.get("extra", {})
        response = extra.get("response")
        if not isinstance(response, dict) or "usage" not in response:
            return
        metric = {
            "instance_id": self.instance_id,
            "model_call_index": self.n_calls,
            **usage_from_response(response),
            **model_timing_from_extra(extra),
        }
        message.setdefault("extra", {}).setdefault("token_timing", {})["model_call"] = metric
        self.model_metrics.append(metric)

    def execute_timed_action(self, action: dict) -> dict:
        command = str(action.get("command") or "")
        if SUBMISSION_MARKER in command:
            return self.env.execute(action)

        output, record = execute_streaming_action(self.env, action)
        self.attach_tool_metric(action, output, command.strip(), record)
        return output

    def attach_tool_metric(self, action: dict, output: dict, command: str, record: dict[str, Any]) -> None:
        metrics = []
        if command and not record.get("skipped") and not is_setup_command(command, self.config.setup_command_categories):
            metric = self.tool_metric(action, command, record)
            metrics.append(metric)
            self.tool_metrics.append(metric)
        output.setdefault("extra", {})["token_timing"] = {"tool_calls": metrics}

    def tool_metric(self, action: dict, command: str, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "tool_call_id": action.get("tool_call_id"),
            "command_category": pipeline_category(command),
            "duration_s": record["duration_s"],
            "time_to_first_output_s": record["time_to_first_output_s"],
            "returncode": record["returncode"],
            "raw_output_chars": len(record["output"]),
            "raw_output_bytes": record.get("output_bytes"),
            "output_events": record.get("output_events", []),
        }


def load_tokenizer(tokenizer_path: str, *, local_files_only: bool):
    if not tokenizer_path:
        return None
    cache_key = (tokenizer_path, local_files_only)
    with TOKENIZER_LOCK:
        if cache_key not in TOKENIZER_CACHE:
            from transformers import AutoTokenizer

            TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        return TOKENIZER_CACHE[cache_key]


def count_tokens(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def execute_streaming_action(env: Any, action: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    command = str(action.get("command") or "")
    stream_command = streaming_command_for_env(env, command)
    if stream_command is None:
        return execute_with_end_sample(env, action)

    output, record = run_streaming_command(**stream_command)
    check_finished(env, output)
    return output, record


def streaming_command_for_env(env: Any, command: str) -> dict[str, Any] | None:
    config = getattr(env, "config", None)
    if config is None:
        return None

    if getattr(env, "container_id", None):
        executable = getattr(config, "executable", "docker")
        cwd = getattr(config, "cwd", "/")
        cmd = [executable, "exec", "-w", cwd]
        for key in getattr(config, "forward_env", []):
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in getattr(config, "env", {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        interpreter = [str(part) for part in getattr(config, "interpreter", ["bash", "-lc"])]
        cmd.extend([str(env.container_id), *interpreter, command])
        return {"cmd": cmd, "timeout": getattr(config, "timeout", 30)}

    class_name = env.__class__.__name__
    if class_name == "LocalEnvironment":
        cwd = getattr(config, "cwd", "") or os.getcwd()
        environment = os.environ | {str(key): str(value) for key, value in getattr(config, "env", {}).items()}
        return {
            "cmd": ["bash", "-c", command],
            "cwd": cwd,
            "env": environment,
            "timeout": getattr(config, "timeout", 30),
            "start_new_session": True,
        }

    return None


def execute_with_end_sample(env: Any, action: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    output = env.execute(action)
    duration = time.perf_counter() - start
    text = output.get("output", "") or ""
    output_bytes = len(text.encode("utf-8"))
    record = streaming_record(
        output=text,
        returncode=output.get("returncode"),
        exception_info=output.get("exception_info", ""),
        duration_s=duration,
        events=[{"t": duration, "output_chars": len(text), "output_bytes": output_bytes}] if text else [],
        final_output_bytes=output_bytes,
    )
    return output, record


def run_streaming_command(
    *,
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int | float | None = None,
    start_new_session: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    output_parts: list[str] = []
    events: list[dict[str, Any]] = []
    output_chars = 0
    output_bytes = 0
    exception_info = ""
    proc: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=start_new_session,
        )
        assert proc.stdout is not None
        selector.register(proc.stdout, selectors.EVENT_READ)

        while True:
            elapsed = time.perf_counter() - start
            if timeout is not None and elapsed > timeout:
                exception_info = f"An error occurred while executing the command: timed out after {timeout} seconds"
                terminate_process(proc, start_new_session=start_new_session)
                break

            for key, _ in selector.select(timeout=STREAM_SELECT_TIMEOUT_S):
                data = os.read(key.fileobj.fileno(), STREAM_READ_CHUNK_BYTES)
                if not data:
                    selector.unregister(key.fileobj)
                    break
                output_bytes += len(data)
                text = decoder.decode(data)
                if text:
                    output_parts.append(text)
                    output_chars += len(text)
                    events.append(
                        {
                            "t": time.perf_counter() - start,
                            "output_chars": output_chars,
                            "output_bytes": output_bytes,
                        }
                    )

            if proc.poll() is not None:
                remaining = b""
                if proc.stdout is not None:
                    try:
                        remaining = proc.stdout.read() or b""
                    except Exception:
                        remaining = b""
                output_bytes += len(remaining)
                text = decoder.decode(remaining, final=True)
                if text:
                    output_parts.append(text)
                    output_chars += len(text)
                    events.append(
                        {
                            "t": time.perf_counter() - start,
                            "output_chars": output_chars,
                            "output_bytes": output_bytes,
                        }
                    )
                break
    except Exception as e:
        exception_info = f"An error occurred while executing the command: {e}"
        if proc is not None and proc.poll() is None:
            terminate_process(proc, start_new_session=start_new_session)
    finally:
        selector.close()

    duration = time.perf_counter() - start
    output_text = "".join(output_parts)
    returncode = proc.returncode if proc is not None and proc.returncode is not None else -1
    output = {"output": output_text, "returncode": returncode, "exception_info": exception_info}
    return output, streaming_record(
        output=output_text,
        returncode=returncode,
        exception_info=exception_info,
        duration_s=duration,
        events=events,
        final_output_bytes=output_bytes,
    )


def streaming_record(
    *,
    output: str,
    returncode: int | None,
    exception_info: str,
    duration_s: float,
    events: list[dict[str, Any]],
    final_output_bytes: int | None = None,
) -> dict[str, Any]:
    first_output_ts = events[0]["t"] if events else None
    last_output_ts = events[-1]["t"] if events else None
    return {
        **empty_record(),
        "start_ts": 0.0,
        "first_stdout_ts": first_output_ts,
        "last_stdout_ts": last_output_ts,
        "first_output_ts": first_output_ts,
        "last_output_ts": last_output_ts,
        "end_ts": duration_s,
        "duration_s": duration_s,
        "time_to_first_stdout_s": first_output_ts,
        "time_to_first_output_s": first_output_ts,
        "returncode": returncode,
        "stdout": output,
        "stderr": "",
        "output": output,
        "output_bytes": final_output_bytes,
        "output_events": events,
        "exception_info": exception_info,
    }


def terminate_process(proc: subprocess.Popen[bytes], *, start_new_session: bool) -> None:
    try:
        if start_new_session:
            os.killpg(proc.pid, 9)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def check_finished(env: Any, output: dict[str, Any]) -> None:
    checker = getattr(env, "_check_finished", None)
    if checker is not None:
        checker(output)


def usage_from_response(response: dict) -> dict[str, Any]:
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": first_choice.get("finish_reason"),
    }


def model_timing_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    timing = extra.get("model_timing")
    if not isinstance(timing, dict):
        return {}
    return {
        "ttft_s": timing.get("ttft_s"),
        "model_total_s": timing.get("model_total_s"),
        "decode_s": timing.get("decode_s"),
    }


def pipeline_category(command: str) -> str:
    if is_compound_shell(command):
        return "compound"
    names = [command_name(part) for part, _ in split_top_level(command, ("|",))]
    return "|".join(name for name in names if name)


def command_name(command: str) -> str:
    tokens = shell_tokens(command)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in COMPOUND_KEYWORDS or is_assignment(token):
            index += 1
            continue
        if token == "env":
            index += 1
            while index < len(tokens) and (is_assignment(tokens[index]) or tokens[index].startswith("-")):
                index += 1
            continue
        if token == "timeout":
            index += 1
            while index < len(tokens) and (tokens[index].startswith("-") or re.match(r"^\d+[smhd]?$", tokens[index])):
                index += 1
            continue
        if token in {"sudo", "command", "builtin", "nohup", "time"}:
            index += 1
            continue
        return Path(token).name
    return ""


def is_setup_command(command: str, setup_categories: list[str]) -> bool:
    categories = pipeline_category(command).split("|")
    return bool(categories) and all(category in setup_categories for category in categories)


def shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        try:
            return shlex.split(command, comments=False, posix=False)
        except ValueError:
            return command.split()


def is_compound_shell(command: str) -> bool:
    return any(token in COMPOUND_KEYWORDS for token in shell_tokens(command))


def is_assignment(token: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", token) is not None


def split_top_level(command: str, separators: tuple[str, ...]) -> list[tuple[str, str]]:
    parts = []
    start = 0
    index = 0
    quote = ""
    escaped = False
    depth = 0

    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = quote != "'"
            index += 1
            continue
        if quote:
            quote = "" if char == quote else quote
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in "({[":
            depth += 1
            index += 1
            continue
        if char in ")}]" and depth:
            depth -= 1
            index += 1
            continue
        if not depth and (separator := separator_at(command, index, separators)):
            parts.append((command[start:index].strip(), separator))
            index += len(separator)
            start = index
            continue
        index += 1

    parts.append((command[start:].strip(), ""))
    return parts


def separator_at(command: str, index: int, separators: tuple[str, ...]) -> str:
    for separator in sorted(separators, key=len, reverse=True):
        if not command.startswith(separator, index):
            continue
        if separator == "|" and (command.startswith("||", index) or (index > 0 and command[index - 1] == "|")):
            continue
        return separator
    return ""


def empty_record() -> dict[str, Any]:
    return {
        "skipped": False,
        "start_ts": None,
        "first_stdout_ts": None,
        "last_stdout_ts": None,
        "first_stderr_ts": None,
        "last_stderr_ts": None,
        "first_output_ts": None,
        "last_output_ts": None,
        "end_ts": None,
        "duration_s": None,
        "time_to_first_stdout_s": None,
        "time_to_first_stderr_s": None,
        "time_to_first_output_s": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "output": "",
        "output_events": [],
        "stdout_events": "",
        "stderr_events": "",
        "output_samples": "",
    }
