"""Token and tool timing hooks for SWE-bench runs."""

from __future__ import annotations

import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from minisweagent.agents.default import AgentConfig
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

SETUP_COMMANDS = ["cd", "export", "source", ".", "alias", "unalias", "set", "unset"]
SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
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


@dataclass(frozen=True)
class CommandSegment:
    command: str
    separator: str


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
        self.tokenizer = load_tokenizer(
            self.config.tokenizer_path,
            local_files_only=self.config.tokenizer_local_files_only,
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
        self.attach_rendered_observation_metrics(outputs, observation_messages)
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
        command = action.get("command", "")
        if SUBMISSION_MARKER in command:
            return self.env.execute(action)

        segments = split_sequential_commands(command)
        marker = f"__MSWEA_TOKEN_TIMING_{uuid.uuid4().hex}__"

        output = self.env.execute({**action, "command": instrumented_command(command, segments, marker)})
        output["output"], segment_records = parse_instrumented_output(
            output.get("output", ""),
            marker=marker,
            segments=segments,
        )
        self.attach_tool_metrics(action, output, segments, segment_records)
        return output

    def attach_tool_metrics(
        self,
        action: dict,
        output: dict,
        segments: list[CommandSegment],
        segment_records: dict[int, dict[str, Any]],
    ) -> None:
        metrics = []
        for index, segment in enumerate(segments):
            if is_setup_segment(segment, self.config.setup_command_categories):
                continue
            record = segment_records.get(index)
            if not record or record.get("skipped"):
                continue
            metric = self.tool_metric(action, segment, index, record, output.get("exception_info", ""))
            metrics.append(metric)
            self.tool_metrics.append(metric)
        output.setdefault("extra", {})["token_timing"] = {"tool_calls": metrics}

    def attach_rendered_observation_metrics(self, outputs: list[dict], observation_messages: list[dict]) -> None:
        for output, message in zip(outputs, observation_messages):
            tool_calls = output.get("extra", {}).get("token_timing", {}).get("tool_calls", [])
            if not tool_calls:
                continue

            rendered = rendered_observation_text(message)
            summary = rendered_observation_summary(self.tokenizer, output, rendered)
            for index, metric in enumerate(tool_calls):
                metric["rendered_observation_owner"] = index == 0
                if index == 0:
                    metric.update(summary)

    def tool_metric(
        self,
        action: dict,
        segment: CommandSegment,
        index: int,
        record: dict[str, Any],
        exception_info: str,
    ) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "tool_call_id": action.get("tool_call_id"),
            "sequence_index": index,
            "sequence_separator": segment.separator,
            "command": segment.command,
            "command_category": pipeline_category(segment.command),
            "start_ts": record["start_ts"],
            "first_stdout_ts": record["first_stdout_ts"],
            "last_stdout_ts": record["last_stdout_ts"],
            "first_stderr_ts": record["first_stderr_ts"],
            "last_stderr_ts": record["last_stderr_ts"],
            "first_output_ts": record["first_output_ts"],
            "last_output_ts": record["last_output_ts"],
            "end_ts": record["end_ts"],
            "duration_s": record["duration_s"],
            "time_to_first_stdout_s": record["time_to_first_stdout_s"],
            "time_to_first_stderr_s": record["time_to_first_stderr_s"],
            "time_to_first_output_s": record["time_to_first_output_s"],
            "returncode": record["returncode"],
            "output_tokens": count_tokens(self.tokenizer, record["output"]),
            "stdout_tokens": count_tokens(self.tokenizer, record["stdout"]),
            "stderr_tokens": count_tokens(self.tokenizer, record["stderr"]),
            "raw_output_tokens": count_tokens(self.tokenizer, record["output"]),
            "raw_stdout_tokens": count_tokens(self.tokenizer, record["stdout"]),
            "raw_stderr_tokens": count_tokens(self.tokenizer, record["stderr"]),
            "raw_chars": len(record["output"]),
            "raw_stdout_chars": len(record["stdout"]),
            "raw_stderr_chars": len(record["stderr"]),
            **stream_timing_summary(self.tokenizer, record),
            "exception_info": exception_info,
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


def rendered_observation_text(message: dict) -> str:
    if "content" in message:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(text_part(item) for item in content if text_part(item))
    if "output" in message:
        return str(message.get("output") or "")
    return ""


def text_part(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    value = item.get("text") or item.get("content")
    return value if isinstance(value, str) else ""


def rendered_observation_summary(tokenizer, output: dict, rendered: str) -> dict[str, Any]:
    raw_output = output.get("output", "") or ""
    elided_chars = rendered_elided_chars(rendered)
    return {
        "rendered_observation_tokens": count_tokens(tokenizer, rendered),
        "rendered_chars": len(rendered),
        "was_truncated": elided_chars is not None,
        "elided_chars": elided_chars,
        "raw_output_chars": len(raw_output),
    }


def rendered_elided_chars(rendered: str) -> int | None:
    match = re.search(r"<elided_chars>\s*(\d+)\s*(?:</elided_chars>)?", rendered)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+characters\s+elided\b", rendered)
    return int(match.group(1)) if match else None


def stream_timing_summary(tokenizer, record: dict[str, Any]) -> dict[str, Any]:
    events = stream_events(record)
    duration = record.get("duration_s")
    raw_output_tokens = count_tokens(tokenizer, record.get("output", ""))
    summary: dict[str, Any] = {
        "stream_event_count": len(events),
        "stream_stdout_event_count": len(record.get("stdout_events", [])),
        "stream_stderr_event_count": len(record.get("stderr_events", [])),
        "stream_token_curve": stream_token_curve(tokenizer, events),
    }
    for seconds in (1, 5, 10, 30):
        tokens = tokens_before_end(tokenizer, events, duration, seconds)
        summary[f"stream_tokens_before_end_{seconds}s"] = tokens
        summary[f"stream_token_fraction_before_end_{seconds}s"] = safe_token_ratio(tokens, raw_output_tokens)
    half_tokens = tokens_before_time(tokenizer, events, duration / 2) if duration is not None else None
    summary["stream_tokens_before_half_duration"] = half_tokens
    summary["stream_token_fraction_before_half_duration"] = safe_token_ratio(half_tokens, raw_output_tokens)
    return summary


def stream_events(record: dict[str, Any]) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    for stream in ("stdout", "stderr"):
        timestamps = record.get(f"{stream}_events", [])
        lines = (record.get(stream, "") or "").splitlines(keepends=True)
        for timestamp, line in zip(timestamps, lines):
            if timestamp is not None:
                events.append((timestamp, line))
    return sorted(events, key=lambda item: item[0])


def stream_token_curve(tokenizer, events: list[tuple[float, str]]) -> list[dict[str, Any]]:
    if tokenizer is None:
        return []
    total = 0
    curve = []
    for timestamp, text in events:
        total += count_tokens(tokenizer, text) or 0
        curve.append({"t": timestamp, "tokens": total})
    return curve


def tokens_before_end(tokenizer, events: list[tuple[float, str]], duration: float | None, seconds_before_end: int) -> int | None:
    if duration is None:
        return None
    return tokens_before_time(tokenizer, events, max(0.0, duration - seconds_before_end))


def tokens_before_time(tokenizer, events: list[tuple[float, str]], cutoff: float) -> int | None:
    if tokenizer is None:
        return None
    return sum(count_tokens(tokenizer, text) or 0 for timestamp, text in events if timestamp <= cutoff)


def safe_token_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def usage_from_response(response: dict) -> dict[str, Any]:
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": first_choice.get("finish_reason"),
        "status": response.get("status"),
        "incomplete_details": response.get("incomplete_details"),
    }


def model_timing_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    timing = extra.get("model_timing")
    if not isinstance(timing, dict):
        return {}
    return {
        "stream": timing.get("stream"),
        "request_start_s": timing.get("request_start_s"),
        "first_chunk_s": timing.get("first_chunk_s"),
        "ttft_s": timing.get("ttft_s"),
        "model_total_s": timing.get("model_total_s"),
        "decode_s": timing.get("decode_s"),
        "stream_chunk_count": timing.get("stream_chunk_count"),
    }


def split_sequential_commands(command: str) -> list[CommandSegment]:
    command = command.strip()
    if not command:
        return []
    if "<<" in command or is_compound_shell(command):
        return [CommandSegment(command, "start")]

    segments = []
    separator = "start"
    for part, next_separator in split_top_level(command, ("&&", "||", ";")):
        if part:
            segments.append(CommandSegment(part, separator))
        separator = next_separator or separator
    return segments


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


def is_setup_segment(segment: CommandSegment, setup_categories: list[str]) -> bool:
    categories = pipeline_category(segment.command).split("|")
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


BASH_PREAMBLE = r"""
__mswea_marker=__MSWEA_MARKER__
__mswea_prev_rc=0
__mswea_final_rc=0

__mswea_capture() {
  local stream_file="$1"
  local combined_file="$2"
  local meta_file="$3"
  local event_file="$4"
  local line now
  while IFS= read -r line || [[ -n "$line" ]]; do
    now="$(date +%s%N)"
    [[ -s "$meta_file" ]] || printf '%s\n' "$now" >"$meta_file"
    printf '%s\n' "$now" >"$meta_file.last"
    printf '%s\n' "$now" >>"$event_file"
    printf '%s\n' "$line" >>"$stream_file"
    printf '%s\n' "$line" >>"$combined_file"
  done
}

__mswea_run_segment() {
  local index="$1"
  local command="$2"
  local stdout_file stderr_file combined_file stdout_meta stderr_meta stdout_events stderr_events stdout_fifo stderr_fifo
  stdout_file="$(mktemp)"
  stderr_file="$(mktemp)"
  combined_file="$(mktemp)"
  stdout_meta="$(mktemp)"
  stderr_meta="$(mktemp)"
  stdout_events="$(mktemp)"
  stderr_events="$(mktemp)"
  stdout_fifo="$(mktemp -u)"
  stderr_fifo="$(mktemp -u)"
  mkfifo "$stdout_fifo" "$stderr_fifo"

  local start_ns end_ns rc stdout_pid stderr_pid
  start_ns="$(date +%s%N)"
  __mswea_capture "$stdout_file" "$combined_file" "$stdout_meta" "$stdout_events" <"$stdout_fifo" &
  stdout_pid=$!
  __mswea_capture "$stderr_file" "$combined_file" "$stderr_meta" "$stderr_events" <"$stderr_fifo" &
  stderr_pid=$!
  eval "$command" >"$stdout_fifo" 2>"$stderr_fifo"
  rc=$?
  wait "$stdout_pid" || true
  wait "$stderr_pid" || true
  end_ns="$(date +%s%N)"

  if [[ "$rc" -eq 0 ]]; then
    local first_line
    IFS= read -r first_line <"$combined_file" || true
    if [[ "$first_line" == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" ]]; then
      cat "$combined_file"
      __mswea_cleanup
      exit "$rc"
    fi
  fi

  cat "$combined_file"
  printf '%s\t%s\tMETA\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$__mswea_marker" "$index" "$start_ns" \
    "$(cat "$stdout_meta" 2>/dev/null)" "$(cat "$stdout_meta.last" 2>/dev/null)" \
    "$(cat "$stderr_meta" 2>/dev/null)" "$(cat "$stderr_meta.last" 2>/dev/null)" "$end_ns" "$rc"
  printf '%s\t%s\tSTDOUT\n' "$__mswea_marker" "$index"
  cat "$stdout_file"
  printf '%s\t%s\tEND\n' "$__mswea_marker" "$index"
  printf '%s\t%s\tSTDERR\n' "$__mswea_marker" "$index"
  cat "$stderr_file"
  printf '%s\t%s\tEND\n' "$__mswea_marker" "$index"
  printf '%s\t%s\tSTDOUT_EVENTS\n' "$__mswea_marker" "$index"
  cat "$stdout_events"
  printf '%s\t%s\tEND\n' "$__mswea_marker" "$index"
  printf '%s\t%s\tSTDERR_EVENTS\n' "$__mswea_marker" "$index"
  cat "$stderr_events"
  printf '%s\t%s\tEND\n' "$__mswea_marker" "$index"
  __mswea_cleanup
  return "$rc"
}

__mswea_cleanup() {
  rm -f "$stdout_file" "$stderr_file" "$combined_file"
  rm -f "$stdout_meta" "$stdout_meta.last" "$stderr_meta" "$stderr_meta.last"
  rm -f "$stdout_events" "$stderr_events"
  rm -f "$stdout_fifo" "$stderr_fifo"
}
"""


def instrumented_command(command: str, segments: list[CommandSegment], marker: str) -> str:
    if not segments:
        return command

    lines = [BASH_PREAMBLE.replace("__MSWEA_MARKER__", shlex.quote(marker))]
    for index, segment in enumerate(segments):
        condition = {
            "start": "true",
            ";": "true",
            "&&": '[[ "$__mswea_prev_rc" -eq 0 ]]',
            "||": '[[ "$__mswea_prev_rc" -ne 0 ]]',
        }[segment.separator]
        lines.extend(
            [
                f"if {condition}; then",
                f"  __mswea_run_segment {index} {shlex.quote(segment.command)}",
                "  __mswea_prev_rc=$?",
                "  __mswea_final_rc=$__mswea_prev_rc",
                "else",
                f"  printf '%s\\t%s\\tSKIPPED\\t%s\\n' \"$__mswea_marker\" {index} \"$__mswea_prev_rc\"",
                "fi",
            ]
        )
    lines.append('exit "$__mswea_final_rc"')
    return "\n".join(lines)


def parse_instrumented_output(
    output: str,
    *,
    marker: str,
    segments: list[CommandSegment],
) -> tuple[str, dict[int, dict[str, Any]]]:
    clean_output = []
    records: dict[int, dict[str, Any]] = {}
    open_stream: tuple[int, str] | None = None

    for line in output.splitlines(keepends=True):
        fields = marker_fields(line, marker)
        if open_stream:
            index, stream = open_stream
            if fields and marker_index(fields) == index and fields[2] == "END":
                open_stream = None
            else:
                records.setdefault(index, empty_record())[stream] += line
            continue

        if not fields:
            clean_output.append(line)
            continue

        index = marker_index(fields)
        if not 0 <= index < len(segments):
            continue
        if fields[2] == "META" and len(fields) in {8, 10}:
            records.setdefault(index, empty_record()).update(meta_record(fields))
        elif fields[2] == "SKIPPED" and len(fields) == 4:
            records[index] = {**empty_record(), "skipped": True, "returncode": int(fields[3])}
        elif fields[2] in {"STDOUT", "STDERR", "STDOUT_EVENTS", "STDERR_EVENTS"}:
            open_stream = (index, fields[2].lower())

    for record in records.values():
        record["output"] = record["stdout"] + record["stderr"]
        record["stdout_events"] = event_times(record["stdout_events"], record.get("start_ns"))
        record["stderr_events"] = event_times(record["stderr_events"], record.get("start_ns"))
    return "".join(clean_output), records


def marker_fields(line: str, marker: str) -> list[str]:
    fields = line.rstrip("\n").split("\t")
    return fields if fields and fields[0] == marker else []


def marker_index(fields: list[str]) -> int:
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return -1


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
        "stdout_events": "",
        "stderr_events": "",
    }


def meta_record(fields: list[str]) -> dict[str, Any]:
    start_ns = int(fields[3])
    first_stdout_ts = relative_seconds(fields[4], start_ns)
    last_stdout_ts = relative_seconds(fields[5], start_ns)
    if len(fields) == 10:
        first_stderr_ts = relative_seconds(fields[6], start_ns)
        last_stderr_ts = relative_seconds(fields[7], start_ns)
        end_ts = relative_seconds(fields[8], start_ns)
        returncode = int(fields[9])
    else:
        first_stderr_ts = None
        last_stderr_ts = None
        end_ts = relative_seconds(fields[6], start_ns)
        returncode = int(fields[7])
    first_output_ts = min_time(first_stdout_ts, first_stderr_ts)
    last_output_ts = max_time(last_stdout_ts, last_stderr_ts)
    return {
        "start_ns": start_ns,
        "start_ts": 0.0,
        "first_stdout_ts": first_stdout_ts,
        "last_stdout_ts": last_stdout_ts,
        "first_stderr_ts": first_stderr_ts,
        "last_stderr_ts": last_stderr_ts,
        "first_output_ts": first_output_ts,
        "last_output_ts": last_output_ts,
        "end_ts": end_ts,
        "duration_s": end_ts,
        "time_to_first_stdout_s": first_stdout_ts,
        "time_to_first_stderr_s": first_stderr_ts,
        "time_to_first_output_s": first_output_ts,
        "returncode": returncode,
    }


def relative_seconds(value: str, start_ns: int) -> float | None:
    if not value:
        return None
    return (int(value) - start_ns) / 1_000_000_000


def event_times(raw_events: str, start_ns: int | None) -> list[float | None]:
    if start_ns is None:
        return []
    times = []
    for line in (raw_events or "").splitlines():
        try:
            times.append(relative_seconds(line, start_ns))
        except ValueError:
            times.append(None)
    return times


def min_time(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def max_time(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None
