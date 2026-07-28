"""Load replay inputs without exposing their storage format to the runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import typer

from minisweagent.run.replay import load_trajectory

SWE_CHAT_INDEX = Path("analysis/session-summary.jsonl")
SWE_CHAT_TRACE = Path("raw/full.jsonl")
SWEChatFormat = Literal["opencode-json", "codex-jsonl"]

CODEX_CALL_OUTPUT_TYPES = {
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
    "tool_search_call": "tool_search_output",
}
CODEX_OUTPUT_TYPES = {output_type for output_type in CODEX_CALL_OUTPUT_TYPES.values()}


def collect_replay_sources(path: Path, swe_chat_format: SWEChatFormat = "opencode-json") -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise typer.BadParameter(f"Replay path does not exist: {path}")

    sources = list(path.rglob("*.traj.json"))
    swe_chat_index = path / SWE_CHAT_INDEX
    if swe_chat_index.is_file():
        sources.extend(_swe_chat_sources_from_index(path, swe_chat_index, swe_chat_format))
    else:
        direct_trace = _direct_swe_chat_trace(path)
        if direct_trace is not None:
            sources.append(direct_trace)

    return sorted(set(sources))


def load_replay_source(path: Path) -> dict[str, Any]:
    try:
        data = load_trajectory(path)
    except json.JSONDecodeError:
        return _codex_trajectory(_load_jsonl(path))
    if data.get("type") == "session_meta":
        return _codex_trajectory(_load_jsonl(path))
    if _is_open_code_trace(data):
        return _open_code_trajectory(data)
    return data


def _swe_chat_sources_from_index(
    root: Path,
    index_path: Path,
    swe_chat_format: SWEChatFormat,
) -> list[Path]:
    sources = []
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") != swe_chat_format:
            continue
        trace_path = root / str(row["session_id"]) / SWE_CHAT_TRACE
        if trace_path.is_file():
            sources.append(trace_path)
    return sources


def _direct_swe_chat_trace(path: Path) -> Path | None:
    candidates = (path / SWE_CHAT_TRACE, path / "full.jsonl")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _is_open_code_trace(data: dict[str, Any]) -> bool:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("info"), dict)
        and message["info"].get("role") in {"user", "assistant"}
        for message in messages
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
        records.append(record)
    if not any(record.get("type") == "session_meta" for record in records):
        raise ValueError(f"Unsupported replay JSONL in {path}")
    return records


def _codex_trajectory(records: list[dict[str, Any]]) -> dict[str, Any]:
    session = next((record.get("payload") or {} for record in records if record.get("type") == "session_meta"), {})
    messages: list[dict[str, Any]] = []
    current = _new_codex_response()

    def flush_response() -> None:
        nonlocal current
        if not current["content"] and not current["calls"]:
            current = _new_codex_response()
            return

        calls = current["calls"]
        missing_outputs = [call for call in calls if call["output"] is None]
        usage = current["usage"] or {}
        completion_tokens = _integer(usage.get("output_tokens"))
        replayable = completion_tokens is not None and completion_tokens > 0 and not missing_outputs
        assistant = {
            "role": "assistant",
            "content": "\n".join(current["content"]),
            "tool_calls": [_codex_tool_call(call) for call in calls],
            "extra": {
                "actions": [_codex_action(call) for call in calls],
                "replay": replayable,
                "token_timing": {
                    "model_call": {
                        "prompt_tokens": _integer(usage.get("input_tokens")),
                        "completion_tokens": completion_tokens,
                        "total_tokens": _integer(usage.get("total_tokens")),
                        "reasoning_tokens": _integer(usage.get("reasoning_output_tokens")),
                        "finish_reason": "tool_calls" if calls else "stop",
                        "ttft_s": None,
                        "model_total_s": None,
                        "decode_s": None,
                    }
                },
            },
        }
        if missing_outputs:
            assistant["extra"]["replay_invalid_reason"] = "incomplete_codex_tool"

        messages.append(assistant)
        completed_calls = sorted(
            (call for call in calls if call["output"] is not None),
            key=lambda call: (call["completed_at"] if call["completed_at"] is not None else float("inf"), call["index"]),
        )
        messages.extend(_codex_tool_message(call, calls) for call in completed_calls)
        for payload in current["following_messages"]:
            _append_codex_history_message(messages, payload)
        current = _new_codex_response()

    for record in records:
        payload = record.get("payload") or {}
        item_type = payload.get("type")

        if record.get("type") == "compacted":
            flush_response()
            replacement = _codex_replacement_history(payload.get("replacement_history") or [])
            assistant = next((message for message in reversed(messages) if message.get("role") == "assistant"), None)
            if assistant is not None and replacement:
                assistant.setdefault("extra", {})["replay_history_after"] = replacement
            continue

        if item_type == "message":
            role = payload.get("role")
            if role == "assistant":
                current["content"].extend(_codex_message_text(payload))
                continue
            if role in {"developer", "user"}:
                if current["calls"] and any(call["output"] is None for call in current["calls"]):
                    current["following_messages"].append(payload)
                    continue
                flush_response()
                _append_codex_history_message(messages, payload)
            continue

        if item_type in CODEX_CALL_OUTPUT_TYPES:
            if current["outputs_seen"]:
                flush_response()
            current["calls"].append(
                {
                    "index": len(current["calls"]),
                    "type": item_type,
                    "name": str(payload.get("name") or "tool_search"),
                    "call_id": str(payload.get("call_id") or f"codex_call_{len(messages)}_{len(current['calls'])}"),
                    "arguments": payload.get("arguments", payload.get("input")),
                    "called_at": _timestamp(record.get("timestamp")),
                    "completed_at": None,
                    "output": None,
                }
            )
            continue

        if item_type in CODEX_OUTPUT_TYPES:
            call_id = str(payload.get("call_id") or "")
            call = next((candidate for candidate in current["calls"] if candidate["call_id"] == call_id), None)
            if call is None:
                continue
            call["completed_at"] = _timestamp(record.get("timestamp"))
            call["output"] = _codex_output_text(payload)
            current["outputs_seen"] = True
            if all(candidate["output"] is not None for candidate in current["calls"]):
                flush_response()
            continue

        if item_type == "token_count" and (current["content"] or current["calls"]):
            info = payload.get("info") or {}
            usage = info.get("last_token_usage")
            if isinstance(usage, dict):
                current["usage"] = usage
                if not current["calls"]:
                    flush_response()
            continue

        if item_type in {"task_complete", "turn_aborted"}:
            flush_response()

    flush_response()
    return {
        "instance_id": str(session.get("id") or ""),
        "trajectory_format": "swe-chat-codex",
        "info": {"source": session},
        "messages": messages,
    }


def _new_codex_response() -> dict[str, Any]:
    return {
        "content": [],
        "calls": [],
        "usage": None,
        "outputs_seen": False,
        "following_messages": [],
    }


def _codex_message_text(payload: dict[str, Any]) -> list[str]:
    text = []
    for part in payload.get("content") or []:
        if not isinstance(part, dict):
            continue
        value = part.get("text")
        if value not in ("", None):
            text.append(str(value))
    return text


def _append_codex_history_message(messages: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    content = "\n".join(_codex_message_text(payload))
    if not content:
        return
    source_role = payload.get("role")
    role = "system" if source_role == "developer" and not messages else "user"
    if source_role == "developer" and role == "user":
        content = f"[Developer instructions]\n{content}"

    if messages and messages[-1].get("role") == role and not messages[-1].get("tool_calls"):
        messages[-1]["content"] = f"{messages[-1].get('content') or ''}\n\n{content}"
        return
    messages.append({"role": role, "content": content})


def _codex_replacement_history(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in items:
        if item.get("type") != "message" or item.get("role") not in {"developer", "user", "assistant"}:
            continue
        if item.get("role") == "assistant":
            content = "\n".join(_codex_message_text(item))
            if content:
                messages.append({"role": "assistant", "content": content})
            continue
        _append_codex_history_message(messages, item)
    return messages


def _codex_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call["call_id"],
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": _codex_arguments_json(call["arguments"]),
        },
    }


def _codex_action(call: dict[str, Any]) -> dict[str, Any]:
    arguments = _codex_arguments(call["arguments"])
    if call["name"] == "exec_command" and isinstance(arguments, dict) and isinstance(arguments.get("cmd"), str):
        command = arguments["cmd"]
    else:
        rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        command = f"{call['name']} {rendered}"
    return {
        "command": command,
        "tool_call_id": call["call_id"],
        "source_tool_call_id": call["call_id"],
    }


def _codex_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
    if value is None:
        return {}
    return value


def _codex_arguments_json(value: Any) -> str:
    return json.dumps(_codex_arguments(value), ensure_ascii=False)


def _codex_tool_message(call: dict[str, Any], group: list[dict[str, Any]]) -> dict[str, Any]:
    called_at = call["called_at"]
    completed_at = call["completed_at"]
    group_start = min(
        (candidate["called_at"] for candidate in group if candidate["called_at"] is not None),
        default=None,
    )
    duration_s = max(0.0, completed_at - called_at) if completed_at is not None and called_at is not None else 0.0
    completion_offset_s = (
        max(0.0, completed_at - group_start) if completed_at is not None and group_start is not None else None
    )
    raw_output = str(call["output"] or "")
    extra: dict[str, Any] = {
        "raw_output": raw_output,
        "returncode": _codex_returncode(raw_output),
        "exception_info": "",
    }
    if completion_offset_s is not None:
        extra["token_timing"] = {
            "tool_calls": [
                {
                    "duration_s": duration_s,
                    "completion_offset_s": completion_offset_s,
                    "output_events": [{"t": duration_s, "output_chars": len(raw_output)}],
                }
            ]
        }
    return {
        "role": "tool",
        "content": raw_output,
        "tool_call_id": call["call_id"],
        "extra": extra,
    }


def _codex_output_text(payload: dict[str, Any]) -> str:
    if "output" in payload:
        output = payload.get("output")
    else:
        output = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "call_id", "status", "execution"}
        }
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _codex_returncode(output: str) -> int:
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        metadata = value.get("metadata") or {}
        returncode = metadata.get("exit_code", value.get("returncode"))
        parsed = _integer(returncode)
        if parsed is not None:
            return parsed
    marker = "Process exited with code "
    if marker in output:
        tail = output.split(marker, 1)[1].splitlines()[0].strip()
        parsed = _integer(tail)
        if parsed is not None:
            return parsed
    return 0


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _open_code_trajectory(source: dict[str, Any]) -> dict[str, Any]:
    source_info = source.get("info") or {}
    messages: list[dict[str, Any]] = []
    compaction_pending = False
    used_tool_call_ids: set[str] = set()

    for message_index, source_message in enumerate(source.get("messages") or []):
        info = source_message.get("info") or {}
        role = info.get("role")
        parts = source_message.get("parts") or []

        if role == "user":
            user_message, compaction_pending = _open_code_user_message(parts)
            if user_message is not None:
                messages.append(user_message)
            continue

        if role != "assistant":
            continue

        tool_call_ids = _unique_tool_call_ids(parts, message_index, used_tool_call_ids)
        assistant, tool_messages = _open_code_assistant_messages(
            info,
            parts,
            tool_call_ids,
            reset_history_after=compaction_pending,
        )
        compaction_pending = False
        if assistant is None:
            continue
        messages.append(assistant)
        messages.extend(tool_messages)

    return {
        "instance_id": str(source_info.get("id") or ""),
        "trajectory_format": "swe-chat-opencode",
        "info": {"source": source_info},
        "messages": messages,
    }


def _open_code_user_message(parts: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    compaction = any(part.get("type") == "compaction" for part in parts)
    content = _join_part_text(parts, part_types={"text", "subtask"})
    if not content and compaction:
        content = "[Conversation compaction requested]"
    if not content:
        return None, compaction
    return {"role": "user", "content": content}, compaction


def _open_code_assistant_messages(
    info: dict[str, Any],
    parts: list[dict[str, Any]],
    tool_call_ids: list[str],
    *,
    reset_history_after: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    tool_parts = [part for part in parts if part.get("type") == "tool"]
    content = _join_part_text(parts, part_types={"text"})
    if not content and not tool_parts:
        return None, []

    incomplete_tool = next(
        (part for part in tool_parts if (part.get("state") or {}).get("status") not in {"completed", "error"}),
        None,
    )
    replayable = _completion_tokens(info) > 0 and incomplete_tool is None
    tool_calls = [_open_code_tool_call(part, call_id) for part, call_id in zip(tool_parts, tool_call_ids, strict=True)]
    actions = [_open_code_action(part, call_id) for part, call_id in zip(tool_parts, tool_call_ids, strict=True)]
    assistant = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {
            "actions": actions,
            "replay": replayable,
            "token_timing": {"model_call": _open_code_model_call(info)},
        },
    }
    if reset_history_after:
        assistant["extra"]["replay_history_after"] = [
            {
                "role": "system",
                "content": f"Conversation summary:\n{content}",
            }
        ]
    if incomplete_tool is not None:
        assistant["extra"]["replay_invalid_reason"] = "incomplete_open_code_tool"

    tool_messages = [
        _open_code_tool_message(part, call_id, tool_parts)
        for part, call_id in zip(tool_parts, tool_call_ids, strict=True)
        if (part.get("state") or {}).get("status") in {"completed", "error"}
    ]
    return assistant, tool_messages


def _unique_tool_call_ids(
    parts: list[dict[str, Any]],
    message_index: int,
    used: set[str],
) -> list[str]:
    call_ids = []
    tool_index = 0
    for part in parts:
        if part.get("type") != "tool":
            continue
        source_id = str(part.get("callID") or "")
        call_id = source_id
        if not call_id or call_id in used:
            base_id = f"open_code_{message_index}_{tool_index}"
            call_id = base_id
            suffix = 2
            while call_id in used:
                call_id = f"{base_id}_{suffix}"
                suffix += 1
        used.add(call_id)
        call_ids.append(call_id)
        tool_index += 1
    return call_ids


def _open_code_tool_call(part: dict[str, Any], call_id: str) -> dict[str, Any]:
    state = part.get("state") or {}
    arguments = state.get("input")
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": str(part.get("tool") or "tool"),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _open_code_action(part: dict[str, Any], call_id: str) -> dict[str, Any]:
    state = part.get("state") or {}
    arguments = state.get("input")
    tool_name = str(part.get("tool") or "tool")
    if tool_name == "bash" and isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
        command = arguments["command"]
    else:
        command = f"{tool_name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
    return {
        "command": command,
        "tool_call_id": call_id,
        "source_tool_call_id": str(part.get("callID") or ""),
    }


def _open_code_tool_message(
    part: dict[str, Any],
    call_id: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    state = part.get("state") or {}
    status = state.get("status")
    timing = state.get("time") or {}
    start_ms = _number(timing.get("start"))
    end_ms = _number(timing.get("end"))
    group_start_ms = min(
        (
            start
            for tool in group
            if (start := _number(((tool.get("state") or {}).get("time") or {}).get("start"))) is not None
        ),
        default=None,
    )
    duration_s = max(0.0, (end_ms - start_ms) / 1000) if start_ms is not None and end_ms is not None else 0.0
    completion_offset_s = (
        max(0.0, (end_ms - group_start_ms) / 1000) if end_ms is not None and group_start_ms is not None else None
    )
    output = state.get("output")
    error = state.get("error")
    raw_output = str(output if output is not None else error or "")
    metadata = state.get("metadata") or {}
    returncode = metadata.get("exit")
    if returncode is None:
        returncode = 0 if status == "completed" else 1

    extra: dict[str, Any] = {
        "raw_output": raw_output,
        "returncode": returncode,
        "exception_info": str(error or ""),
    }
    if completion_offset_s is not None:
        extra["token_timing"] = {
            "tool_calls": [
                {
                    "duration_s": duration_s,
                    "completion_offset_s": completion_offset_s,
                    "output_events": [{"t": duration_s, "output_chars": len(raw_output)}],
                }
            ]
        }

    return {
        "role": "tool",
        "content": raw_output,
        "tool_call_id": call_id,
        "extra": extra,
    }


def _open_code_model_call(info: dict[str, Any]) -> dict[str, Any]:
    tokens = info.get("tokens") or {}
    timing = info.get("time") or {}
    created_ms = _number(timing.get("created"))
    completed_ms = _number(timing.get("completed"))
    prompt_tokens, completion_tokens = _open_code_token_counts(info)
    model_total_s = (
        max(0.0, (completed_ms - created_ms) / 1000) if created_ms is not None and completed_ms is not None else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": tokens.get("total"),
        "reasoning_tokens": tokens.get("reasoning"),
        "finish_reason": info.get("finish"),
        "ttft_s": None,
        "model_total_s": model_total_s,
        "decode_s": None,
    }


def _completion_tokens(info: dict[str, Any]) -> int:
    _, value = _open_code_token_counts(info)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _open_code_token_counts(info: dict[str, Any]) -> tuple[int | None, int | None]:
    tokens = info.get("tokens") or {}
    prompt_tokens = _integer(tokens.get("input"))
    cache = tokens.get("cache") or {}
    cache_read = _integer(cache.get("read")) or 0
    cache_write = _integer(cache.get("write")) or 0
    if prompt_tokens is not None:
        prompt_tokens += cache_read + cache_write

    total_tokens = _integer(tokens.get("total"))
    if total_tokens is not None and prompt_tokens is not None:
        return prompt_tokens, max(0, total_tokens - prompt_tokens)
    return prompt_tokens, _integer(tokens.get("output"))


def _join_part_text(parts: list[dict[str, Any]], *, part_types: set[str]) -> str:
    text = []
    for part in parts:
        part_type = part.get("type")
        if part_type not in part_types:
            continue
        value = part.get("prompt") if part_type == "subtask" else part.get("text")
        if value not in ("", None):
            text.append(str(value))
    return "\n".join(text)


def _integer(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
