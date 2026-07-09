"""Message cleanup helpers for replay."""

from __future__ import annotations

import copy
import json
from typing import Any


def api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in message.items() if key != "extra"} for message in messages]


def tokenizer_safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    safe_messages = []
    for message in messages:
        safe_message = {key: value for key, value in message.items() if key in allowed}
        if "tool_calls" in safe_message:
            safe_message["tool_calls"] = tokenizer_safe_tool_calls(safe_message["tool_calls"])
        safe_messages.append(safe_message)
    return safe_messages


def tokenizer_safe_tool_calls(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    safe_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            safe_calls.append(tool_call)
            continue
        safe_call = copy.deepcopy(tool_call)
        function = safe_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            try:
                function["arguments"] = json.loads(function["arguments"])
            except json.JSONDecodeError:
                pass
        safe_calls.append(safe_call)
    return safe_calls
