"""LiteLLM model wrapper for BrowseComp-Plus retrieval tools."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template
from pydantic import Field

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import (
    LitellmModel,
    LitellmModelConfig,
    StreamingResponseBuilder,
    chunk_has_generated_payload,
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the fixed BrowseComp-Plus corpus. Returns top hits with docid, score, and snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

GET_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_document",
        "description": "Retrieve a full document from the fixed BrowseComp-Plus corpus by docid.",
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {
                    "type": "string",
                    "description": "Document id returned by a previous search.",
                }
            },
            "required": ["docid"],
        },
    },
}


class BrowseCompToolModelConfig(LitellmModelConfig):
    include_get_document: bool = False
    tool_choice: str | dict[str, Any] | None = "auto"
    tool_parallel_calls: bool | None = None
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<tool>{{output.extra.tool_name}}</tool>\n"
        "<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )
    format_error_template: str = (
        "Tool call error:\n\n<error>\n{{ error }}\n</error>\n\n"
        "Use only the search tool{% if include_get_document %} or get_document tool{% endif %}, "
        "or provide the final answer directly with no tool call."
    )
    search_tool: dict[str, Any] = Field(default_factory=lambda: SEARCH_TOOL.copy())
    get_document_tool: dict[str, Any] = Field(default_factory=lambda: GET_DOCUMENT_TOOL.copy())


class BrowseCompToolModel(LitellmModel):
    """A LiteLLM chat model that exposes BrowseComp retrieval tools."""

    def __init__(self, *, config_class: Callable = BrowseCompToolModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _tools(self) -> list[dict[str, Any]]:
        tools = [self.config.search_tool]
        if self.config.include_get_document:
            tools.append(self.config.get_document_tool)
        return tools

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request_kwargs = self.config.model_kwargs | kwargs
        if self.config.tool_choice is not None and "tool_choice" not in request_kwargs:
            request_kwargs["tool_choice"] = self.config.tool_choice
        if self.config.tool_parallel_calls is not None and "parallel_tool_calls" not in request_kwargs:
            request_kwargs["parallel_tool_calls"] = self.config.tool_parallel_calls
        return request_kwargs

    def _query(self, messages: list[dict[str, str]], **kwargs):
        request_kwargs = self._request_kwargs(kwargs)
        if request_kwargs.get("stream"):
            return self._query_streaming(messages, **request_kwargs)
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=self._tools(),
                **request_kwargs,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _query_streaming(self, messages: list[dict[str, str]], **request_kwargs):
        request_kwargs = dict(request_kwargs)
        request_kwargs["stream"] = True
        request_kwargs.setdefault("stream_options", {"include_usage": True})

        start = time.perf_counter()
        first_chunk = None
        first_token = None
        builder = StreamingResponseBuilder()

        stream = litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=self._tools(),
            **request_kwargs,
        )
        for chunk in stream:
            now = time.perf_counter()
            if first_chunk is None:
                first_chunk = now
            data = chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else chunk
            builder.add_chunk(data)
            if first_token is None and chunk_has_generated_payload(data):
                first_token = now

        end = time.perf_counter()
        response = builder.response()
        response._mswea_model_timing = {
            "request_start_s": start,
            "first_chunk_s": (first_chunk - start) if first_chunk is not None else None,
            "ttft_s": (first_token - start) if first_token is not None else None,
            "model_total_s": end - start,
            "decode_s": (end - first_token) if first_token is not None else None,
        }
        return response

    def _parse_actions(self, response) -> list[dict]:
        message = response.choices[0].message
        tool_calls = _get(message, "tool_calls") or []
        if not tool_calls:
            return []

        actions = []
        for tool_call in tool_calls:
            function = _get(tool_call, "function") or {}
            name = _get(function, "name")
            raw_arguments = _get(function, "arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except Exception as e:
                self._raise_format_error(f"Error parsing {name or 'tool'} arguments: {e}.")

            if name == "search":
                query = arguments.get("query") if isinstance(arguments, dict) else None
                if not isinstance(query, str) or not query.strip():
                    self._raise_format_error("Missing non-empty 'query' argument in search tool call.")
                actions.append(
                    {
                        "tool_name": "search",
                        "arguments": {"query": query},
                        "query": query,
                        "tool_call_id": _get(tool_call, "id"),
                    }
                )
                continue

            if name == "get_document" and self.config.include_get_document:
                docid = arguments.get("docid") if isinstance(arguments, dict) else None
                if not isinstance(docid, str) or not docid.strip():
                    self._raise_format_error("Missing non-empty 'docid' argument in get_document tool call.")
                actions.append(
                    {
                        "tool_name": "get_document",
                        "arguments": {"docid": docid},
                        "docid": docid,
                        "tool_call_id": _get(tool_call, "id"),
                    }
                )
                continue

            allowed = "search and get_document" if self.config.include_get_document else "search"
            self._raise_format_error(f"Unknown tool '{name}'. Allowed tools: {allowed}.")

        return actions

    def _raise_format_error(self, error: str) -> None:
        raise FormatError(
            {
                "role": "user",
                "content": Template(self.config.format_error_template, undefined=StrictUndefined).render(
                    error=error,
                    include_get_document=self.config.include_get_document,
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
