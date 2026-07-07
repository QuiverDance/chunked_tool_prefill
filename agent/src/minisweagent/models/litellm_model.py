import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        request_kwargs = self.config.model_kwargs | kwargs
        if request_kwargs.get("stream"):
            return self._query_streaming(messages, **request_kwargs)
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
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
            tools=[BASH_TOOL],
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

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        # Note: all model.query() implementations must persist the response on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        if model_timing := getattr(response, "_mswea_model_timing", None):
            message["extra"]["model_timing"] = model_timing
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class StreamingResponseBuilder:
    """Build a LiteLLM ModelResponse from OpenAI-compatible streaming chunks."""

    def __init__(self):
        self.response_id = None
        self.created = None
        self.model = None
        self.system_fingerprint = None
        self.role = "assistant"
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason = None
        self.usage = None

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        self.response_id = chunk.get("id") or self.response_id
        self.created = chunk.get("created") or self.created
        self.model = chunk.get("model") or self.model
        self.system_fingerprint = chunk.get("system_fingerprint") or self.system_fingerprint
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        self.finish_reason = choice.get("finish_reason") or self.finish_reason
        delta = choice.get("delta") or {}
        if delta.get("role"):
            self.role = delta["role"]
        if delta.get("content") is not None:
            self.content_parts.append(delta.get("content") or "")
        if delta.get("reasoning_content"):
            self.reasoning_parts.append(delta["reasoning_content"])
        for tool_call in delta.get("tool_calls") or []:
            self.add_tool_call(tool_call)

    def add_tool_call(self, tool_call: dict[str, Any]) -> None:
        index = int(tool_call.get("index") or 0)
        state = self.tool_calls.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
        )
        state["id"] = tool_call.get("id") or state["id"]
        state["type"] = tool_call.get("type") or state["type"]
        function = tool_call.get("function") or {}
        state["function"]["name"] = function.get("name") or state["function"]["name"]
        if function.get("arguments") is not None:
            state["function"]["arguments"] += function.get("arguments") or ""

    def response(self):
        from litellm.types.utils import ModelResponse

        message = {
            "role": self.role,
            "content": "".join(self.content_parts) if self.content_parts else None,
        }
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call["id"] or f"call_{index}",
                    "type": tool_call["type"] or "function",
                    "function": {
                        "name": tool_call["function"]["name"] or "",
                        "arguments": tool_call["function"]["arguments"],
                    },
                }
                for index, tool_call in sorted(self.tool_calls.items())
            ]
        if self.reasoning_parts:
            message["reasoning_content"] = "".join(self.reasoning_parts)

        return ModelResponse(
            id=self.response_id,
            created=self.created,
            model=self.model,
            object="chat.completion",
            system_fingerprint=self.system_fingerprint,
            choices=[{"index": 0, "finish_reason": self.finish_reason, "message": message}],
            usage=self.usage,
        )


def chunk_has_generated_payload(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
        return True
    if delta.get("function_call"):
        return True
    for tool_call in delta.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if tool_call.get("id") or function.get("name") or function.get("arguments"):
            return True
    return False
