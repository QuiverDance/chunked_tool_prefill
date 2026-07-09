"""HTTP backend for live replay."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from minisweagent.run.replay_types import ReplayError, ReplayStep


class HttpReplayBackend:
    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        prefill_url: str = "",
        completion_url: str = "",
        timeout: float = 600,
        ignore_eos: bool = True,
    ):
        self.model_name = served_model_name(model_name)
        self.prefill_url = prefill_url
        self.prefill_abort_url = f"{prefill_url.rstrip('/')}/abort" if prefill_url else ""
        self.completion_url = completion_url or f"{api_base.rstrip('/')}/completions"
        self.timeout = timeout
        self.ignore_eos = ignore_eos

    def prefill(
        self,
        token_ids: list[int],
        *,
        cache_salt: str,
        step: ReplayStep,
        label: str,
        request_id: str | None = None,
    ) -> None:
        if not self.prefill_url:
            raise ReplayError("missing_prefill_url")
        payload = {
            "model": self.model_name,
            "input_token_ids": token_ids,
            "cache_salt": cache_salt,
            "metadata": {
                "instance_id": step.instance_id,
                "step_index": step.step_index,
                "label": label,
            },
        }
        if request_id:
            payload["request_id"] = request_id
        try:
            response = requests.post(self.prefill_url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise ReplayError(f"prefill_failed:{type(e).__name__}") from e
        if response.status_code >= 400:
            raise ReplayError(f"prefill_failed:{response.status_code}:{response.text[:500]}")

    def cancel_prefill(self, request_id: str) -> None:
        if not self.prefill_abort_url:
            raise ReplayError("missing_prefill_abort_url")
        try:
            response = requests.post(self.prefill_abort_url, json={"request_id": request_id}, timeout=5)
        except requests.RequestException as e:
            raise ReplayError(f"prefill_cancel_failed:{type(e).__name__}") from e
        if response.status_code >= 400:
            raise ReplayError(f"prefill_cancel_failed:{response.status_code}:{response.text[:500]}")

    def generate_tokens(
        self,
        token_ids: list[int],
        *,
        max_tokens: int,
        cache_salt: str,
        step: ReplayStep,
        label: str,
    ) -> dict[str, Any]:
        metadata = {
            "instance_id": step.instance_id,
            "step_index": step.step_index,
            "prompt_tokens": len(token_ids),
            "label": label,
        }
        payload = {
            "model": self.model_name,
            "prompt": token_ids,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": max(1, int(max_tokens)),
            "ignore_eos": self.ignore_eos,
            "cache_salt": cache_salt,
            "metadata": metadata,
        }

        start = time.perf_counter()
        cached_tokens = None
        prompt_tokens = None
        completion_tokens = None
        first_token_at = None
        ttft = None
        with requests.post(self.completion_url, json=payload, stream=True, timeout=self.timeout) as response:
            if response.status_code >= 400:
                raise ReplayError(f"completion_failed:{response.status_code}:{response.text[:500]}")
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line.removeprefix("data:").strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cached_tokens = cached_tokens_from_chunk(chunk, cached_tokens)
                prompt_tokens = prompt_tokens_from_chunk(chunk, prompt_tokens)
                completion_tokens = completion_tokens_from_chunk(chunk, completion_tokens)
                if first_token_at is None and completion_chunk_has_generated_payload(chunk):
                    ttft = time.perf_counter() - start
                    first_token_at = start + ttft
        end = time.perf_counter()
        if first_token_at is None or ttft is None:
            raise ReplayError("no_generated_stream_payload")
        return {
            "ttft_s": ttft,
            "request_start_at": start,
            "first_token_at": first_token_at,
            "model_total_s": end - start,
            "decode_s": end - first_token_at,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens or len(token_ids),
            "completion_tokens": completion_tokens,
        }


def completion_chunk_has_generated_payload(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices") or []
    if not choices:
        return False
    choice = choices[0]
    if choice.get("text") not in (None, ""):
        return True
    return choice.get("finish_reason") is not None or choice.get("stop_reason") is not None


def cached_tokens_from_chunk(chunk: dict[str, Any], current: int | None) -> int | None:
    usage = chunk.get("usage") or {}
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return details.get("cached_tokens", current)


def prompt_tokens_from_chunk(chunk: dict[str, Any], current: int | None) -> int | None:
    usage = chunk.get("usage") or {}
    value = usage.get("prompt_tokens", usage.get("input_tokens"))
    return value if isinstance(value, int) else current


def completion_tokens_from_chunk(chunk: dict[str, Any], current: int | None) -> int | None:
    usage = chunk.get("usage") or {}
    value = usage.get("completion_tokens", usage.get("output_tokens"))
    return value if isinstance(value, int) else current


def served_model_name(model_name: str) -> str:
    return model_name.removeprefix("hosted_vllm/")
