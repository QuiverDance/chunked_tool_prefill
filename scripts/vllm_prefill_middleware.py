"""Add replay-only prefill and abort endpoints to a vLLM API server."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

active_prefills: set[str] = set()
aborted_prefills: set[str] = set()
prefill_finished: dict[str, asyncio.Event] = {}


async def prefill_middleware(
    request: Any,
    call_next: Callable[[Any], Awaitable[Any]],
) -> Any:
    if request.method == "POST" and request.url.path == "/v1/prefill":
        return await _prefill(request)
    if request.method == "POST" and request.url.path == "/v1/prefill/abort":
        return await _abort(request)
    if request.method == "POST" and request.url.path == "/v1/prefill/reset":
        return await _reset_prefix_cache(request)
    return await call_next(request)


async def _prefill(request: Any) -> Any:
    from starlette.responses import JSONResponse
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    payload = await _json_payload(request)
    if isinstance(payload, JSONResponse):
        return payload

    token_ids = payload.get("input_token_ids")
    valid_token_ids = (
        isinstance(token_ids, list)
        and bool(token_ids)
        and all(type(token_id) is int for token_id in token_ids)
    )
    if not valid_token_ids:
        return JSONResponse(
            {"error": "input_token_ids must be a non-empty list of integers"},
            status_code=400,
        )

    request_id = str(payload.get("request_id") or f"prefill-{uuid.uuid4().hex}")
    prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
    if cache_salt := payload.get("cache_salt"):
        prompt["cache_salt"] = str(cache_salt)

    engine = request.app.state.engine_client
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    if request_id in active_prefills:
        return JSONResponse({"error": f"prefill request is already active: {request_id}"}, status_code=409)

    finished = asyncio.Event()
    active_prefills.add(request_id)
    prefill_finished[request_id] = finished
    try:
        try:
            async for _ in engine.generate(prompt, sampling, request_id):
                pass
        except Exception as error:
            return JSONResponse(
                {"error": f"prefill failed: {type(error).__name__}: {error}"},
                status_code=500,
            )

        status = "aborted" if request_id in aborted_prefills else "completed"
        return JSONResponse(
            {
                "request_id": request_id,
                "prompt_tokens": len(token_ids),
                "status": status,
            }
        )
    finally:
        active_prefills.discard(request_id)
        aborted_prefills.discard(request_id)
        prefill_finished.pop(request_id, None)
        finished.set()


async def _abort(request: Any) -> Any:
    from starlette.responses import JSONResponse

    payload = await _json_payload(request)
    if isinstance(payload, JSONResponse):
        return payload

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return JSONResponse({"error": "request_id must be a non-empty string"}, status_code=400)

    finished = prefill_finished.get(request_id)
    try:
        if request_id in active_prefills:
            aborted_prefills.add(request_id)
        await request.app.state.engine_client.abort(request_id)
        if finished is not None:
            await asyncio.wait_for(finished.wait(), timeout=5)
    except TimeoutError:
        return JSONResponse(
            {"error": f"prefill did not stop after abort: {request_id}"},
            status_code=504,
        )
    except Exception as error:
        return JSONResponse(
            {"error": f"prefill abort failed: {type(error).__name__}: {error}"},
            status_code=500,
        )

    return JSONResponse({"request_id": request_id, "status": "aborted"})


async def _reset_prefix_cache(request: Any) -> Any:
    from starlette.responses import JSONResponse

    if active_prefills:
        return JSONResponse(
            {"error": f"cannot reset prefix cache with {len(active_prefills)} active prefill request(s)"},
            status_code=409,
        )

    try:
        reset = await request.app.state.engine_client.reset_prefix_cache()
    except Exception as error:
        return JSONResponse(
            {"error": f"prefix cache reset failed: {type(error).__name__}: {error}"},
            status_code=500,
        )
    if not reset:
        return JSONResponse(
            {"error": "prefix cache reset was rejected because requests are still using KV cache"},
            status_code=409,
        )
    return JSONResponse({"status": "reset"})


async def _json_payload(request: Any) -> dict[str, Any] | Any:
    from starlette.responses import JSONResponse

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "request body must be JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
    return payload
