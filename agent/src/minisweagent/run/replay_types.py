"""Shared data shapes for trace replay."""

from __future__ import annotations

from dataclasses import dataclass


class ReplayError(RuntimeError):
    """Raised when a replay trial cannot be measured."""


@dataclass(frozen=True)
class ReplayStep:
    instance_id: str
    step_index: int


@dataclass
class AsyncPrefillRequest:
    token_ids: list[int]
    cache_salt: str
    step: ReplayStep
    label: str
    request_id: str


@dataclass(frozen=True)
class AsyncPrefillCompletion:
    request_id: str
    label: str
    prefix_len: int
    finished_at: float


@dataclass(frozen=True)
class PromptTokenState:
    text: str
    token_ids: list[int]
