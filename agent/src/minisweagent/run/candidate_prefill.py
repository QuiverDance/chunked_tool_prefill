"""Planning primitives for Candidate Tool Prefill."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HistoricalToolCall:
    command: str
    output: dict[str, Any]
    call_index: int

    @property
    def raw_output(self) -> str:
        return str(self.output.get("output") or "")


@dataclass(frozen=True)
class CandidateBranch:
    candidate_index: int
    token_ids: tuple[int, ...]
    cached_prefix_len: int


@dataclass(frozen=True)
class CandidatePrefillPlan:
    shared_prefix_len: int
    branches: tuple[CandidateBranch, ...]

    @classmethod
    def build(
        cls,
        cached_prompt_tokens: Sequence[int],
        candidate_prompt_tokens: Sequence[Sequence[int]],
        *,
        block_size: int,
    ) -> CandidatePrefillPlan:
        unique_candidates = unique_token_sequences(candidate_prompt_tokens)
        if not unique_candidates:
            return cls(shared_prefix_len=0, branches=())

        ordered_candidates = prefix_tree_order(unique_candidates)
        materialized = [tuple(cached_prompt_tokens)]
        branches = []
        for candidate_index, token_ids in ordered_candidates:
            cached_prefix_len = align_down(
                max(common_prefix_length(token_ids, prefix) for prefix in materialized),
                block_size,
            )
            branches.append(CandidateBranch(candidate_index, token_ids, cached_prefix_len))
            materialized.append(token_ids)

        shared_prefix_len = align_down(
            common_prefix_length_many([token_ids for _, token_ids in unique_candidates]),
            block_size,
        )
        return cls(shared_prefix_len=shared_prefix_len, branches=tuple(branches))


def select_similar_candidates(
    command: str,
    history: Sequence[HistoricalToolCall],
    *,
    top_k: int,
) -> list[HistoricalToolCall]:
    if top_k <= 0:
        return []

    current_terms = command_terms(command)
    ranked = sorted(
        history,
        key=lambda candidate: (
            jaccard_similarity(current_terms, command_terms(candidate.command)),
            candidate.call_index,
        ),
        reverse=True,
    )
    selected = []
    seen_outputs = set()
    for candidate in ranked:
        if candidate.raw_output in seen_outputs:
            continue
        selected.append(candidate)
        seen_outputs.add(candidate.raw_output)
        if len(selected) == top_k:
            break
    return selected


@dataclass
class _PrefixNode:
    children: dict[int, _PrefixNode] = field(default_factory=dict)
    candidate_index: int | None = None
    min_rank: int | None = None

    def add(self, candidate_index: int, token_ids: tuple[int, ...]) -> None:
        node = self
        node.remember_rank(candidate_index)
        for token_id in token_ids:
            node = node.children.setdefault(token_id, _PrefixNode())
            node.remember_rank(candidate_index)
        node.candidate_index = candidate_index

    def remember_rank(self, rank: int) -> None:
        self.min_rank = rank if self.min_rank is None else min(self.min_rank, rank)

    def candidate_order(self) -> list[int]:
        ordered = []
        pending = list(reversed(self.traversal_entries()))
        while pending:
            entry = pending.pop()
            if entry.candidate_index is not None:
                ordered.append(entry.candidate_index)
                continue
            if entry.node is not None:
                pending.extend(reversed(entry.node.traversal_entries()))
        return ordered

    def traversal_entries(self) -> list[_TraversalEntry]:
        entries = []
        if self.candidate_index is not None:
            entries.append(_TraversalEntry.for_candidate(self.candidate_index))
        entries.extend(_TraversalEntry.for_node(child) for child in self.children.values())
        return sorted(entries, key=lambda entry: entry.rank)

    def first_rank(self) -> int:
        if self.min_rank is None:
            raise RuntimeError("prefix node has no candidate rank")
        return self.min_rank


@dataclass(frozen=True)
class _TraversalEntry:
    rank: int
    candidate_index: int | None = None
    node: _PrefixNode | None = None

    @classmethod
    def for_candidate(cls, candidate_index: int) -> _TraversalEntry:
        return cls(rank=candidate_index, candidate_index=candidate_index)

    @classmethod
    def for_node(cls, node: _PrefixNode) -> _TraversalEntry:
        return cls(rank=node.first_rank(), node=node)


def unique_token_sequences(candidate_prompt_tokens: Sequence[Sequence[int]]) -> list[tuple[int, tuple[int, ...]]]:
    unique = []
    seen = set()
    for candidate_index, candidate in enumerate(candidate_prompt_tokens):
        token_ids = tuple(candidate)
        if token_ids in seen:
            continue
        seen.add(token_ids)
        unique.append((candidate_index, token_ids))
    return unique


def prefix_tree_order(candidates: list[tuple[int, tuple[int, ...]]]) -> list[tuple[int, tuple[int, ...]]]:
    root = _PrefixNode()
    by_index = dict(candidates)
    for candidate_index, token_ids in candidates:
        root.add(candidate_index, token_ids)
    return [(candidate_index, by_index[candidate_index]) for candidate_index in root.candidate_order()]


def common_prefix_length(first: Sequence[int], second: Sequence[int]) -> int:
    for index, (first_token, second_token) in enumerate(zip(first, second)):
        if first_token != second_token:
            return index
    return min(len(first), len(second))


def common_prefix_length_many(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    return (
        min(common_prefix_length(sequences[0], sequence) for sequence in sequences[1:])
        if len(sequences) > 1
        else len(sequences[0])
    )


def align_down(value: int, block_size: int) -> int:
    if block_size <= 1:
        return max(0, value)
    return max(0, value) // block_size * block_size


def command_terms(command: str) -> frozenset[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    ignored = {"&&", "||", ";", "|", "<", ">", ">>", "2>", "2>&1"}
    return frozenset(token for token in tokens[:256] if token not in ignored)


def jaccard_similarity(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0
