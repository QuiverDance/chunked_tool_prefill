from minisweagent.run.candidate_prefill import (
    CandidatePrefillPlan,
    HistoricalToolCall,
    select_similar_candidates,
)


def test_candidate_prefill_plan_preserves_rank_while_finishing_shared_subtrees() -> None:
    cached_prompt = [1, 2]
    candidates = [
        [1, 2, 3, 4, 5, 6, 9],
        [1, 2, 3, 4, 8, 7],
        [1, 2, 3, 4, 5, 6, 10],
        [1, 2, 3, 4, 5, 6, 9],
        [1, 2, 3, 4, 6, 11],
    ]

    plan = CandidatePrefillPlan.build(cached_prompt, candidates, block_size=2)

    assert plan.shared_prefix_len == 4
    assert [branch.candidate_index for branch in plan.branches] == [0, 2, 1, 4]
    assert [branch.cached_prefix_len for branch in plan.branches] == [2, 6, 4, 4]
    assert [list(branch.token_ids) for branch in plan.branches] == [
        candidates[0],
        candidates[2],
        candidates[1],
        candidates[4],
    ]


def test_candidate_selection_prefers_similarity_then_recency_and_distinct_outputs() -> None:
    history = [
        HistoricalToolCall("pytest tests/a.py -q", {"output": "same"}, 0),
        HistoricalToolCall("cat README.md", {"output": "unrelated"}, 1),
        HistoricalToolCall("pytest tests/b.py -q", {"output": "same"}, 2),
        HistoricalToolCall("pytest tests/c.py -q", {"output": "recent"}, 3),
    ]

    selected = select_similar_candidates("pytest tests/d.py -q", history, top_k=3)

    assert [candidate.raw_output for candidate in selected] == ["recent", "same", "unrelated"]


def test_candidate_selection_returns_nothing_when_top_k_is_zero() -> None:
    history = [HistoricalToolCall("pytest", {"output": "passed"}, 0)]

    assert select_similar_candidates("pytest", history, top_k=0) == []
