import pytest

from biomrag.schemas import EvidenceItem, QueryExample, UtilityScore
from biomrag.selection.streaming_selector import (
    NEUTRAL_UTILITY,
    streaming_proxy_deliberate,
    streaming_proxy_select,
)


class ToyScorer:
    """Scores above the neutral point mark a net beneficial proposal."""

    def score(self, query, evidence):
        scores = {
            ("a",): 6,
            ("a", "b"): 7,
            ("b", "c"): 9,
        }
        key = tuple(sorted(item.id for item in evidence))
        return UtilityScore(
            score=scores.get(key, 4),
            rationale=f"Utility for {key}",
        )


def test_streaming_selector_replaces_redundant_item_when_subset_score_improves():
    query = QueryExample(id="q1", question="lung cavity")
    candidates = [
        EvidenceItem(id="a", text="lung cavity"),
        EvidenceItem(id="b", text="weak distractor"),
        EvidenceItem(id="c", text="strong evidence"),
    ]
    embeddings = {
        "a": [1.0, 0.0],
        "b": [0.9, 0.1],
        "c": [0.0, 1.0],
    }

    selected = streaming_proxy_select(
        query,
        candidates,
        ToyScorer(),
        budget=2,
        embedding_lookup=lambda item: embeddings[item.id],
    )

    assert [item.id for item in selected] == ["b", "c"]


def test_streaming_selector_uses_exactly_one_call_per_candidate_and_keeps_ties():
    class TieScorer:
        def score(self, query, evidence):
            return UtilityScore(score=5, rationale="Equal utility.")

    query = QueryExample(id="q1", question="finding")
    candidates = [
        EvidenceItem(id="a", retrieval_rank=1),
        EvidenceItem(id="b", retrieval_rank=2),
        EvidenceItem(id="c", retrieval_rank=3),
    ]
    result = streaming_proxy_deliberate(
        query,
        candidates,
        TieScorer(),
        budget=2,
        embedding_lookup=lambda item: {
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
        }[item.id],
    )

    assert len(result.steps) == len(candidates)
    assert [item.id for item in result.selected] == []
    assert [step.accepted for step in result.steps] == [False, False, False]
    assert result.score == NEUTRAL_UTILITY


def test_streaming_selector_returns_empty_subset_when_no_candidate_is_beneficial():
    class HarmfulScorer:
        def score(self, query, evidence):
            return UtilityScore(score=2, rationale="Distracting evidence.")

    query = QueryExample(id="q1", question="finding")
    candidates = [
        EvidenceItem(id="a", retrieval_rank=1),
        EvidenceItem(id="b", retrieval_rank=2),
    ]
    result = streaming_proxy_deliberate(
        query,
        candidates,
        HarmfulScorer(),
        budget=5,
    )

    assert result.selected == []
    assert result.score == NEUTRAL_UTILITY
    assert len(result.steps) == len(candidates)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (5, []),
        (6, ["a"]),
    ],
)
def test_first_candidate_enters_only_when_net_beneficial(score, expected):
    class FixedScorer:
        def score(self, query, evidence):
            return UtilityScore(score=score, rationale=f"Utility {score}.")

    query = QueryExample(id="q1", question="finding")
    candidates = [EvidenceItem(id="a", retrieval_rank=1)]
    result = streaming_proxy_deliberate(
        query,
        candidates,
        FixedScorer(),
        budget=5,
    )

    assert [item.id for item in result.selected] == expected


def test_streaming_selector_rejects_noncontiguous_retrieval_order():
    candidates = [
        EvidenceItem(id="a", retrieval_rank=1),
        EvidenceItem(id="b", retrieval_rank=3),
    ]
    with pytest.raises(ValueError, match="retrieval-rank order"):
        streaming_proxy_deliberate(
            QueryExample(id="q", question="q"),
            candidates,
            ToyScorer(),
            budget=2,
        )
