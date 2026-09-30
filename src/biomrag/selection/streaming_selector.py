"""Utility-aware streaming evidence selection.

The selector mirrors Algorithm 1 of the paper. The empty subset is never
scored, because the proxy has no proposal to judge; the running score starts
at the neutral point of the ordinal 1-10 utility scale instead. The first
candidate therefore enters the subset only when its proposal is already net
beneficial, and every later candidate is accepted only when the complete
proposal it forms outscores the best proposal seen so far. An empty subset is
a legal outcome, meaning the generator answers from parametric knowledge.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from biomrag.schemas import EvidenceItem, QueryExample, UtilityScore

NEUTRAL_UTILITY = 5
"""Neutral point of the ordinal 1-10 utility scale (see the proxy prompt).

Scores of 4-5 mean little or no meaningful improvement, so a proposal must
score above this constant before any evidence is retained.
"""


class ProxyScorer(Protocol):
    """Interface for a proxy reasoner that scores evidence subsets."""

    def score(self, query: QueryExample, evidence: Sequence[EvidenceItem]) -> UtilityScore:
        """Return a utility score for a non-empty candidate evidence subset."""


EmbeddingLookup = Callable[[EvidenceItem], Sequence[float] | None]


@dataclass(slots=True)
class SelectionStep:
    candidate_id: str
    proposal_ids: list[str]
    evicted_id: str | None
    score: int
    rationale: str
    accepted: bool


@dataclass(slots=True)
class SelectionResult:
    selected: list[EvidenceItem]
    score: int
    rationale: str
    steps: list[SelectionStep]

    def trace_dicts(self) -> list[dict]:
        return [asdict(step) for step in self.steps]


def streaming_proxy_select(
    query: QueryExample,
    candidates: Sequence[EvidenceItem],
    scorer: ProxyScorer,
    *,
    budget: int = 5,
    embedding_lookup: EmbeddingLookup | None = None,
) -> list[EvidenceItem]:
    """Select a compact evidence subset with streaming proxy replacement.

    Args:
        query: Biomedical query.
        candidates: Retrieved candidates ordered by retrieval score.
        scorer: Proxy that scores subset answer utility.
        budget: Maximum selected evidence items.
        embedding_lookup: Lookup for normalized retriever embeddings.

    Returns:
        Selected evidence items in retained order, possibly empty.
    """
    return streaming_proxy_deliberate(
        query,
        candidates,
        scorer,
        budget=budget,
        embedding_lookup=embedding_lookup,
    ).selected


def streaming_proxy_deliberate(
    query: QueryExample,
    candidates: Sequence[EvidenceItem],
    scorer: ProxyScorer,
    *,
    budget: int = 5,
    embedding_lookup: EmbeddingLookup | None = None,
) -> SelectionResult:
    """Run BioMRAG streaming selection and retain each proposal rationale.

    Every candidate receives exactly one proxy call, so a query with ``M``
    candidates performs ``M`` evaluations. The empty subset is never scored;
    the running score starts at the neutral point of the 1-10 ordinal scale,
    and the returned subset stays empty when no proposal outscores it.
    """
    if not candidates:
        return SelectionResult([], NEUTRAL_UTILITY, "", [])
    if budget < 2:
        raise ValueError("BioMRAG selection requires an evidence budget of at least 2.")
    candidate_ids = [item.id for item in candidates]
    if any(not item_id for item_id in candidate_ids):
        raise ValueError("Every candidate must have a non-empty evidence ID.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate evidence IDs must be unique within each query.")
    _validate_candidate_order(candidates)
    if len(candidates) > budget and embedding_lookup is None:
        raise ValueError(
            "Retriever embeddings are required for redundancy-guided replacement."
        )

    selected: list[EvidenceItem] = []
    current_score = NEUTRAL_UTILITY
    current_rationale = ""
    steps: list[SelectionStep] = []

    for candidate in candidates:
        evicted_id = None
        if len(selected) < budget:
            proposal = [*selected, candidate]
        else:
            replacement = _most_redundant_item(
                selected,
                embedding_lookup=embedding_lookup,
            )
            evicted_id = replacement.id
            replacement_index = next(
                index for index, item in enumerate(selected) if item is replacement
            )
            proposal = [
                item
                for index, item in enumerate(selected)
                if index != replacement_index
            ]
            proposal.append(candidate)

        proposal_score = scorer.score(query, proposal)
        numeric_proposal_score = _ordinal_score(proposal_score)
        accepted = numeric_proposal_score > current_score
        steps.append(
            SelectionStep(
                candidate_id=candidate.id,
                proposal_ids=[item.id for item in proposal],
                evicted_id=evicted_id,
                score=numeric_proposal_score,
                rationale=proposal_score.rationale,
                accepted=accepted,
            )
        )
        if accepted:
            selected = proposal
            current_score = numeric_proposal_score
            current_rationale = proposal_score.rationale

    return SelectionResult(
        selected=selected,
        score=current_score,
        rationale=current_rationale,
        steps=steps,
    )


def _most_redundant_item(
    selected: Sequence[EvidenceItem],
    *,
    embedding_lookup: EmbeddingLookup | None,
) -> EvidenceItem:
    if len(selected) == 1:
        return selected[0]
    if embedding_lookup is None:
        raise ValueError("Retriever embeddings are required for redundancy scoring.")

    return max(
        selected,
        key=lambda item: internal_redundancy(
            item, selected, embedding_lookup=embedding_lookup
        ),
    )


def internal_redundancy(
    item: EvidenceItem,
    selected: Sequence[EvidenceItem],
    *,
    embedding_lookup: EmbeddingLookup,
) -> float:
    sims = [
        cosine_positive_similarity(
            _required_embedding(item, embedding_lookup),
            _required_embedding(other, embedding_lookup),
        )
        for other in selected
        if other.id != item.id
    ]
    return float(sum(sims) / len(sims)) if sims else 0.0


def _required_embedding(
    item: EvidenceItem,
    embedding_lookup: EmbeddingLookup,
) -> Sequence[float]:
    embedding = embedding_lookup(item)
    if embedding is None:
        raise ValueError(f"Missing retriever embedding for evidence ID {item.id!r}.")
    return embedding


def cosine_positive_similarity(
    a: Sequence[float], b: Sequence[float]
) -> float:
    """Cosine similarity clipped at zero for redundancy accounting."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Retriever embeddings must have the same non-zero dimension.")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    a_norm = math.sqrt(sum(float(x) * float(x) for x in a))
    b_norm = math.sqrt(sum(float(y) * float(y) for y in b))
    denom = a_norm * b_norm
    if denom == 0.0:
        raise ValueError("Retriever embeddings must have non-zero norm.")
    return max(0.0, dot / denom)


def _ordinal_score(result: UtilityScore) -> int:
    if isinstance(result.score, bool):
        raise ValueError("Proxy utility must be an integer from 1 to 10.")
    numeric = float(result.score)
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 10:
        raise ValueError(
            f"Proxy utility must be an integer from 1 to 10; got {result.score!r}."
        )
    if not str(result.rationale).strip():
        raise ValueError("Proxy utility must include a non-empty rationale.")
    return int(numeric)


def _validate_candidate_order(candidates: Sequence[EvidenceItem]) -> None:
    ranks = [item.retrieval_rank for item in candidates]
    if all(rank is None for rank in ranks):
        return
    if any(rank is None for rank in ranks):
        raise ValueError("Candidate retrieval ranks must be either all present or absent.")
    numeric = [int(rank) for rank in ranks]
    if numeric != list(range(1, len(candidates) + 1)):
        raise ValueError(
            "Candidates must be supplied in contiguous retrieval-rank order."
        )
