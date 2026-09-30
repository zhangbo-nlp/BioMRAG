import pytest

from biomrag.proxy.build_distillation_data import build_proxy_sft_row
from biomrag.proxy.filter_teacher_annotations import rejection_reason
from biomrag.proxy.sample_candidate_subsets import sample_subsets
from biomrag.proxy.scoring import parse_proxy_response
from biomrag.proxy.select_release_subsets import select_rows


def _annotation(parent_id, subset_type, index, score):
    return {
        "parent_id": parent_id,
        "subset_id": f"{parent_id}:{subset_type}:{index}",
        "subset_type": subset_type,
        "teacher_utility": {"score": score, "rationale": "Response change noted."},
    }


def test_release_selection_keeps_prefixes_and_prefers_rare_scores():
    # Two queries: each has five prefixes (common score 6) and ten five-item
    # subsets, of which two carry the globally rare score 9.
    rows = []
    for parent in ("q1", "q2"):
        rows.extend(_annotation(parent, "prefix", i, 6) for i in range(5))
        rows.extend(_annotation(parent, "random", i, 9) for i in range(2))
        rows.extend(_annotation(parent, "random", i + 2, 6) for i in range(8))

    selected = select_rows(rows, keep_per_query=12)

    assert len(selected) == 24
    per_query = {}
    for row in selected:
        per_query.setdefault(row["parent_id"], []).append(row)
    for kept in per_query.values():
        assert len(kept) == 12
        assert sum(row["subset_type"] == "prefix" for row in kept) == 5
        assert sum(row["subset_type"] == "random" for row in kept) == 7
        # The rare score 9 subsets survive, and only five score-6 randoms remain.
        assert sum(row["teacher_utility"]["score"] == 9 for row in kept) == 2
        assert sum(
            row["teacher_utility"]["score"] == 6 and row["subset_type"] == "random"
            for row in kept
        ) == 5


def test_release_selection_rejects_queries_with_too_few_subsets():
    rows = [_annotation("q1", "prefix", i, 6) for i in range(5)]
    with pytest.raises(ValueError, match="release requires"):
        select_rows(rows, keep_per_query=12)


def test_sample_subsets_builds_prefix_and_random_rows():
    row = {
        "id": "q1",
        "question": "finding?",
        "retrieved_docs": [
            {"id": f"k{i}", "text": "evidence", "retrieval_rank": i}
            for i in range(1, 8)
        ],
    }

    rows = sample_subsets(
        row,
        top_k=7,
        prefix_sizes=[1, 2],
        random_subsets=2,
        random_min_size=1,
        random_max_size=3,
    )

    assert len(rows) == 4
    assert rows[0]["subset_type"] == "prefix"
    assert rows[0]["parent_id"] == "q1"
    assert rows[0]["retrieved_docs"] == rows[0]["selected_docs"]
    assert sum(row["subset_type"] == "random" for row in rows) == 2


def test_filter_teacher_annotations_rejects_bad_rows():
    good = {"teacher_utility": {"score": 8, "rationale": "Evidence improves the answer."}}
    bad = {"teacher_utility": {"score": 11, "rationale": "too high"}}

    assert rejection_reason(good, require_change_terms=True) == ""
    assert rejection_reason(bad, require_change_terms=False) == "invalid_score"


def test_subset_sampling_builds_five_prefixes_and_ten_five_item_subsets():
    row = {
        "id": "q1",
        "retrieved_docs": [
            {"id": f"k{i}", "text": f"evidence {i}", "retrieval_rank": i}
            for i in range(1, 31)
        ],
    }
    rows = sample_subsets(row)

    assert len(rows) == 15
    assert [item["subset_type"] for item in rows[:5]] == ["prefix"] * 5
    assert [len(item["retrieved_docs"]) for item in rows[:5]] == [1, 2, 3, 4, 5]
    assert [item["subset_type"] for item in rows[5:]] == ["random"] * 10
    assert {len(item["retrieved_docs"]) for item in rows[5:]} == {5}


def test_subset_sampling_rejects_short_candidate_pool():
    with pytest.raises(ValueError, match="requires 30"):
        sample_subsets(
            {"id": "q", "retrieved_docs": [{"id": "k1"}]},
        )


def test_proxy_parser_requires_integer_score_and_nonempty_rationale():
    parsed = parse_proxy_response("Rationale: useful\nScore: 8")
    assert parsed.score == 8
    with pytest.raises(ValueError):
        parse_proxy_response('{"rationale": "useful", "score": 8.5}')
    with pytest.raises(ValueError, match="Malformed"):
        parse_proxy_response("Score: 7")
    with pytest.raises(ValueError, match="empty rationale"):
        parse_proxy_response('{"rationale": "", "score": 7}')


def test_distillation_builder_emits_rationale_and_score_without_reference_inputs():
    response = {
        "id": "q1",
        "question": "What is shown?",
        "image_path": "query.png",
        "selected_docs": [
            {
                "id": "k1",
                "text": "supporting fact",
                "image_path": "evidence.png",
                "modality": "image-text",
            }
        ],
    }
    rating = {"score": 8, "rationale": "The evidence improves correctness."}
    output = build_proxy_sft_row(response, rating)

    assert output["messages"][-1]["content"] == (
        "Rationale: The evidence improves correctness.\nScore: 8"
    )
    assert output["images"] == ["query.png", "evidence.png"]
    user_prompt = output["messages"][1]["content"]
    assert user_prompt.count("<image>") == 2
    assert "reference" not in user_prompt.lower()
