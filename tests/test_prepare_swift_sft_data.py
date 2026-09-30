import pytest

from biomrag.proxy.prepare_swift_sft_data import split_rows, to_swift_row


def test_to_swift_row_preserves_exact_image_order_and_loss_flag():
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "<image>\nquestion\n<image>\nevidence",
            },
            {"role": "assistant", "content": "Rationale: useful\nScore: 8"},
        ],
        "images": ["query.png", "evidence.png"],
        "score": 8,
    }

    converted = to_swift_row(row)

    assert converted["messages"][1]["content"].count("<image>") == 2
    assert converted["messages"][-1]["loss"] is True
    assert converted["images"] == ["query.png", "evidence.png"]


def test_to_swift_row_rejects_image_token_mismatch():
    row = {
        "messages": [
            {"role": "user", "content": "<image>\nquestion"},
            {"role": "assistant", "content": "Rationale: useful\nScore: 8"},
        ],
        "images": ["query.png", "evidence.png"],
    }
    with pytest.raises(ValueError, match="must match exactly"):
        to_swift_row(row)


def test_proxy_validation_split_keeps_parent_queries_disjoint():
    rows = [
        {"parent_id": parent, "subset_id": f"{parent}:{index}"}
        for parent in ("q1", "q2", "q3", "q4")
        for index in range(3)
    ]
    train, validation = split_rows(rows, val_ratio=0.25, seed=42)

    train_parents = {row["parent_id"] for row in train}
    validation_parents = {row["parent_id"] for row in validation}
    assert train_parents.isdisjoint(validation_parents)
    assert train and validation
