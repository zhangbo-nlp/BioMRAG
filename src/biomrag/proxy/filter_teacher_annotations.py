"""Filter malformed teacher utility annotations before proxy SFT."""

from __future__ import annotations

import argparse
from typing import Any

from biomrag.schemas import QueryExample
from biomrag.utils.jsonl import iter_jsonl, write_json, write_jsonl

CHANGE_TERMS = {
    "no-evidence",
    "evidence-conditioned",
    "improve",
    "improves",
    "improved",
    "degrade",
    "degrades",
    "worse",
    "better",
    "support",
    "supports",
    "change",
    "changes",
    "changed",
    "relative",
    "conditioned",
    "without",
    "adds",
    "misleading",
    "distracting",
    "redundant",
    "complementary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default="")
    parser.add_argument("--require-change-terms", action="store_true")
    parser.add_argument("--max-reference-overlap", type=float, default=0.9)
    parser.add_argument("--expected-input-count", type=int, default=0)
    parser.add_argument("--expected-output-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept_count = 0
    input_count = 0
    removed = {
        "missing_teacher_utility": 0,
        "invalid_score": 0,
        "empty_rationale": 0,
        "no_change_terms": 0,
        "reference_restatement": 0,
    }

    def kept_rows():
        nonlocal input_count, kept_count
        for row in iter_jsonl(args.input):
            input_count += 1
            reason = rejection_reason(
                row,
                require_change_terms=args.require_change_terms,
                max_reference_overlap=args.max_reference_overlap,
            )
            if reason:
                removed[reason] += 1
                continue
            kept_count += 1
            yield row

    write_jsonl(args.output, kept_rows())
    if args.expected_input_count and input_count != args.expected_input_count:
        raise ValueError(
            f"Read {input_count} teacher annotations; "
            f"expected {args.expected_input_count}."
        )
    if args.expected_output_count and kept_count != args.expected_output_count:
        raise ValueError(
            f"Kept {kept_count} teacher annotations; "
            f"expected {args.expected_output_count}."
        )
    if args.audit:
        write_json(
            args.audit,
            {
                "input": input_count,
                "kept": kept_count,
                "removed": removed,
            },
        )


def rejection_reason(
    row: dict[str, Any],
    *,
    require_change_terms: bool,
    max_reference_overlap: float = 0.9,
) -> str:
    utility = row.get("teacher_utility")
    if not isinstance(utility, dict):
        return "missing_teacher_utility"
    try:
        score_value = float(utility.get("score"))
    except (TypeError, ValueError):
        return "invalid_score"
    if not score_value.is_integer() or not 1 <= score_value <= 10:
        return "invalid_score"
    rationale = str(utility.get("rationale", "")).strip()
    if not rationale:
        return "empty_rationale"
    if require_change_terms:
        lowered = rationale.lower()
        if not any(term in lowered for term in CHANGE_TERMS):
            return "no_change_terms"
    reference = str(
        row.get("answer")
        or row.get("reference")
        or row.get("reference_answer")
        or QueryExample.from_dict(row).answer
        or ""
    ).strip()
    if reference and _token_overlap(rationale, reference) >= max_reference_overlap:
        return "reference_restatement"
    return ""


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


if __name__ == "__main__":
    main()
