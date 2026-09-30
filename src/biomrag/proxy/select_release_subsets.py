"""Select the released subset of annotated proxy-distillation subsets.

Distillation annotates fifteen subsets per query, namely five top-ranked
prefixes of sizes 1 to 5 and ten five-item subsets. The released corpus keeps
twelve of the fifteen per query: all five prefixes, which give systematic
coverage of subset sizes, and seven of the ten five-item subsets, preferring
the globally rarest utility scores so that scarce low- and high-utility
supervision survives the reduction. The rule is deterministic, so the release
is reproducible from the annotated pool alone.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from biomrag.utils.jsonl import iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Filtered annotation JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default="")
    parser.add_argument("--keep-per-query", type=int, default=12)
    parser.add_argument("--expected-input-count", type=int, default=0)
    parser.add_argument("--expected-output-count", type=int, default=0)
    return parser.parse_args()


def utility_score(row: dict[str, Any]) -> int | None:
    """Return the ordinal teacher utility score of one annotated row."""
    utility = row.get("teacher_utility")
    if not isinstance(utility, dict):
        return None
    try:
        value = float(utility.get("score"))
    except (TypeError, ValueError):
        return None
    if not value.is_integer() or not 1 <= value <= 10:
        return None
    return int(value)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    keep_per_query: int = 12,
) -> list[dict[str, Any]]:
    """Return the release subset of annotated rows, in original order."""
    if keep_per_query < 1:
        raise ValueError("--keep-per-query must be at least 1.")
    scores = [utility_score(row) for row in rows]
    counts = Counter(score for score in scores if score is not None)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("parent_id", ""))].append(index)

    kept_indices: list[int] = []
    for parent_id, indices in groups.items():
        if len(indices) < keep_per_query:
            raise ValueError(
                f"Query {parent_id!r} has {len(indices)} annotated subsets; "
                f"the release requires {keep_per_query}."
            )
        prefixes = [index for index in indices if rows[index].get("subset_type") == "prefix"]
        tail = [index for index in indices if rows[index].get("subset_type") != "prefix"]
        # Prefer the globally rarest utility scores, then the original order.
        tail.sort(key=lambda index: (counts.get(scores[index], 0), index))
        kept = prefixes[:keep_per_query]
        remaining = keep_per_query - len(kept)
        if remaining > 0:
            kept.extend(tail[:remaining])
        kept_indices.extend(kept)

    kept_indices.sort()
    return [rows[index] for index in kept_indices]


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.input))
    selected = select_rows(rows, keep_per_query=args.keep_per_query)

    write_jsonl(args.output, selected)
    if args.expected_input_count and len(rows) != args.expected_input_count:
        raise ValueError(
            f"Read {len(rows)} teacher annotations; "
            f"expected {args.expected_input_count}."
        )
    if args.expected_output_count and len(selected) != args.expected_output_count:
        raise ValueError(
            f"Selected {len(selected)} release annotations; "
            f"expected {args.expected_output_count}."
        )
    if args.audit:
        kept_scores = Counter(utility_score(row) for row in selected)
        write_json(
            args.audit,
            {
                "input": len(rows),
                "selected": len(selected),
                "keep_per_query": args.keep_per_query,
                "queries": len({str(row.get('parent_id', '')) for row in rows}),
                "score_distribution_source": {
                    str(score): count
                    for score, count in sorted(
                        Counter(
                            score for score in (utility_score(row) for row in rows)
                            if score is not None
                        ).items()
                    )
                },
                "score_distribution_release": {
                    str(score): count
                    for score, count in sorted(kept_scores.items())
                    if score is not None
                },
            },
        )


if __name__ == "__main__":
    main()
