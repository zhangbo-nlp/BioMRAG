"""Sample complete evidence subsets for proxy distillation."""

from __future__ import annotations

import argparse
import random
from typing import Any

from biomrag.utils.jsonl import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Retrieved-candidate JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--prefix-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--random-subsets", type=int, default=10)
    parser.add_argument("--random-min-size", type=int, default=5)
    parser.add_argument("--random-max-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-query-count", type=int, default=0)
    parser.add_argument("--expected-output-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    query_count = 0
    output_count = 0
    seen_parent_ids: set[str] = set()

    def sampled_rows():
        nonlocal query_count, output_count
        for row in iter_jsonl(args.input):
            parent_id = str(row.get("id", ""))
            if not parent_id or parent_id in seen_parent_ids:
                raise ValueError(
                    "Proxy-distillation query IDs must be non-empty and unique; "
                    f"got {parent_id!r}."
                )
            seen_parent_ids.add(parent_id)
            query_count += 1
            for subset in sample_subsets(
                row,
                top_k=args.top_k,
                prefix_sizes=args.prefix_sizes,
                random_subsets=args.random_subsets,
                random_min_size=args.random_min_size,
                random_max_size=args.random_max_size,
                rng=rng,
            ):
                output_count += 1
                yield subset

    write_jsonl(args.output, sampled_rows())
    if args.expected_query_count and query_count != args.expected_query_count:
        raise ValueError(
            f"Read {query_count} proxy-distillation queries; "
            f"expected {args.expected_query_count}."
        )
    if args.expected_output_count and output_count != args.expected_output_count:
        raise ValueError(
            f"Created {output_count} subset requests; "
            f"expected {args.expected_output_count}."
        )


def sample_subsets(
    row: dict[str, Any],
    *,
    top_k: int = 30,
    prefix_sizes: list[int] | None = None,
    random_subsets: int = 10,
    random_min_size: int = 5,
    random_max_size: int = 5,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    prefix_sizes = prefix_sizes or [1, 2, 3, 4, 5]
    rng = rng or random.Random(42)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if random_subsets < 0:
        raise ValueError("random_subsets cannot be negative.")
    if random_min_size < 1 or random_max_size < random_min_size:
        raise ValueError(
            "random subset sizes must satisfy 1 <= min_size <= max_size."
        )
    if len(set(prefix_sizes)) != len(prefix_sizes) or any(
        size < 1 for size in prefix_sizes
    ):
        raise ValueError("prefix_sizes must contain unique positive integers.")
    parent_id = str(row.get("id", ""))
    if not parent_id:
        raise ValueError("Each proxy-distillation query requires a non-empty id.")
    retrieved = row.get("retrieved_docs")
    if not isinstance(retrieved, list):
        raise ValueError(f"Query {parent_id!r} requires a retrieved_docs list.")
    if len(retrieved) < top_k:
        raise ValueError(
            f"Query {parent_id!r} has {len(retrieved)} candidates; "
            f"proxy distillation requires {top_k}."
        )
    candidates = list(retrieved[:top_k])
    candidate_ids = [str(item.get("id", "")) for item in candidates]
    if any(not item_id for item_id in candidate_ids):
        raise ValueError(f"Query {parent_id!r} has evidence without an id.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"Query {parent_id!r} has duplicate evidence IDs.")
    candidate_ranks = [item.get("retrieval_rank") for item in candidates]
    if any(rank is None for rank in candidate_ranks):
        raise ValueError(
            f"Query {parent_id!r} requires contiguous retrieval ranks 1..{top_k}."
        )
    try:
        numeric_ranks = [int(rank) for rank in candidate_ranks]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Query {parent_id!r} has a non-integer retrieval rank."
        ) from exc
    if numeric_ranks != list(range(1, top_k + 1)):
        raise ValueError(
            f"Query {parent_id!r} candidates are not ordered by ranks 1..{top_k}."
        )
    outputs = []
    seen: set[tuple[str, ...]] = set()

    for size in prefix_sizes:
        if size > len(candidates):
            raise ValueError(
                f"Prefix size {size} exceeds the {len(candidates)}-item pool."
            )
        subset = candidates[:size]
        outputs.append(
            _make_subset_row(
                row,
                subset,
                parent_id=parent_id,
                subset_type="prefix",
                subset_index=len(outputs),
            )
        )
        seen.add(_subset_key(subset))

    min_size = random_min_size
    max_size = min(random_max_size, len(candidates))
    if min_size > max_size:
        raise ValueError(
            f"Random subset minimum {min_size} exceeds pool size {len(candidates)}."
        )
    attempts = 0
    random_added = 0
    while random_added < random_subsets and attempts < max(random_subsets * 20, 1):
        attempts += 1
        size = rng.randint(min_size, max_size)
        subset = rng.sample(candidates, size)
        subset.sort(key=lambda item: int(item.get("retrieval_rank") or top_k + 1))
        key = _subset_key(subset)
        if key in seen:
            continue
        outputs.append(
            _make_subset_row(
                row,
                subset,
                parent_id=parent_id,
                subset_type="random",
                subset_index=len(outputs),
            )
        )
        seen.add(key)
        random_added += 1

    if random_added != random_subsets:
        raise RuntimeError(
            f"Query {parent_id!r} produced {random_added} unique random subsets; "
            f"expected {random_subsets}."
        )
    return outputs


def _make_subset_row(
    row: dict[str, Any],
    subset: list[dict[str, Any]],
    *,
    parent_id: str,
    subset_type: str,
    subset_index: int,
) -> dict[str, Any]:
    output = {
        key: value
        for key, value in row.items()
        if key not in {"retrieved_docs", "selected_docs", "response"}
    }
    output["parent_id"] = parent_id
    output["subset_id"] = f"{parent_id}:{subset_type}:{subset_index}"
    output["subset_type"] = subset_type
    output["retrieved_docs"] = subset
    output["selected_docs"] = subset
    return output


def _subset_key(subset: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item.get("id", "")) for item in subset)


if __name__ == "__main__":
    main()
