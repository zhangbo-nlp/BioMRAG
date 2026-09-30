"""Merge no-evidence and evidence-conditioned answers for teacher annotation."""

from __future__ import annotations

import argparse
from typing import Any

from biomrag.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-responses", required=True)
    parser.add_argument("--no-evidence-responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-evidence-count", type=int, default=0)
    parser.add_argument("--expected-no-evidence-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_evidence_rows = read_jsonl(args.no_evidence_responses)
    if (
        args.expected_no_evidence_count
        and len(no_evidence_rows) != args.expected_no_evidence_count
    ):
        raise ValueError(
            f"Read {len(no_evidence_rows)} no-evidence responses; "
            f"expected {args.expected_no_evidence_count}."
        )
    no_evidence_by_key: dict[str, str] = {}
    for row in no_evidence_rows:
        key = _parent_key(row)
        response = str(
            row.get("response") or row.get("prediction") or ""
        ).strip()
        if not response:
            raise ValueError(f"Empty no-evidence response for parent {key!r}.")
        if key in no_evidence_by_key and no_evidence_by_key[key] != response:
            raise ValueError(
                f"No-evidence responses disagree for parent query {key!r}."
            )
        no_evidence_by_key[key] = response

    evidence_rows = read_jsonl(args.evidence_responses)
    if (
        args.expected_evidence_count
        and len(evidence_rows) != args.expected_evidence_count
    ):
        raise ValueError(
            f"Read {len(evidence_rows)} evidence responses; "
            f"expected {args.expected_evidence_count}."
        )
    evidence_parent_keys = {_parent_key(row) for row in evidence_rows}
    if evidence_parent_keys != set(no_evidence_by_key):
        raise ValueError(
            "Evidence-conditioned and no-evidence parent ID sets must match exactly."
        )

    def merged_rows():
        seen_subsets = set()
        for row in evidence_rows:
            parent_key = _parent_key(row)
            subset_key = str(row.get("subset_id") or row.get("id") or "")
            if not subset_key or subset_key in seen_subsets:
                raise ValueError(
                    f"Evidence responses need unique subset IDs; got {subset_key!r}."
                )
            seen_subsets.add(subset_key)
            if parent_key not in no_evidence_by_key:
                raise KeyError(
                    f"Missing no-evidence response for parent={parent_key}"
                )
            response = str(
                row.get("response") or row.get("prediction") or ""
            ).strip()
            if not response:
                raise ValueError(
                    f"Empty evidence-conditioned response for subset {subset_key!r}."
                )
            output = dict(row)
            output["response_without_knowledge"] = no_evidence_by_key[parent_key]
            yield output

    write_jsonl(args.output, merged_rows())


def _parent_key(row: dict[str, Any]) -> str:
    key = str(row.get("parent_id") or row.get("id") or "")
    if not key:
        raise ValueError("Each response row requires parent_id or id.")
    return key


if __name__ == "__main__":
    main()
