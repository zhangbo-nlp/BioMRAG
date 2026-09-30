"""Build supervised proxy fine-tuning data from response and teacher-rating files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from biomrag.proxy.prompts import PROXY_SYSTEM_PROMPT, build_proxy_user_prompt
from biomrag.schemas import EvidenceItem, QueryExample
from biomrag.utils.jsonl import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="", help="Teacher-annotated JSONL.")
    parser.add_argument("--responses", default="", help="Evidence-conditioned response JSONL.")
    parser.add_argument("--ratings", default="", help="Teacher rating JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    args = parser.parse_args()
    if not args.annotations and not (args.responses and args.ratings):
        parser.error("Pass --annotations or both --responses and --ratings.")
    return args


def main() -> None:
    args = parse_args()
    output_count = 0

    def build_rows(pairs):
        nonlocal output_count
        for response_item, rating_item in pairs:
            output_count += 1
            yield build_proxy_sft_row(
                response_item,
                rating_item,
                data_root=args.data_root,
            )

    if args.annotations:
        write_jsonl(
            args.output,
            build_rows((item, item) for item in iter_jsonl(args.annotations)),
        )
        _validate_count(output_count, args.expected_count)
        return

    responses = _rows_by_key(args.responses)
    ratings = _rows_by_key(args.ratings)
    if set(responses) != set(ratings):
        missing_ratings = sorted(set(responses) - set(ratings))
        missing_responses = sorted(set(ratings) - set(responses))
        raise ValueError(
            "Response/rating IDs do not match. "
            f"Missing ratings={missing_ratings[:5]}, "
            f"missing responses={missing_responses[:5]}."
        )
    write_jsonl(
        args.output,
        build_rows((responses[key], ratings[key]) for key in responses),
    )
    _validate_count(output_count, args.expected_count)


def build_proxy_sft_row(
    response_item: dict[str, Any],
    rating_item: dict[str, Any],
    *,
    data_root: str = "",
) -> dict[str, Any]:
    score, rationale = extract_score_and_rationale(rating_item)
    query = extract_query(response_item, data_root=data_root)
    evidence = extract_evidence(response_item, data_root=data_root)
    if not query.id or not query.question.strip():
        raise ValueError("Proxy SFT rows require a non-empty query id and question.")
    if not evidence:
        raise ValueError(f"Proxy SFT query {query.id!r} has an empty evidence subset.")
    output = {
        "messages": [
            {"role": "system", "content": PROXY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_proxy_user_prompt(
                    query, evidence, include_image_tokens=True
                ),
            },
            {
                "role": "assistant",
                "content": f"Rationale: {rationale}\nScore: {score}",
            },
        ],
        "images": [
            path for path in [query.image_path, *[e.image_path for e in evidence]] if path
        ],
        "score": score,
        "rationale": rationale,
    }
    for key in ("id", "parent_id", "subset_id", "subset_type", "dataset"):
        if key in response_item:
            output[key] = response_item[key]
    return output


def extract_score_and_rationale(item: dict[str, Any]) -> tuple[int, str]:
    if isinstance(item.get("teacher_utility"), dict):
        return extract_score_and_rationale(item["teacher_utility"])
    if "score" in item:
        value = item["score"]
        if isinstance(value, bool):
            raise ValueError("Teacher utility score must be an integer in [1, 10].")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Teacher utility score must be an integer in [1, 10]."
            ) from exc
        if not numeric.is_integer():
            raise ValueError(f"Teacher utility score is not integral: {value!r}")
        score = int(numeric)
        rationale = str(item.get("rationale", "")).strip()
        if not 1 <= score <= 10 or not rationale:
            raise ValueError("Teacher utility requires a score in [1, 10] and rationale.")
        return score, rationale
    rating_text = str(item.get("rating", ""))
    rating_text = rating_text.replace("**", "")
    score_match = re.search(r"(?:score|rating)\D+(\d+)", rating_text, re.I)
    score = int(score_match.group(1)) if score_match else 0
    reason_match = re.search(r"(?:reason|rationale)\s*:?\s*(.*)", rating_text, re.I | re.S)
    rationale = reason_match.group(1).strip() if reason_match else rating_text.strip()
    if score <= 0:
        raise ValueError(f"Could not parse utility score from rating: {rating_text[:120]}")
    if score > 10 or not rationale:
        raise ValueError(f"Invalid utility target: score={score}, rationale={rationale!r}")
    return score, rationale


def extract_query(item: dict[str, Any], data_root: str = "") -> QueryExample:
    query = QueryExample.from_dict(item)
    if query.question:
        query.image_path = _rooted(query.image_path, data_root)
        return query

    messages = item.get("messages", [])
    question_parts = []
    image_path = ""
    for message in messages:
        if message.get("role") == "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            if message.get("role") == "user":
                question_parts.append(f"Q: {content}")
            elif message.get("role") == "assistant":
                question_parts.append(f"A: {content}")
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "image" and not image_path:
                    image_path = _rooted(part.get("image", ""), data_root)
                elif part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                question_parts.append(f"{message.get('role', 'user')}: {' '.join(text_parts)}")
    return QueryExample(id=str(item.get("id", "")), question="\n".join(question_parts), image_path=image_path)


def extract_evidence(item: dict[str, Any], data_root: str = "") -> list[EvidenceItem]:
    for key in ("selected_docs", "retrieved_docs", "evidence", "candidate_subset"):
        if key in item:
            evidence = [EvidenceItem.from_dict(doc) for doc in item[key]]
            for evidence_item in evidence:
                evidence_item.image_path = _rooted(
                    evidence_item.image_path,
                    data_root,
                )
            return evidence
    messages = item.get("messages", [])
    if not messages:
        return []
    system_message = messages[0]
    content = system_message.get("content", [])
    if isinstance(content, str):
        return [EvidenceItem(id="k1", text=content)]
    evidence = []
    pending_image = ""
    idx = 1
    for part in content[1:]:
        if part.get("type") == "image":
            pending_image = _rooted(part.get("image", ""), data_root)
        elif part.get("type") == "text":
            text = part.get("text", "").strip()
            if text:
                evidence.append(
                    EvidenceItem(id=f"k{idx}", text=text, image_path=pending_image)
                )
                idx += 1
                pending_image = ""
    return evidence


def _rooted(path: str, data_root: str) -> str:
    if not path or Path(path).is_absolute() or not data_root:
        return path
    return str(Path(data_root) / path)


def _rows_by_key(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = str(row.get("subset_id") or row.get("id") or "")
        if not key:
            raise ValueError(f"Every row in {path} needs subset_id or id.")
        if key in rows:
            raise ValueError(f"Duplicate row key {key!r} in {path}.")
        rows[key] = row
    return rows


def _validate_count(actual: int, expected: int) -> None:
    if expected and actual != expected:
        raise ValueError(f"Built {actual} proxy SFT rows; expected {expected}.")


if __name__ == "__main__":
    main()
