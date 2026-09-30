"""Create teacher utility labels for candidate evidence subsets."""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from biomrag.proxy.prompts import TEACHER_SYSTEM_PROMPT, build_teacher_user_prompt
from biomrag.schemas import EvidenceItem, QueryExample
from biomrag.utils.jsonl import iter_jsonl, write_jsonl
from biomrag.utils.multimodal import content_from_image_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with query/evidence/answers.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "gpt-4o"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TEACHER_API_BASE_URL")
        or os.environ.get("API_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TEACHER_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Fail unless this many teacher requests are processed; 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    processed = 0

    def annotated_rows():
        nonlocal processed
        for idx, row in enumerate(
            tqdm(iter_jsonl(args.input), desc="Teacher annotation")
        ):
            if args.limit is not None and idx >= args.limit:
                break
            query = QueryExample.from_dict(row)
            evidence_rows = row.get("retrieved_docs") or row.get("selected_docs") or []
            evidence = [EvidenceItem.from_dict(doc) for doc in evidence_rows]
            if not query.id or not query.question.strip():
                raise ValueError("Teacher requests require a query id and question.")
            evidence_ids = [item.id for item in evidence]
            if (
                not evidence_ids
                or any(not item_id for item_id in evidence_ids)
                or len(set(evidence_ids)) != len(evidence_ids)
            ):
                raise ValueError(
                    f"Teacher request {row.get('subset_id')!r} requires a "
                    "non-empty subset with unique evidence IDs."
                )
            no_evidence_answer = str(
                row.get("response_without_knowledge", "")
            ).strip()
            generated_answer = str(row.get("response", "")).strip()
            reference_answer = str(
                row.get("answer", "")
                or row.get("reference", "")
                or row.get("reference_answer", "")
                or query.answer
            ).strip()
            if (
                not no_evidence_answer
                or not generated_answer
                or not reference_answer
            ):
                raise ValueError(
                    f"Teacher request {row.get('subset_id') or row.get('id')!r} "
                    "requires no-evidence, evidence-conditioned, and reference answers."
                )
            prompt = build_teacher_user_prompt(
                query=query,
                evidence=evidence,
                no_evidence_answer=no_evidence_answer,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                include_image_tokens=True,
            )
            image_paths = [
                path
                for path in [
                    query.image_path,
                    *[item.image_path for item in evidence],
                ]
                if path
            ]
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": content_from_image_tokens(prompt, image_paths),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            try:
                parsed = json.loads(_extract_json_object(content))
            except json.JSONDecodeError:
                parsed = {"rationale": content, "score": None}
            output = dict(row)
            output["teacher_utility"] = parsed
            processed += 1
            yield output

    write_jsonl(args.output, annotated_rows())
    if args.expected_count and processed != args.expected_count:
        raise ValueError(
            f"Processed {processed} teacher requests; expected {args.expected_count}."
        )


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text


if __name__ == "__main__":
    main()
