"""Generate answers from selected BioMRAG evidence."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from tqdm import tqdm

from biomrag.proxy.prompts import (
    NO_RETRIEVAL_SYSTEM_PROMPT,
    build_rag_generation_prompt,
)
from biomrag.schemas import EvidenceItem, QueryExample
from biomrag.utils.jsonl import iter_jsonl, write_jsonl
from biomrag.utils.multimodal import image_url_part


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with selected_docs.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "GENERATOR_MODEL",
            "Qwen/Qwen2.5-VL-72B-Instruct",
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GENERATOR_API_BASE_URL")
        or os.environ.get("API_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GENERATOR_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.001)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    generated_count = 0
    seen_output_ids: set[str] = set()
    seen_no_retrieval_queries: set[str] = set()

    def generated_rows():
        nonlocal generated_count
        for row in tqdm(iter_jsonl(args.input), desc="Generating"):
            if args.no_retrieval:
                parent_id = str(row.get("parent_id") or row.get("id") or "")
                if not parent_id:
                    raise ValueError(
                        "No-retrieval proxy responses require parent_id or id."
                    )
                if parent_id in seen_no_retrieval_queries:
                    continue
                seen_no_retrieval_queries.add(parent_id)
            started = time.perf_counter()
            query = QueryExample.from_dict(row)
            output_id = str(row.get("subset_id") or query.id)
            if not query.id or not query.question.strip():
                raise ValueError(
                    "Generation rows require a non-empty query id and question."
                )
            if output_id in seen_output_ids:
                raise ValueError(f"Duplicate generation request id: {output_id!r}")
            seen_output_ids.add(output_id)
            if "selected_docs" in row:
                evidence_rows = row["selected_docs"]
            else:
                evidence_rows = row.get("retrieved_docs") or []
                if not evidence_rows:
                    raise ValueError(
                        f"Retrieval-augmented generation request {output_id!r} "
                        "has no evidence."
                    )
            evidence = [EvidenceItem.from_dict(doc) for doc in evidence_rows]
            # An empty subset is a legal BioMRAG outcome: no proposal was net
            # beneficial, so the generator answers from parametric knowledge.
            no_retrieval = args.no_retrieval or not evidence
            messages = build_messages(query, evidence, no_retrieval=no_retrieval)
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            output = dict(row)
            output["response"] = (
                completion.choices[0].message.content or ""
            ).strip()
            if not output["response"]:
                raise ValueError(
                    f"Generator returned an empty answer for {output_id!r}."
                )
            output["generation_latency"] = time.perf_counter() - started
            output["total_latency"] = sum(
                float(output.get(key, 0.0))
                for key in (
                    "retrieval_latency",
                    "selection_latency",
                    "generation_latency",
                )
            )
            generated_count += 1
            yield output

    write_jsonl(args.output, generated_rows())
    if args.expected_count and generated_count != args.expected_count:
        raise ValueError(
            f"Generated {generated_count} answers; expected {args.expected_count}."
        )


def build_messages(
    query: QueryExample,
    evidence: list[EvidenceItem],
    *,
    no_retrieval: bool = False,
) -> list[dict[str, Any]]:
    system_prompt = (
        NO_RETRIEVAL_SYSTEM_PROMPT
        if no_retrieval
        else build_rag_generation_prompt(evidence)
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(
        _history_before_current_query(
            query.history,
            current_question=query.question,
        )
    )

    user_content: list[dict[str, Any]] = []
    if query.image_path:
        user_content.append(image_url_part(query.image_path))
    user_content.append({"type": "text", "text": query.question})
    if not no_retrieval:
        for item in evidence:
            if not item.image_path:
                continue
            user_content.append(
                {
                    "type": "text",
                    "text": f"Retrieved visual evidence [{item.id}]:",
                }
            )
            user_content.append(image_url_part(item.image_path))
    messages.append({"role": "user", "content": user_content})
    return messages


def _history_before_current_query(
    history: list[dict[str, Any]],
    *,
    current_question: str,
) -> list[dict[str, Any]]:
    """Keep prior dialogue turns while excluding the current and target answer."""
    if not history:
        return []
    current_user = -1
    normalized_question = _normalized_text(current_question)
    for index, message in enumerate(history):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role not in {"user", "human"}:
            continue
        if _normalized_text(_message_text(message)) == normalized_question:
            current_user = index
    cutoff = current_user if current_user >= 0 else len(history)

    output: list[dict[str, Any]] = []
    for message in history[:cutoff]:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role == "human":
            role = "user"
        elif role in {"gpt", "model"}:
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        content = _normalize_history_content(
            message.get("content", message.get("value", ""))
        )
        if content:
            output.append({"role": role, "content": content})
    return output


def _normalize_history_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content.replace("<image>", "").strip()
    if not isinstance(content, list):
        return str(content) if content else ""

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text" and item.get("text"):
            parts.append({"type": "text", "text": str(item["text"])})
        elif kind in {"image", "image_url"}:
            image = item.get("image")
            if kind == "image_url":
                image_url = item.get("image_url")
                image = (
                    image_url.get("url")
                    if isinstance(image_url, dict)
                    else image_url
                )
            if image:
                parts.append(image_url_part(str(image)))
    return parts


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", message.get("value", ""))
    if isinstance(content, str):
        return content.replace("<image>", "").strip()
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(item.get("text", "")).strip()
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and item.get("text")
    )


def _normalized_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


if __name__ == "__main__":
    main()
