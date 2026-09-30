"""Segment text-only biomedical documents into approximately 200-token passages."""

from __future__ import annotations

import argparse

from biomrag.utils.jsonl import iter_jsonl, write_json, write_jsonl

DEFAULT_TOKENIZER = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["title", "text", "contents", "abstract", "passage"],
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=0)
    parser.add_argument("--expected-input-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_tokens < 1:
        raise ValueError("target-tokens must be positive.")
    if not 0 <= args.overlap_tokens < args.target_tokens:
        raise ValueError(
            "overlap-tokens must be non-negative and smaller than target-tokens."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Passage segmentation requires a fast tokenizer.")

    input_count = 0
    output_count = 0

    def passages():
        nonlocal input_count, output_count
        for row in iter_jsonl(args.input):
            input_count += 1
            parent_id = str(row.get(args.id_field, "")).strip()
            if not parent_id:
                raise ValueError(
                    f"Text document at input row {input_count} has no identifier."
                )
            text = " ".join(
                str(row.get(field, "") or "").strip()
                for field in args.text_fields
                if str(row.get(field, "") or "").strip()
            )
            if not text:
                raise ValueError(f"Text document {parent_id!r} is empty.")
            chunks = segment_text(
                text,
                tokenizer=tokenizer,
                target_tokens=args.target_tokens,
                overlap_tokens=args.overlap_tokens,
            )
            for chunk_index, chunk in enumerate(chunks):
                output_count += 1
                yield {
                    "id": (
                        parent_id
                        if len(chunks) == 1
                        else f"{parent_id}::chunk{chunk_index:04d}"
                    ),
                    "text": chunk,
                    "image_path": "",
                    "source": args.source,
                    "modality": "text",
                    "metadata": {
                        "parent_id": parent_id,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                    },
                }

    write_jsonl(args.output, passages())
    if args.expected_input_count and input_count != args.expected_input_count:
        raise ValueError(
            f"Read {input_count} source documents; "
            f"expected {args.expected_input_count}."
        )
    write_json(
        args.audit,
        {
            "input_documents": input_count,
            "output_passages": output_count,
            "target_tokens": args.target_tokens,
            "overlap_tokens": args.overlap_tokens,
            "tokenizer": args.tokenizer,
            "source": args.source,
        },
    )


def segment_text(
    text: str,
    *,
    tokenizer,
    target_tokens: int = 200,
    overlap_tokens: int = 0,
) -> list[str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
        if int(end) > int(start)
    ]
    if not offsets:
        raise ValueError("Tokenizer produced no content tokens.")
    step = target_tokens - overlap_tokens
    chunks = []
    for start_index in range(0, len(offsets), step):
        end_index = min(start_index + target_tokens, len(offsets))
        start_char = offsets[start_index][0]
        end_char = offsets[end_index - 1][1]
        chunk = text[start_char:end_char].strip()
        if chunk:
            chunks.append(chunk)
        if end_index == len(offsets):
            break
    if not chunks:
        raise ValueError("Passage segmentation produced no non-empty chunks.")
    return chunks


if __name__ == "__main__":
    main()
