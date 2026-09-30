"""Prepare proxy supervision data for ms-swift SFT."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from biomrag.utils.jsonl import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-output", default="")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        to_swift_row(row, drop_missing_images=args.drop_missing_images)
        for row in iter_jsonl(args.input)
    ]
    if args.expected_count and len(rows) != args.expected_count:
        raise ValueError(
            f"Prepared {len(rows)} ms-swift rows; expected {args.expected_count}."
        )
    if args.val_output and args.val_ratio > 0:
        train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
        write_jsonl(args.output, train_rows)
        write_jsonl(args.val_output, val_rows)
    else:
        write_jsonl(args.output, rows)


def to_swift_row(row: dict[str, Any], *, drop_missing_images: bool = False) -> dict[str, Any]:
    messages = [normalize_message(message) for message in row.get("messages", [])]
    if not messages:
        raise ValueError("Each row must contain a non-empty messages list.")

    images = normalize_images(row.get("images", []), drop_missing=drop_missing_images)
    ensure_image_tokens(messages, len(images))

    for message in reversed(messages):
        if message.get("role") == "assistant":
            message.setdefault("loss", True)
            break

    output = {"messages": messages}
    if images:
        output["images"] = images
    if "score" in row:
        output["score"] = row["score"]
    if "rationale" in row:
        output["rationale"] = row["rationale"]
    for key in (
        "id",
        "parent_id",
        "subset_id",
        "subset_type",
        "dataset",
    ):
        if key in row:
            output[key] = row[key]
    return output


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "")).strip()
    content = message.get("content", "")
    if isinstance(content, list):
        content = flatten_multimodal_content(content)
    return {"role": role, "content": str(content)}


def flatten_multimodal_content(content: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in content:
        if item.get("type") == "image":
            parts.append("<image>")
        elif item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def normalize_images(images: Any, *, drop_missing: bool) -> list[str]:
    if not isinstance(images, list):
        return []
    output = []
    for image in images:
        image_path = str(image)
        if not image_path:
            continue
        if drop_missing and not image_path.startswith(("http://", "https://", "data:")):
            if not Path(image_path).exists():
                continue
        output.append(image_path)
    return output


def ensure_image_tokens(messages: list[dict[str, Any]], image_count: int) -> None:
    existing = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
    )
    if existing != image_count:
        raise ValueError(
            f"Found {existing} <image> tokens and {image_count} image paths; "
            "the ordered multimodal inputs must match exactly."
        )


def split_rows(
    rows: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < val_ratio < 1:
        return rows, []
    group_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = str(
            row.get("parent_id")
            or row.get("id")
            or row.get("subset_id")
            or index
        )
        group_to_indices.setdefault(group, []).append(index)

    groups = list(group_to_indices)
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_rows = max(1, int(round(len(rows) * val_ratio)))
    val_indices: set[int] = set()
    for group in groups:
        if val_indices and len(val_indices) >= target_rows:
            break
        val_indices.update(group_to_indices[group])
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]
    if not train_rows or not val_rows:
        raise ValueError(
            "Group-wise proxy split must leave non-empty train and validation sets."
        )
    return train_rows, val_rows


if __name__ == "__main__":
    main()
