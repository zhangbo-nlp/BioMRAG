"""Assemble canonical knowledge manifests with disk-backed ID validation."""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

from biomrag.schemas import EvidenceItem
from biomrag.utils.jsonl import iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--minimum-count", type=int, default=26_000_001)
    parser.add_argument("--commit-every", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_count < 0 or args.commit_every < 1:
        raise ValueError("minimum-count and commit-every are invalid.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_index = output_path.with_suffix(output_path.suffix + ".ids.sqlite3")
    if id_index.exists():
        raise FileExistsError(f"Temporary ID index already exists: {id_index}")
    connection = sqlite3.connect(id_index)
    connection.execute(
        "CREATE TABLE identifiers(identifier TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    counts = Counter()
    pending_ids: list[tuple[str]] = []

    def rows():
        for input_path in args.inputs:
            for row in iter_jsonl(input_path):
                item = _validated_item(row)
                pending_ids.append((item.id,))
                if len(pending_ids) >= args.commit_every:
                    _insert_ids(connection, pending_ids)
                    pending_ids.clear()
                counts["total"] += 1
                counts[f"modality:{item.modality}"] += 1
                counts[f"source:{item.source}"] += 1
                yield item.to_dict()
        if pending_ids:
            _insert_ids(connection, pending_ids)
            pending_ids.clear()

    try:
        write_jsonl(output_path, rows())
        connection.commit()
        if counts["total"] < args.minimum_count:
            raise ValueError(
                f"Assembled {counts['total']} entries; "
                f"expected at least {args.minimum_count}."
            )
        write_json(
            args.audit,
            {
                "total": counts["total"],
                "modalities": {
                    key.removeprefix("modality:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("modality:")
                },
                "sources": {
                    key.removeprefix("source:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("source:")
                },
                "inputs": args.inputs,
            },
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        id_index.unlink(missing_ok=True)


def _validated_item(row: dict) -> EvidenceItem:
    item = EvidenceItem.from_dict(row)
    if not item.id or not item.source.strip():
        raise ValueError("Every knowledge item requires a non-empty id and source.")
    has_text = bool(item.text.strip())
    has_image = bool(item.image_path.strip())
    expected_modality = (
        "image-text" if has_text and has_image else "image" if has_image else "text"
    )
    if not has_text and not has_image:
        raise ValueError(f"Knowledge item {item.id!r} has no text or image.")
    if item.modality != expected_modality:
        raise ValueError(
            f"Knowledge item {item.id!r} declares {item.modality!r}, "
            f"but its fields imply {expected_modality!r}."
        )
    return item


def _insert_ids(
    connection: sqlite3.Connection,
    identifiers: list[tuple[str]],
) -> None:
    try:
        connection.executemany(
            "INSERT INTO identifiers(identifier) VALUES (?)",
            identifiers,
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Knowledge-base identifiers must be globally unique.") from exc


if __name__ == "__main__":
    main()
