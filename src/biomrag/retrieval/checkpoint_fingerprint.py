"""Stable fingerprints for retriever weights and preprocessing assets."""

from __future__ import annotations

import hashlib
from pathlib import Path

ASSET_PATTERNS = (
    "config.json",
    "model*.safetensors",
    "pytorch_model*.bin",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def checkpoint_fingerprint(path: str | Path) -> str:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Retriever checkpoint is not a directory: {directory}")
    files = sorted(
        {
            candidate
            for pattern in ASSET_PATTERNS
            for candidate in directory.glob(pattern)
            if candidate.is_file()
        },
        key=lambda candidate: candidate.name,
    )
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(
            f"Retriever checkpoint is missing config.json: {directory}"
        )
    if not any(
        path.name.startswith(("model", "pytorch_model"))
        and path.suffix in {".safetensors", ".bin"}
        for path in files
    ):
        raise FileNotFoundError(
            f"Retriever checkpoint has no model weight files: {directory}"
        )

    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
