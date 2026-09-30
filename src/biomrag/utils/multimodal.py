"""OpenAI-compatible multimodal message helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Sequence

REMOTE_IMAGE_PREFIXES = ("http://", "https://", "data:")


def image_reference(path_or_url: str) -> str:
    """Return a remote URL unchanged or encode a local image as a data URI."""
    value = str(path_or_url or "").strip()
    if not value:
        raise ValueError("Image reference cannot be empty.")
    if value.startswith(REMOTE_IMAGE_PREFIXES):
        return value

    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def image_url_part(path_or_url: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": image_reference(path_or_url)},
    }


def content_from_image_tokens(
    prompt: str,
    image_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Replace ordered ``<image>`` tokens with multimodal content parts."""
    sections = prompt.split("<image>")
    expected = len(sections) - 1
    if expected != len(image_paths):
        raise ValueError(
            f"Prompt contains {expected} <image> tokens but received "
            f"{len(image_paths)} image paths."
        )

    content: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        if section:
            content.append({"type": "text", "text": section})
        if index < len(image_paths):
            content.append(image_url_part(image_paths[index]))
    return content or [{"type": "text", "text": ""}]

