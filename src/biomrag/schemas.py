"""BioMRAG data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceItem:
    """A text, image, or image-text knowledge item."""

    id: str
    text: str = ""
    image_path: str = ""
    source: str = ""
    modality: str = "text"
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        known = {
            "id",
            "text",
            "image_path",
            "source",
            "modality",
            "retrieval_rank",
            "retrieval_score",
            "metadata",
        }
        metadata = dict(data.get("metadata") or {})
        metadata.update({k: v for k, v in data.items() if k not in known})
        text = (
            data.get("text")
            or data.get("caption")
            or data.get("report")
            or data.get("passage")
            or data.get("abstract")
            or ""
        )
        image_path = _first_image_reference(data)
        modality = str(data.get("modality", "") or "").strip().lower()
        if not modality:
            modality = (
                "image-text"
                if text and image_path
                else "image"
                if image_path
                else "text"
            )
        if modality not in {"text", "image", "image-text"}:
            raise ValueError(f"Unsupported evidence modality: {modality!r}")
        return cls(
            id=str(data.get("id", "")),
            text=str(text),
            image_path=image_path,
            source=str(data.get("source", "") or ""),
            modality=modality,
            retrieval_rank=data.get("retrieval_rank"),
            retrieval_score=data.get("retrieval_score"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "text": self.text,
            "image_path": self.image_path,
            "source": self.source,
            "modality": self.modality,
            "metadata": self.metadata,
        }
        if self.retrieval_rank is not None:
            data["retrieval_rank"] = self.retrieval_rank
        if self.retrieval_score is not None:
            data["retrieval_score"] = self.retrieval_score
        return data


@dataclass(slots=True)
class QueryExample:
    """A biomedical question or dialogue query."""

    id: str
    question: str
    image_path: str = ""
    answer: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryExample":
        question = data.get("question") or data.get("query") or ""
        if not question and "messages" in data:
            question = _extract_last_user_text(data["messages"])
        if not question and "conversations" in data:
            question = _extract_last_user_text(data["conversations"])
        answer = (
            data.get("answer")
            or data.get("reference")
            or data.get("reference_answer")
            or ""
        )
        history = list(
            data.get("history")
            or data.get("messages")
            or data.get("conversations")
            or []
        )
        if not answer:
            answer = _extract_target_assistant_text(history)
        image_path = _first_image_reference(data)
        if not image_path:
            image_path = _extract_last_user_image(history)
        return cls(
            id=str(data.get("id", "")),
            question=str(question),
            image_path=image_path,
            answer=str(answer),
            history=history,
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "id",
                    "question",
                    "query",
                    "image_path",
                    "answer",
                    "reference",
                    "reference_answer",
                    "history",
                    "messages",
                    "conversations",
                }
            },
        )


@dataclass(slots=True)
class UtilityScore:
    """Proxy utility output for an evidence subset."""

    score: float
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role not in {"user", "human"}:
            continue
        content = message.get("content", message.get("value", ""))
        if isinstance(content, str):
            return content.replace("<image>", "").strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(part.strip() for part in parts if part.strip())
    return ""


def _extract_target_assistant_text(messages: list[dict[str, Any]]) -> str:
    last_user_index = -1
    for index, message in enumerate(messages):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"}:
            last_user_index = index
    if last_user_index < 0:
        return ""
    for message in messages[last_user_index + 1 :]:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role not in {"assistant", "gpt", "model"}:
            continue
        content = message.get("content", message.get("value", ""))
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text")
            ]
            return " ".join(part for part in parts if part)
    return ""


def _first_image_reference(data: dict[str, Any]) -> str:
    for key in ("image_path", "image"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    images = data.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_last_user_image(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role not in {"user", "human"}:
            continue
        content = message.get("content", message.get("value", ""))
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "image" and part.get("image"):
                return str(part["image"])
            if kind == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and image_url.get("url"):
                    return str(image_url["url"])
                if isinstance(image_url, str):
                    return image_url
    return ""
