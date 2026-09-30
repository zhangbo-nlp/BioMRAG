"""Proxy scoring adapters."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence

from biomrag.proxy.prompts import PROXY_SYSTEM_PROMPT, build_proxy_user_prompt
from biomrag.schemas import EvidenceItem, QueryExample, UtilityScore
from biomrag.utils.multimodal import content_from_image_tokens


class OpenAICompatibleProxyScorer:
    """Score evidence with a deployed proxy behind an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=(
                api_key
                or os.environ.get("PROXY_API_KEY")
                or os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            ),
            base_url=(
                base_url
                or os.environ.get("PROXY_API_BASE_URL")
                or os.environ.get("API_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            ),
        )

    def score(self, query: QueryExample, evidence: Sequence[EvidenceItem]) -> UtilityScore:
        prompt = build_proxy_user_prompt(
            query,
            evidence,
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": PROXY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content_from_image_tokens(prompt, image_paths),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        return parse_proxy_response(content)


def parse_proxy_response(text: str) -> UtilityScore:
    """Parse and validate one rationale plus an integer score in [1, 10]."""
    data: dict | None = None
    try:
        data = json.loads(_extract_json_object(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        data = None

    if data is not None:
        score = _validated_score(data.get("score"))
        rationale = str(data.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Proxy response contains an empty rationale.")
        return UtilityScore(score=score, rationale=rationale, raw=data)

    score_match = re.search(r"(?:^|\n)\s*score\s*:\s*(\d+)\s*$", text, re.I)
    rationale_match = re.search(
        r"(?:^|\n)\s*rationale\s*:\s*(.*?)(?=\n\s*score\s*:)",
        text,
        re.I | re.S,
    )
    if score_match is None or rationale_match is None:
        raise ValueError(f"Malformed proxy response: {text[:240]!r}")
    score = _validated_score(score_match.group(1))
    rationale = rationale_match.group(1).strip()
    if not rationale:
        raise ValueError("Proxy response contains an empty rationale.")
    return UtilityScore(score=score, rationale=rationale, raw={"text": text})


def _validated_score(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Proxy score must be an integer from 1 to 10.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Proxy score must be an integer from 1 to 10.") from exc
    if not numeric.is_integer() or not 1 <= numeric <= 10:
        raise ValueError(f"Proxy score is outside the ordinal scale: {value!r}")
    return int(numeric)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
