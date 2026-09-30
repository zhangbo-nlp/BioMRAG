"""Filter benchmark leakage from candidate knowledge-base entries."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

from biomrag.utils.jsonl import iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", required=True, help="Candidate knowledge JSONL.")
    parser.add_argument("--blocked", nargs="+", required=True, help="Benchmark split JSONL files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--min-ngram-hits", type=int, default=1)
    parser.add_argument("--caption-semantic-matches", default="")
    parser.add_argument("--caption-semantic-threshold", type=float, default=0.92)
    parser.add_argument("--qa-semantic-matches", default="")
    parser.add_argument("--qa-semantic-threshold", type=float, default=0.92)
    parser.add_argument("--manual-exclusions", default="")
    parser.add_argument("--minimum-kept", type=int, default=0)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip unreadable image paths instead of failing the leakage audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ngram_size < 1:
        raise ValueError("--ngram-size must be at least 1.")
    if args.min_ngram_hits < 1:
        raise ValueError("--min-ngram-hits must be at least 1.")
    if args.minimum_kept < 0:
        raise ValueError("--minimum-kept cannot be negative.")
    blocked = build_blocklist(
        args.blocked,
        ngram_size=args.ngram_size,
        allow_missing_images=args.allow_missing_images,
    )
    removed_counts = {
        "held_out_partition": 0,
        "exact_image_hash": 0,
        "perceptual_image_hash": 0,
        "caption_or_report_similarity": 0,
        "question_answer_overlap": 0,
        "manual_review": 0,
    }
    caption_semantic_ids = load_semantic_matches(
        args.caption_semantic_matches,
        threshold=args.caption_semantic_threshold,
    )
    qa_semantic_ids = load_semantic_matches(
        args.qa_semantic_matches,
        threshold=args.qa_semantic_threshold,
    )
    manual_ids = load_id_set(args.manual_exclusions)
    kept_count = 0

    def kept_rows():
        nonlocal kept_count
        for row in tqdm(iter_jsonl(args.kb), desc="Filtering KB"):
            reason = leakage_reason(
                row,
                blocked,
                args,
                caption_semantic_ids=caption_semantic_ids,
                qa_semantic_ids=qa_semantic_ids,
                manual_ids=manual_ids,
            )
            if reason:
                removed_counts[reason] += 1
                continue
            kept_count += 1
            yield row

    write_jsonl(args.output, kept_rows())
    audit = {
        "kept": kept_count,
        "removed": removed_counts,
        "blocked_images": len(blocked.image_sha256),
        "blocked_phashes": len(blocked.image_phash),
        "blocked_ngrams": len(blocked.qa_ngrams),
        "caption_semantic_matches": len(caption_semantic_ids),
        "qa_semantic_matches": len(qa_semantic_ids),
        "manual_exclusions": len(manual_ids),
    }
    write_json(
        args.audit,
        audit,
    )
    if args.minimum_kept and kept_count < args.minimum_kept:
        raise ValueError(
            f"Decontaminated index has {kept_count} entries; "
            f"expected at least {args.minimum_kept}."
        )


class Blocklist:
    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.image_sha256: set[str] = set()
        self.image_phash: set[int] = set()
        self.image_phash_index = HammingBKTree()
        self.qa_ngrams: set[str] = set()


def build_blocklist(
    paths: list[str],
    *,
    ngram_size: int,
    allow_missing_images: bool = False,
) -> Blocklist:
    blocked = Blocklist()
    for path in paths:
        for row in iter_jsonl(path):
            if "id" in row:
                blocked.ids.add(str(row["id"]))
            image_path = _image_path(row)
            if image_path:
                sha = image_sha256(
                    image_path,
                    strict=not allow_missing_images,
                )
                if sha:
                    blocked.image_sha256.add(sha)
                phash = image_dhash(
                    image_path,
                    strict=not allow_missing_images,
                )
                if phash is not None:
                    blocked.image_phash.add(phash)
                    blocked.image_phash_index.add(phash)
            blocked.qa_ngrams.update(
                text_ngrams(_question_answer_text(row), n=ngram_size)
            )
    return blocked


def leakage_reason(
    row: dict,
    blocked: Blocklist,
    args: argparse.Namespace,
    *,
    caption_semantic_ids: set[str] | None = None,
    qa_semantic_ids: set[str] | None = None,
    manual_ids: set[str] | None = None,
) -> str:
    if str(row.get("id", "")) in blocked.ids:
        return "held_out_partition"
    allow_missing_images = bool(getattr(args, "allow_missing_images", False))
    image_path = _image_path(row)
    if image_path:
        sha = image_sha256(image_path, strict=not allow_missing_images)
        if sha and sha in blocked.image_sha256:
            return "exact_image_hash"
        phash = image_dhash(image_path, strict=not allow_missing_images)
        if phash is not None and blocked.image_phash_index.within_distance(
            phash,
            args.phash_threshold,
        ):
            return "perceptual_image_hash"
    item_id = str(row.get("id", ""))
    if caption_semantic_ids and item_id in caption_semantic_ids:
        return "caption_or_report_similarity"
    if qa_semantic_ids and item_id in qa_semantic_ids:
        return "question_answer_overlap"
    hits = len(
        text_ngrams(_candidate_overlap_text(row), n=args.ngram_size)
        & blocked.qa_ngrams
    )
    if hits >= args.min_ngram_hits:
        return "question_answer_overlap"
    if manual_ids and str(row.get("id", "")) in manual_ids:
        return "manual_review"
    return ""


def load_semantic_matches(path: str, *, threshold: float) -> set[str]:
    if not path:
        return set()
    blocked = set()
    for row in iter_jsonl(path):
        score = row.get("score", row.get("similarity"))
        if score is None:
            raise ValueError(
                f"Semantic-match row requires score or similarity: {row!r}"
            )
        try:
            score_value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid semantic similarity: {score!r}") from exc
        if score_value >= threshold:
            item_id = row.get("id") or row.get("kb_id") or row.get("candidate_id")
            if item_id is None:
                raise ValueError(
                    f"Semantic-match row requires a candidate identifier: {row!r}"
                )
            blocked.add(str(item_id))
    return blocked


def load_id_set(path: str) -> set[str]:
    if not path:
        return set()
    values = set()
    for row in iter_jsonl(path):
        item_id = row.get("id") or row.get("kb_id") or row.get("candidate_id")
        if item_id is None:
            raise ValueError(f"Manual-exclusion row requires an identifier: {row!r}")
        values.add(str(item_id))
    return values


def image_sha256(path: str, *, strict: bool = False) -> str:
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError as exc:
        if strict:
            raise FileNotFoundError(
                f"Unreadable image during SHA-256 audit: {path}"
            ) from exc
        return ""


def image_dhash(path: str, *, strict: bool = False) -> int | None:
    from PIL import Image

    try:
        with Image.open(Path(path)) as source:
            image = source.convert("L").resize((9, 8))
    except OSError as exc:
        if strict:
            raise FileNotFoundError(
                f"Unreadable image during perceptual-hash audit: {path}"
            ) from exc
        return None
    pixels = list(image.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits.append(pixels[offset + col] > pixels[offset + col + 1])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def text_ngrams(text: str, *, n: int) -> set[str]:
    tokens = normalize_text(text).split()
    return {" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    replacements = {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "μ": " micro ",
        "µ": " micro ",
        "millimeter": " mm ",
        "millimeters": " mm ",
        "centimeter": " cm ",
        "centimeters": " cm ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[\W_]+", " ", text)
    return " ".join(text.split())


def _image_path(row: dict) -> str:
    for key in ("image_path", "image"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    images = row.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _question_answer_text(row: dict) -> str:
    parts = [
        str(row.get("question", "") or ""),
        str(row.get("answer", "") or row.get("reference", "") or ""),
    ]
    turns = row.get("messages") or row.get("conversations") or []
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("from") or "").lower()
            if role not in {"user", "human", "assistant", "gpt", "model"}:
                continue
            content = turn.get("content", turn.get("value", ""))
            if isinstance(content, str):
                parts.append(content.replace("<image>", ""))
            elif isinstance(content, list):
                parts.extend(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
    return " ".join(parts)


def _candidate_overlap_text(row: dict) -> str:
    return " ".join(
        str(row.get(key, "") or "")
        for key in ("text", "caption", "report", "question", "answer")
    )


class HammingBKTree:
    """BK-tree for exact radius search over 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: tuple[int, dict[int, tuple]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            node_value, children = node
            distance = hamming_distance(value, node_value)
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child

    def within_distance(self, value: int, threshold: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = hamming_distance(value, node_value)
            if distance <= threshold:
                return True
            lower = distance - threshold
            upper = distance + threshold
            stack.extend(
                child
                for edge, child in children.items()
                if lower <= edge <= upper
            )
        return False


if __name__ == "__main__":
    main()
