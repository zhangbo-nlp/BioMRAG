"""Run subset-utility evidence selection on retrieved candidates."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tqdm import tqdm

from biomrag.proxy.scoring import OpenAICompatibleProxyScorer
from biomrag.schemas import EvidenceItem, QueryExample
from biomrag.selection.streaming_selector import streaming_proxy_deliberate
from biomrag.utils.jsonl import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with retrieved_docs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=0,
        help="Fail unless every row has this many candidates; 0 disables the check.",
    )
    parser.add_argument("--proxy-model", default="biomrag-proxy")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("PROXY_API_BASE_URL")
        or os.environ.get("API_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PROXY_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--retriever-model", default="")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = OpenAICompatibleProxyScorer(
        model=args.proxy_model,
        base_url=args.api_base_url,
        api_key=args.api_key,
    )
    embedder = load_embedder(args.retriever_model, device=args.device)
    outputs = []
    for row in tqdm(iter_jsonl(args.input), desc="Selecting evidence"):
        started = time.perf_counter()
        query = QueryExample.from_dict(row)
        candidates = [EvidenceItem.from_dict(doc) for doc in row.get("retrieved_docs", [])]
        if args.expected_candidates and len(candidates) != args.expected_candidates:
            raise ValueError(
                f"Query {query.id!r} has {len(candidates)} candidates; "
                f"expected {args.expected_candidates}."
            )
        lookup = build_embedding_lookup(
            candidates,
            embedder=embedder,
        )
        result = streaming_proxy_deliberate(
            query,
            candidates,
            scorer,
            budget=args.budget,
            embedding_lookup=lookup,
        )
        output = dict(row)
        output["selected_docs"] = [
            item.to_dict() for item in result.selected
        ]
        output["selection_score"] = result.score
        output["selection_rationale"] = result.rationale
        output["selection_trace"] = result.trace_dicts()
        output["proxy_calls"] = len(result.steps)
        if output["proxy_calls"] != len(candidates):
            raise RuntimeError(
                f"Selector made {output['proxy_calls']} proxy calls for "
                f"{len(candidates)} candidates."
            )
        output["selection_latency"] = time.perf_counter() - started
        output["total_latency"] = sum(
            float(output.get(key, 0.0))
            for key in ("retrieval_latency", "selection_latency")
        )
        outputs.append(output)
    write_jsonl(args.output, outputs)


def build_embedding_lookup(
    candidates: list[EvidenceItem],
    *,
    embedder=None,
):
    embeddings = {
        item.id: item.metadata["embedding"]
        for item in candidates
        if isinstance(item.metadata.get("embedding"), list)
    }
    missing = [item for item in candidates if item.id not in embeddings]
    if missing and embedder is not None:
        matrix = embedder.encode_batch([item.to_dict() for item in missing])
        embeddings.update(
            {
                item.id: vector.astype(float).tolist()
                for item, vector in zip(missing, matrix)
            }
        )
    if missing and embedder is None:
        missing_ids = [item.id for item in missing]
        raise ValueError(
            "Redundancy-guided selection requires stored embeddings or "
            f"--retriever-model; missing IDs: {missing_ids[:5]}"
        )
    if not candidates:
        return None

    def lookup(item: EvidenceItem):
        return embeddings.get(item.id)

    return lookup


def load_embedder(model_path: str, *, device: str | None):
    if not model_path:
        return None
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Retriever checkpoint does not exist: {model_path}")
    from biomrag.retrieval.embedder import BioMRAGEmbedder

    return (
        BioMRAGEmbedder(model_path, device=device)
        if device
        else BioMRAGEmbedder(model_path)
    )


if __name__ == "__main__":
    main()
