"""Retrieve BioMRAG evidence candidates from a FAISS index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from biomrag.retrieval.checkpoint_fingerprint import checkpoint_fingerprint
from biomrag.retrieval.embedder import BioMRAGEmbedder
from biomrag.schemas import QueryExample
from biomrag.utils.jsonl import iter_jsonl, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Directory with index.faiss.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", default=None, help="Retriever checkpoint override.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.nprobe < 1:
        raise ValueError("top-k and nprobe must be positive.")
    index_dir = Path(args.index)

    import faiss

    default_model_path = index_dir.parent / "retriever_inference"
    if not default_model_path.exists():
        default_model_path = index_dir.parent / "retriever"
    model_path = args.model or str(default_model_path)
    actual_fingerprint = checkpoint_fingerprint(model_path)
    embedder = BioMRAGEmbedder(
        model_path,
        device=args.device or ("cuda" if _cuda_available() else "cpu"),
    )
    index = faiss.read_index(str(index_dir / "index.faiss"))
    config = _read_index_config(index_dir)
    if config.get("metric") != "inner_product" or not config.get("normalized"):
        raise ValueError(
            "BioMRAG retrieval requires a normalized inner-product index."
        )
    if int(config.get("dimension", 0)) != int(embedder.model.projection_dim):
        raise ValueError("Retriever projection dimension does not match the index.")
    expected_fingerprint = str(config.get("checkpoint_sha256", ""))
    if not expected_fingerprint:
        raise ValueError("Index metadata is missing the checkpoint fingerprint.")
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "The retrieval checkpoint does not match the indexed checkpoint."
        )
    if hasattr(index, "nprobe"):
        if args.nprobe > int(config.get("nlist", args.nprobe)):
            raise ValueError("nprobe cannot exceed the index nlist.")
        index.nprobe = args.nprobe
    metadata = MetadataStore(index_dir)
    if len(metadata) != int(index.ntotal):
        raise ValueError(
            f"FAISS index has {index.ntotal} vectors but metadata has "
            f"{len(metadata)} rows."
        )
    stored_embeddings = None
    if args.include_embeddings:
        embeddings_path = index_dir / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"{embeddings_path} is required when --include-embeddings is set."
            )
        stored_embeddings = np.load(embeddings_path, mmap_mode="r")
        if stored_embeddings.shape != (
            int(index.ntotal),
            int(embedder.model.projection_dim),
        ):
            raise ValueError(
                "Stored embedding matrix shape does not match the FAISS index."
            )

    outputs = []
    try:
        for row in tqdm(iter_jsonl(args.queries), desc="Retrieving"):
            started = time.perf_counter()
            query = QueryExample.from_dict(row)
            vector = embedder.encode(
                text=query.question,
                image_path=query.image_path,
            )
            vector = vector.reshape(1, -1).astype("float32")
            if vector.shape[1] != int(index.d) or not np.isfinite(vector).all():
                raise ValueError(
                    f"Invalid query embedding for query {query.id!r}: {vector.shape}"
                )
            faiss.normalize_L2(vector)
            scores, ids = index.search(vector, args.top_k)
            retrieved = []
            for rank, (idx, score) in enumerate(
                zip(ids[0], scores[0]),
                start=1,
            ):
                if idx < 0:
                    continue
                item = dict(metadata[int(idx)])
                item["retrieval_rank"] = rank
                item["retrieval_score"] = float(score)
                if stored_embeddings is not None:
                    item["embedding"] = (
                        stored_embeddings[int(idx)].astype(float).tolist()
                    )
                retrieved.append(item)
            expected_results = min(args.top_k, int(index.ntotal))
            if len(retrieved) != expected_results:
                raise RuntimeError(
                    f"Query {query.id!r} returned {len(retrieved)} candidates; "
                    f"expected {expected_results}."
                )
            result_ids = [str(item.get("id", "")) for item in retrieved]
            if any(not item_id for item_id in result_ids):
                raise ValueError("Retrieved knowledge entries require non-empty IDs.")
            if len(set(result_ids)) != len(result_ids):
                raise ValueError(
                    f"Query {query.id!r} retrieved duplicate knowledge IDs."
                )
            output = dict(row)
            output["retrieved_docs"] = retrieved
            output["retrieval_latency"] = time.perf_counter() - started
            output["total_latency"] = output["retrieval_latency"]
            outputs.append(output)
    finally:
        metadata.close()

    write_jsonl(args.output, outputs)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _read_index_config(index_dir: Path) -> dict:
    config_path = index_dir / "index_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing BioMRAG index metadata: {config_path}"
        )
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


class MetadataStore:
    """Random-access JSONL metadata using byte offsets when available."""

    def __init__(self, index_dir: Path):
        self.path = index_dir / "metadata.jsonl"
        offsets_path = index_dir / "metadata_offsets.npy"
        self._rows = None
        self._offsets = None
        self._handle = None
        if offsets_path.exists():
            self._offsets = np.load(offsets_path, mmap_mode="r")
            self._handle = self.path.open("rb")
        else:
            self._rows = read_jsonl(self.path)

    def __len__(self) -> int:
        return (
            len(self._rows)
            if self._rows is not None
            else len(self._offsets)
        )

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._rows is not None:
            return self._rows[index]
        self._handle.seek(int(self._offsets[index]))
        return json.loads(self._handle.readline())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


if __name__ == "__main__":
    main()
