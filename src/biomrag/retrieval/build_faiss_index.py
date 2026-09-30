"""Build a FAISS index for BioMRAG knowledge entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from biomrag.retrieval.checkpoint_fingerprint import checkpoint_fingerprint
from biomrag.retrieval.embedder import BioMRAGEmbedder
from biomrag.schemas import EvidenceItem
from biomrag.utils.jsonl import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", required=True, help="Knowledge JSONL.")
    parser.add_argument("--model", required=True, help="Trained retriever checkpoint.")
    parser.add_argument("--output", required=True, help="Output index directory.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="ivfpq")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--m", type=int, default=64, help="PQ sub-vectors.")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--train-sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        for key, value in config.items():
            attr = key.replace("-", "_")
            if getattr(args, attr, None) == parser.get_default(attr):
                setattr(args, attr, value)
    return args


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import faiss

    model_fingerprint = checkpoint_fingerprint(args.model)
    embedder = BioMRAGEmbedder(
        args.model,
        device=args.device or ("cuda" if _cuda_available() else "cpu"),
    )
    row_count = sum(1 for _ in iter_jsonl(args.input))
    if row_count == 0:
        raise ValueError("Knowledge JSONL contains no entries.")

    dimension = int(embedder.model.projection_dim)
    if args.index_type == "ivfpq" and dimension % args.m != 0:
        raise ValueError(
            f"Embedding dimension {dimension} must be divisible by PQ m={args.m}."
        )

    embedding_path = output_dir / (
        "embeddings.npy" if args.save_embeddings else ".embeddings.tmp.npy"
    )
    matrix = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype="float32",
        shape=(row_count, dimension),
    )
    offsets = np.lib.format.open_memmap(
        output_dir / "metadata_offsets.npy",
        mode="w+",
        dtype="uint64",
        shape=(row_count,),
    )

    batch: list[dict] = []
    position = 0
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("wb") as metadata_handle:
        for row in tqdm(
            iter_jsonl(args.input),
            total=row_count,
            desc="Embedding knowledge",
        ):
            batch.append(_validated_item(row).to_dict())
            if len(batch) >= args.batch_size:
                position = _persist_batch(
                    batch,
                    position=position,
                    matrix=matrix,
                    offsets=offsets,
                    metadata_handle=metadata_handle,
                    embedder=embedder,
                    faiss=faiss,
                )
                batch = []
        if batch:
            position = _persist_batch(
                batch,
                position=position,
                matrix=matrix,
                offsets=offsets,
                metadata_handle=metadata_handle,
                embedder=embedder,
                faiss=faiss,
            )

    if position != row_count:
        raise RuntimeError(f"Indexed {position} rows, expected {row_count}.")
    matrix.flush()
    offsets.flush()

    index = _create_index(faiss, matrix, args)
    if not index.is_trained:
        training_matrix = _training_sample(matrix, args)
        index.train(training_matrix)
    for start in tqdm(
        range(0, row_count, args.batch_size),
        desc="Adding FAISS vectors",
    ):
        end = min(start + args.batch_size, row_count)
        index.add(np.ascontiguousarray(matrix[start:end]))

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "index_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dimension": dimension,
                "normalized": True,
                "index_type": (
                    "flat" if type(index).__name__ == "IndexFlatIP" else "ivfpq"
                ),
                "metric": "inner_product",
                "nlist": args.nlist,
                "m": args.m,
                "nbits": args.nbits,
                "nprobe": args.nprobe,
                "rows": row_count,
                "checkpoint_sha256": model_fingerprint,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    del matrix
    if not args.save_embeddings:
        embedding_path.unlink()


def _persist_batch(
    batch: list[dict],
    *,
    position: int,
    matrix,
    offsets,
    metadata_handle,
    embedder,
    faiss,
) -> int:
    vectors = embedder.encode_batch(batch).astype("float32", copy=False)
    faiss.normalize_L2(vectors)
    end = position + len(batch)
    matrix[position:end] = vectors
    for index, item in enumerate(batch, start=position):
        offsets[index] = metadata_handle.tell()
        payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        metadata_handle.write(payload.encode("utf-8") + b"\n")
    return end


def _validated_item(row: dict) -> EvidenceItem:
    item = EvidenceItem.from_dict(row)
    if not item.id:
        raise ValueError("Every indexed knowledge item requires a non-empty id.")
    has_text = bool(item.text.strip())
    has_image = bool(item.image_path.strip())
    expected_modality = (
        "image-text" if has_text and has_image else "image" if has_image else "text"
    )
    if not has_text and not has_image:
        raise ValueError(f"Knowledge item {item.id!r} has no text or image.")
    if item.modality != expected_modality:
        raise ValueError(
            f"Knowledge item {item.id!r} declares modality={item.modality!r}, "
            f"but its fields imply {expected_modality!r}."
        )
    return item


def _validate_args(args) -> None:
    positive = {
        "batch_size": args.batch_size,
        "nlist": args.nlist,
        "m": args.m,
        "nbits": args.nbits,
        "nprobe": args.nprobe,
        "train_sample_size": args.train_sample_size,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Index parameters must be positive: {invalid}")
    if args.nprobe > args.nlist:
        raise ValueError("nprobe cannot exceed nlist.")


def _create_index(faiss, matrix: np.ndarray, args):
    dim = matrix.shape[1]
    if args.index_type == "flat" or len(matrix) < max(args.nlist * 20, 10000):
        return faiss.IndexFlatIP(dim)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        args.nlist,
        args.m,
        args.nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = min(args.nprobe, args.nlist)
    return index


def _training_sample(matrix: np.ndarray, args) -> np.ndarray:
    target = min(
        len(matrix),
        max(args.train_sample_size, args.nlist * 40),
    )
    if target == len(matrix):
        return np.ascontiguousarray(matrix)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(matrix), size=target, replace=False))
    return np.ascontiguousarray(matrix[indices])


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
