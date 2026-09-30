#!/usr/bin/env bash
set -euo pipefail

args=(
  --index "${INDEX_DIR:-outputs/index}" \
  --model "${RETRIEVER_MODEL:-outputs/retriever_inference}" \
  --queries "${QUERIES_JSONL:-data/eval/vqa_rad_test.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/retrieved/vqa_rad_test.jsonl}" \
  --top-k "${TOP_K:-30}" \
  --nprobe "${NPROBE:-32}"
)

if [[ "${INCLUDE_EMBEDDINGS:-0}" == "1" ]]; then
  args+=(--include-embeddings)
fi

python -m biomrag.retrieval.search_faiss_index "${args[@]}" "$@"
