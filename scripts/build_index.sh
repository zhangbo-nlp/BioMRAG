#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.retrieval.build_faiss_index \
  --config "${INDEX_CONFIG:-configs/indexing.yaml}" \
  --input "${KB_JSONL:-data/kb/decontaminated_kb.jsonl}" \
  --model "${RETRIEVER_MODEL:-outputs/retriever_inference}" \
  --output "${INDEX_DIR:-outputs/index}" \
  "$@"
