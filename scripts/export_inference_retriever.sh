#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.retrieval.export_inference_retriever \
  --input "${TRAINED_RETRIEVER:-outputs/retriever}" \
  --output "${OUTPUT_RETRIEVER:-outputs/retriever_inference}" \
  "$@"
