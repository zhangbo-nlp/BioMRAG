#!/usr/bin/env bash
set -euo pipefail

args=(
  --input "${RETRIEVED_JSONL:-outputs/retrieved/vqa_rad_test.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/selected/vqa_rad_test.jsonl}" \
  --budget "${EVIDENCE_BUDGET:-5}" \
  --expected-candidates "${EXPECTED_CANDIDATES:-30}" \
  --proxy-model "${PROXY_MODEL:-biomrag-proxy-qwen2.5-vl-7b}" \
  --retriever-model "${RETRIEVER_MODEL:-outputs/retriever_inference}"
)

python -m biomrag.selection.run_selection "${args[@]}" "$@"
