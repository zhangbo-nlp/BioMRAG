#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.data.prepare_text_passages \
  --input "${INPUT_JSONL:?set INPUT_JSONL}" \
  --output "${OUTPUT_JSONL:?set OUTPUT_JSONL}" \
  --audit "${AUDIT_JSON:?set AUDIT_JSON}" \
  --source "${SOURCE:?set SOURCE}" \
  --target-tokens "${TARGET_TOKENS:-200}" \
  --overlap-tokens "${OVERLAP_TOKENS:-0}" \
  "$@"
