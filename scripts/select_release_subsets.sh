#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.proxy.select_release_subsets \
  --input "${ANNOTATIONS_JSONL:-outputs/proxy/teacher_annotations.filtered.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/proxy/teacher_annotations.release.jsonl}" \
  --audit "${AUDIT_JSON:-outputs/proxy/release_selection_audit.json}" \
  --keep-per-query "${KEEP_PER_QUERY:-12}" \
  --expected-input-count "${EXPECTED_INPUT_COUNT:-118650}" \
  --expected-output-count "${EXPECTED_OUTPUT_COUNT:-94920}" \
  "$@"
