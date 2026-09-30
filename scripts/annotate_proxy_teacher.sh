#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.proxy.teacher_annotation \
  --input "${INPUT_JSONL:-outputs/proxy/teacher_requests.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/proxy/teacher_annotations.jsonl}" \
  --model "${TEACHER_MODEL:-gpt-4o}" \
  --expected-count "${EXPECTED_COUNT:-118650}" \
  "$@"
