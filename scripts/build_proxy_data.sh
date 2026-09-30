#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${RESPONSES_JSONL:-}" && -n "${RATINGS_JSONL:-}" ]]; then
  python -m biomrag.proxy.build_distillation_data \
    --responses "${RESPONSES_JSONL}" \
    --ratings "${RATINGS_JSONL}" \
    --output "${OUTPUT_JSONL:-data/proxy/proxy_train.jsonl}" \
    --expected-count "${EXPECTED_COUNT:-94920}"
elif [[ -n "${ANNOTATIONS_JSONL:-}" ]]; then
  python -m biomrag.proxy.build_distillation_data \
    --annotations "${ANNOTATIONS_JSONL}" \
    --output "${OUTPUT_JSONL:-data/proxy/proxy_train.jsonl}" \
    --expected-count "${EXPECTED_COUNT:-94920}"
else
  python -m biomrag.proxy.build_distillation_data \
    --annotations "${ANNOTATIONS_JSONL:-outputs/proxy/teacher_annotations.release.jsonl}" \
    --output "${OUTPUT_JSONL:-data/proxy/proxy_train.jsonl}" \
    --expected-count "${EXPECTED_COUNT:-94920}"
fi
