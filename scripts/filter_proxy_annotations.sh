#!/usr/bin/env bash
set -euo pipefail

args=(
  --input "${ANNOTATIONS_JSONL:-outputs/proxy/teacher_annotations.jsonl}"
  --output "${OUTPUT_JSONL:-outputs/proxy/teacher_annotations.filtered.jsonl}"
  --audit "${AUDIT_JSON:-outputs/proxy/teacher_filter_audit.json}"
  --expected-input-count "${EXPECTED_INPUT_COUNT:-118650}"
  --expected-output-count "${EXPECTED_OUTPUT_COUNT:-118650}"
)

if [[ "${REQUIRE_CHANGE_TERMS:-1}" == "1" ]]; then
  args+=(--require-change-terms)
fi

python -m biomrag.proxy.filter_teacher_annotations "${args[@]}" "$@"
