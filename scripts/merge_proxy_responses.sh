#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.proxy.merge_paired_responses \
  --evidence-responses "${EVIDENCE_RESPONSES_JSONL:-outputs/proxy/evidence_responses.jsonl}" \
  --no-evidence-responses "${NO_EVIDENCE_RESPONSES_JSONL:-outputs/proxy/no_evidence_responses.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/proxy/teacher_requests.jsonl}" \
  --expected-evidence-count "${EXPECTED_EVIDENCE_COUNT:-118650}" \
  --expected-no-evidence-count "${EXPECTED_NO_EVIDENCE_COUNT:-7910}" \
  "$@"
