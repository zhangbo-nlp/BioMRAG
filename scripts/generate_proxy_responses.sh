#!/usr/bin/env bash
set -euo pipefail

input="${SUBSET_REQUESTS_JSONL:-outputs/proxy/subset_requests.jsonl}"
model="${RESPONSE_BACKBONE:-Qwen/Qwen2.5-VL-7B-Instruct}"
api_args=()
if [[ -n "${RESPONSE_API_BASE_URL:-}" ]]; then
  api_args+=(--base-url "${RESPONSE_API_BASE_URL}")
fi
if [[ -n "${RESPONSE_API_KEY:-}" ]]; then
  api_args+=(--api-key "${RESPONSE_API_KEY}")
fi

python -m biomrag.generation.run_generation \
  --input "${input}" \
  --output "${EVIDENCE_RESPONSES_JSONL:-outputs/proxy/evidence_responses.jsonl}" \
  --model "${model}" \
  --expected-count "${EXPECTED_EVIDENCE_COUNT:-118650}" \
  "${api_args[@]}"

python -m biomrag.generation.run_generation \
  --input "${input}" \
  --output "${NO_EVIDENCE_RESPONSES_JSONL:-outputs/proxy/no_evidence_responses.jsonl}" \
  --model "${model}" \
  --no-retrieval \
  --expected-count "${EXPECTED_NO_EVIDENCE_COUNT:-7910}" \
  "${api_args[@]}"

bash scripts/merge_proxy_responses.sh
