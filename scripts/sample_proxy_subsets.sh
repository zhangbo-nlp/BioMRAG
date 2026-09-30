#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.proxy.sample_candidate_subsets \
  --input "${RETRIEVED_JSONL:-outputs/retrieved/proxy_train_candidates.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/proxy/subset_requests.jsonl}" \
  --top-k "${TOP_K:-30}" \
  --random-subsets "${RANDOM_SUBSETS:-10}" \
  --random-min-size "${RANDOM_MIN_SIZE:-5}" \
  --random-max-size "${RANDOM_MAX_SIZE:-5}" \
  --seed "${SEED:-42}" \
  --expected-query-count "${EXPECTED_QUERY_COUNT:-7910}" \
  --expected-output-count "${EXPECTED_OUTPUT_COUNT:-118650}" \
  "$@"
