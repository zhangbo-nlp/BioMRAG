#!/usr/bin/env bash
set -euo pipefail

args=(
  --input "${INPUT_JSONL:-data/proxy/proxy_train.jsonl}"
  --output "${OUTPUT_JSONL:-data/proxy/proxy_train.swift.jsonl}"
  --val-output "${VAL_OUTPUT_JSONL:-data/proxy/proxy_val.swift.jsonl}"
  --val-ratio "${VAL_RATIO:-0.02}"
  --seed "${SEED:-42}"
  --expected-count "${EXPECTED_COUNT:-94920}"
)

if [[ "${DROP_MISSING_IMAGES:-0}" == "1" ]]; then
  args+=(--drop-missing-images)
fi

python -m biomrag.proxy.prepare_swift_sft_data "${args[@]}"
