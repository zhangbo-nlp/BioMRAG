#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.generation.run_generation \
  --input "${SELECTED_JSONL:-outputs/selected/vqa_rad_test.jsonl}" \
  --output "${OUTPUT_JSONL:-outputs/predictions/vqa_rad_test.jsonl}" \
  --model "${GENERATOR_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct}" \
  --expected-count "${EXPECTED_COUNT:-451}" \
  "$@"
