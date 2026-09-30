#!/usr/bin/env bash
set -euo pipefail

python -m biomrag.evaluation.latency_summary \
  --input "${LATENCY_JSONL:-outputs/eval/latency_records.jsonl}" \
  --output "${OUTPUT_JSON:-outputs/eval/latency_summary.json}" \
  "$@"
