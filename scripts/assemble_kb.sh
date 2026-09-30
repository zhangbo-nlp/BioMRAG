#!/usr/bin/env bash
set -euo pipefail

read -r -a kb_inputs <<< "${KB_INPUTS:?set KB_INPUTS}"

python -m biomrag.data.assemble_kb \
  --inputs "${kb_inputs[@]}" \
  --output "${OUTPUT_JSONL:-data/kb/raw_kb.jsonl}" \
  --audit "${AUDIT_JSON:-outputs/audit/kb_assembly.json}" \
  --minimum-count "${MINIMUM_COUNT:-26000001}" \
  "$@"
