#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BLOCKED_SPLITS:-}" ]]; then
  read -r -a blocked_splits <<< "${BLOCKED_SPLITS}"
else
  blocked_splits=(
    data/held_out/mimic_cxr_validation.jsonl
    data/held_out/pmc_oa_validation.jsonl
    data/held_out/roco_validation.jsonl
    data/held_out/llava_med_align_validation.jsonl
    data/held_out/llava_med_inst_validation.jsonl
    data/held_out/llava_med_inst_test.jsonl
    data/held_out/vqa_rad_test.jsonl
    data/held_out/slake_validation.jsonl
    data/held_out/slake_test.jsonl
    data/held_out/pathvqa_validation.jsonl
    data/held_out/pathvqa_test.jsonl
  )
fi

args=(
  --kb "${KB_JSONL:-data/kb/raw_kb.jsonl}"
  --blocked "${blocked_splits[@]}"
  --output "${OUTPUT_JSONL:-data/kb/decontaminated_kb.jsonl}"
  --audit "${AUDIT_JSON:-outputs/audit/decontamination.json}"
  --caption-semantic-matches "${CAPTION_SEMANTIC_MATCHES_JSONL:-data/audit/caption_report_semantic_matches.jsonl}"
  --qa-semantic-matches "${QA_SEMANTIC_MATCHES_JSONL:-data/audit/question_answer_semantic_matches.jsonl}"
  --manual-exclusions "${MANUAL_EXCLUSIONS_JSONL:-data/audit/manual_exclusions.jsonl}"
  --minimum-kept "${MINIMUM_KEPT:-26000001}"
)

python -m biomrag.data.decontaminate_index "${args[@]}" "$@"
