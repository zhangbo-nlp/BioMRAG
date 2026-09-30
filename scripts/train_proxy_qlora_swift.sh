#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
proxy_seed="${PROXY_SEED:-21}"

if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI not found. Install the proxy training extra before running this script." >&2
  exit 127
fi

python -m biomrag.proxy.validate_swift_config \
  --config "${CONFIG:-configs/proxy_qlora_swift.yaml}" \
  --world-size "${NPROC_PER_NODE}" \
  --check-runtime

swift sft \
  --config "${CONFIG:-configs/proxy_qlora_swift.yaml}" \
  --seed "${proxy_seed}" \
  --data_seed "${proxy_seed}" \
  --output_dir "${OUTPUT_DIR:-outputs/proxy/qwen2_5_vl_7b_qlora_seed${proxy_seed}}" \
  "$@"
