#!/usr/bin/env bash
set -euo pipefail

adapter="${PROXY_ADAPTER:?set PROXY_ADAPTER to an ms-swift checkpoint directory}"

if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI not found. Install the proxy training extra first." >&2
  exit 127
fi

python -m biomrag.proxy.validate_swift_config \
  --config "${CONFIG:-configs/proxy_qlora_swift.yaml}" \
  --world-size 2 \
  --check-runtime

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

swift deploy \
  --model "${PROXY_BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --adapters "${adapter}" \
  --infer_backend pt \
  --temperature 0 \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --served_model_name "${SERVED_MODEL_NAME:-biomrag-proxy-qwen2.5-vl-7b}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  "$@"
