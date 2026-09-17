#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/holdout-4096-add16}
QUESTIONS=${QUESTIONS:-$ROOT/scripts/ascend310b-accuracy-data/fixed-20.json}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,17-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    GGML_CANN_TIED_OUTPUT_A8W8=off \
    python3 "$ROOT/scripts/ascend310b-fixed20.py" \
    --cli "$BUILD/bin/llama-simple" --model "$MODEL" --questions "$QUESTIONS" \
    --mode fp16 --output "$OUT/fixed20-fp16.json"

env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    python3 "$ROOT/scripts/ascend310b-fixed20.py" \
    --cli "$BUILD/bin/llama-simple" --model "$MODEL" --questions "$QUESTIONS" \
    --mode candidate --output "$OUT/fixed20-candidate.json"

echo "completed: $OUT/fixed20-*.json"
