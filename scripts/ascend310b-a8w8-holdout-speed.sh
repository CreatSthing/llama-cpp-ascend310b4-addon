#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/holdout-4096}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,16-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

for tokens in 64 256; do
    echo "[tg$tokens] FP16/NZ, three repeats, batch=1, ubatch=1"
    env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n "$tokens" -r 3 \
        -b 1 -ub 1 -ngl 99 -fa 0 2>&1 | tee "$OUT/tg$tokens-fp16.log"
    echo "[tg$tokens] selective A8W8, three repeats, batch=1, ubatch=1"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n "$tokens" -r 3 \
        -b 1 -ub 1 -ngl 99 -fa 0 2>&1 | tee "$OUT/tg$tokens-candidate.log"
done
