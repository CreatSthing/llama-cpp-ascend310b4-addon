#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/layer-cosine-add16}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,17-27}
PROMPT=${PROMPT:-'请用简洁语言解释量化模型如何减少计算量，并比较整数矩阵乘法与浮点矩阵乘法。 Explain why careful numerical validation is required after quantizing a language model.'}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

runner="$BUILD/bin/llama-eval-callback"
common=(-m "$MODEL" -p "$PROMPT" -c 256 -b 1 -ub 1 -ngl 99 -fa 0 --seed 1234)

env LLAMA_LAYER_DUMP="$OUT/fp16.bin" \
    GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$runner" "${common[@]}" > "$OUT/fp16.log" 2>&1

env LLAMA_LAYER_DUMP="$OUT/candidate.bin" \
    GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$runner" "${common[@]}" > "$OUT/candidate.log" 2>&1

python3 "$ROOT/scripts/compare-layer-dumps.py" \
    "$OUT/fp16.bin" "$OUT/candidate.bin" --output "$OUT/layer-cosine.csv" \
    | tee "$OUT/layer-cosine.txt"

echo "completed: $OUT"
