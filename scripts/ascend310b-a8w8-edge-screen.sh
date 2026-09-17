#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/edge-screen}
CORPUS=${CORPUS:-$ROOT/wikitext-2-raw/wiki.valid.raw}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

ppl="$BUILD/bin/llama-perplexity"
export LLAMA_PPL_REALISTIC_DECODE=1
common=(-m "$MODEL" -f "$CORPUS" -ngl 99 -fa 0 -c 258 -b 258 -ub 258 --chunks 1 --seed 1234)
ref="$OUT/fp16.kld"

env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$ppl" "${common[@]}" --kl-divergence-base "$ref" 2>&1 | tee "$OUT/fp16.log"

# Current seven-layer candidate plus exactly one extra A8W8 layer per trial.
names=(current add13 add16 add17 add18 add19)
disabled=(
    '0-7,13,16-27'
    '0-7,16-27'
    '0-7,13,17-27'
    '0-7,13,16,18-27'
    '0-7,13,16-17,19-27'
    '0-7,13,16-18,20-27'
)

for i in "${!names[@]}"; do
    echo "[${names[$i]}] FP16 layers ${disabled[$i]}"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="${disabled[$i]}" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" --kl-divergence-base "$ref" --kl-divergence \
        2>&1 | tee "$OUT/${names[$i]}.log"
done

echo "completed: $OUT"
