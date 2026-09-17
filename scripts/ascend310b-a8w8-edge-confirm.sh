#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/edge-confirm-add16}
EN=${EN:-$ROOT/wikitext-2-raw/wiki.valid.raw}
ZH=${ZH:-$ROOT/accuracy-a8w8/ceval-val-corpus.txt}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,17-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

ppl="$BUILD/bin/llama-perplexity"
export LLAMA_PPL_REALISTIC_DECODE=1
common=(-m "$MODEL" -ngl 99 -fa 0 -c 514 -b 514 -ub 514 --chunks 1 --seed 1234)

for lang in en zh; do
    if [[ "$lang" == en ]]; then corpus="$EN"; else corpus="$ZH"; fi
    ref="$OUT/$lang-fp16.kld"
    env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" -f "$corpus" --kl-divergence-base "$ref" \
        2>&1 | tee "$OUT/$lang-fp16.log"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" -f "$corpus" --kl-divergence-base "$ref" --kl-divergence \
        2>&1 | tee "$OUT/$lang-candidate.log"
done

for tokens in 64 256; do
    env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n "$tokens" -r 3 \
        -b 1 -ub 1 -ngl 99 -fa 0 2>&1 | tee "$OUT/tg$tokens-fp16.log"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n "$tokens" -r 3 \
        -b 1 -ub 1 -ngl 99 -fa 0 2>&1 | tee "$OUT/tg$tokens-candidate.log"
done

echo "completed: $OUT"
