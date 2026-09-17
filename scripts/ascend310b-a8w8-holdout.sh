#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/holdout-4096}
EN=${EN:-$ROOT/wikitext-2-raw/wiki.valid.raw}
ZH=${ZH:-$ROOT/accuracy-a8w8/ceval-val-corpus.txt}
CONTEXT=${CONTEXT:-514}
CHUNKS=${CHUNKS:-16}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,16-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

for corpus in "$EN" "$ZH"; do test -s "$corpus"; done
ppl="$BUILD/bin/llama-perplexity"
export LLAMA_PPL_REALISTIC_DECODE=1
common=(-m "$MODEL" -ngl 99 -fa 0 -c "$CONTEXT" -b "$CONTEXT" -ub "$CONTEXT" --chunks "$CHUNKS" --seed 1234)

for lang in en zh; do
    if [[ "$lang" == en ]]; then corpus="$EN"; else corpus="$ZH"; fi
    ref="$OUT/$lang-fp16.kld"
    echo "[$lang] FP16, $CONTEXT context, $CHUNKS chunks, batch=1, ubatch=1"
    env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" -f "$corpus" --kl-divergence-base "$ref" \
        2>&1 | tee "$OUT/$lang-fp16.log"

    echo "[$lang] selective A8W8, FP16 layers $DISABLED_LAYERS, batch=1, ubatch=1"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" -f "$corpus" \
        --kl-divergence-base "$ref" --kl-divergence \
        2>&1 | tee "$OUT/$lang-candidate.log"
done

echo "completed: $OUT"
