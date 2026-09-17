#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/holdout-4096-add16}
EN=${EN:-$ROOT/wikitext-2-raw/wiki.valid.raw}
ZH=${ZH:-$ROOT/accuracy-a8w8/ceval-val-corpus.txt}
EN_REFERENCE=${EN_REFERENCE:-$ROOT/accuracy-a8w8/holdout-4096-add16-add18/en-fp16.kld}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,17-27}
SKIP_EN=${SKIP_EN:-0}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
test -s "$EN_REFERENCE"

ppl="$BUILD/bin/llama-perplexity"
export LLAMA_PPL_REALISTIC_DECODE=1
common=(-m "$MODEL" -ngl 99 -fa 0 -c 514 -b 514 -ub 514 --chunks 16 --seed 1234)

if [[ "$SKIP_EN" != 1 ]]; then
    echo "[en] selective A8W8 against completed FP16 reference"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" "${common[@]}" -f "$EN" \
        --kl-divergence-base "$EN_REFERENCE" --kl-divergence \
        2>&1 | tee "$OUT/en-candidate.log"
fi

zh_ref="$OUT/zh-fp16.kld"
echo "[zh] FP16 reference"
env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$ppl" "${common[@]}" -f "$ZH" --kl-divergence-base "$zh_ref" \
    2>&1 | tee "$OUT/zh-fp16.log"

echo "[zh] selective A8W8"
env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$ppl" "${common[@]}" -f "$ZH" \
    --kl-divergence-base "$zh_ref" --kl-divergence \
    2>&1 | tee "$OUT/zh-candidate.log"

echo "completed: $OUT"
