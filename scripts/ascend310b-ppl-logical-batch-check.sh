#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
CORPUS=${CORPUS:-$ROOT/accuracy-a8w8/ceval-val-corpus.txt}
OUT=${OUT:-$ROOT/accuracy-a8w8/logical-batch-check}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,17-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

ppl="$BUILD/bin/llama-perplexity"
# n_batch is only the logical submission size. n_ubatch=1 keeps every physical
# graph at one token, which is required by the 310B A8W8 decode operator.
common=(-m "$MODEL" -f "$CORPUS" -ngl 99 -fa 0 -c 514 -b 514 -ub 1 --chunks 1 --seed 1234)

env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$ppl" "${common[@]}" --kl-divergence-base "$OUT/fp16.kld" \
    2>&1 | tee "$OUT/fp16.log"

env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$ppl" "${common[@]}" --kl-divergence-base "$OUT/fp16.kld" --kl-divergence \
    2>&1 | tee "$OUT/candidate.log"

echo "completed: $OUT"
