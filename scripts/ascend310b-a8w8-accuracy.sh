#!/usr/bin/env bash
set -euo pipefail

# Accuracy comparison for the Qwen2.5-1.5B 310B4 decode-only A8W8 path.
# PPL uses batched FP16 prefill and a separate ubatch=1 context for the scored
# decode half. Generation speed tests still use b=1/ub=1.

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
MODE=${1:-quick}
OUT=${OUT:-$ROOT/accuracy-a8w8/$MODE}
CORPUS=${CORPUS:-$ROOT/wikitext-2-raw/wiki.test.raw}

case "$MODE" in
    smoke) CTX=128; CHUNKS=1 ;;
    quick) CTX=512; CHUNKS=1 ;;
    full)  CTX=512; CHUNKS=8 ;;
    *) echo "usage: $0 [smoke|quick|full]" >&2; exit 2 ;;
esac

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
if [[ ! -f "$CORPUS" ]]; then
    archive="$ROOT/wikitext-2-raw-v1.zip"
    curl -L "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" -o "$archive"
    (cd "$ROOT" && unzip -o "$archive" >/dev/null)
    rm -f "$archive"
fi

PPL="$BUILD/bin/llama-perplexity"
if [[ ! -x "$PPL" ]]; then
    cmake --build "$BUILD" --target llama-perplexity -j4
fi

common=("$PPL" -m "$MODEL" -f "$CORPUS" -ngl 99 -fa 0 \
        -c "$CTX" -b "$CTX" -ub "$CTX" --chunks "$CHUNKS" --seed 1234)

export LLAMA_PPL_REALISTIC_DECODE=1
base_logits="$OUT/fp16-reference.kld"

echo "[1/3] FP16/NZ reference"
env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on \
    GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "${common[@]}" --kl-divergence-base "$base_logits" \
    2>&1 | tee "$OUT/01-fp16-reference.log"

echo "[2/3] FFN dynamic A8W8, FP16/NZ output"
env GGML_CANN_A8W8=on GGML_CANN_WEIGHT_NZ=on \
    GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "${common[@]}" --kl-divergence-base "$base_logits" --kl-divergence \
    2>&1 | tee "$OUT/02-ffn-a8w8.log"

echo "[3/3] FFN plus tied-output dynamic A8W8"
env GGML_CANN_A8W8=on GGML_CANN_WEIGHT_NZ=on \
    GGML_CANN_TIED_OUTPUT_NZ=off GGML_CANN_TIED_OUTPUT_A8W8=on \
    "${common[@]}" --kl-divergence-base "$base_logits" --kl-divergence \
    2>&1 | tee "$OUT/03-full-a8w8.log"

echo "Results: $OUT"
