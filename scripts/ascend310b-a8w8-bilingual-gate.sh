#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/bilingual-256}
EN=${EN:-$ROOT/wikitext-2-raw/wiki.test.raw}
ZH=${ZH:-$ROOT/scripts/ascend310b-accuracy-data/zh-quick.txt}
QUESTIONS=${QUESTIONS:-$ROOT/scripts/ascend310b-accuracy-data/fixed-20.json}
DISABLED_LAYERS=${DISABLED_LAYERS:-0-7,13,16-27}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

cmake --build "$BUILD" --target llama-perplexity llama-simple llama-bench -j4
ppl="$BUILD/bin/llama-perplexity"
export LLAMA_PPL_REALISTIC_DECODE=1
common=(-ngl 99 -fa 0 -c 514 -b 514 -ub 514 --chunks 1 --seed 1234)
cat "$ZH" "$ZH" "$ZH" > "$OUT/zh-expanded.txt"

for lang in en zh; do
    if [[ "$lang" == en ]]; then corpus="$EN"; else corpus="$OUT/zh-expanded.txt"; fi
    ref="$OUT/$lang-fp16.kld"
    echo "[$lang] FP16 reference, batch=1, ubatch=1"
    env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" -m "$MODEL" -f "$corpus" "${common[@]}" --kl-divergence-base "$ref" \
        2>&1 | tee "$OUT/$lang-fp16.log"

    echo "[$lang] selective A8W8, FP16 layers $DISABLED_LAYERS, batch=1, ubatch=1"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
        "$ppl" -m "$MODEL" -f "$corpus" "${common[@]}" \
        --kl-divergence-base "$ref" --kl-divergence \
        2>&1 | tee "$OUT/$lang-candidate.log"
done

env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    python3 "$ROOT/scripts/ascend310b-fixed20.py" \
    --cli "$BUILD/bin/llama-simple" --model "$MODEL" --questions "$QUESTIONS" \
    --mode fp16 --output "$OUT/fixed20-fp16.json"
env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    python3 "$ROOT/scripts/ascend310b-fixed20.py" \
    --cli "$BUILD/bin/llama-simple" --model "$MODEL" --questions "$QUESTIONS" \
    --mode candidate --output "$OUT/fixed20-candidate.json"

echo "[speed] FP16/NZ"
env GGML_CANN_A8W8=off GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
    "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n 64 -r 3 -b 1 -ub 1 -ngl 99 -fa 0 \
    2>&1 | tee "$OUT/speed-fp16.log"
echo "[speed] selective A8W8"
env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$DISABLED_LAYERS" \
    GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on GGML_CANN_TIED_OUTPUT_A8W8=off \
    "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n 64 -r 3 -b 1 -ub 1 -ngl 99 -fa 0 \
    2>&1 | tee "$OUT/speed-candidate.log"

echo "Results: $OUT"
