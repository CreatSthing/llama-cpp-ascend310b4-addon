#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/HwHiAiUser/llama-cpp-310b}
BUILD=${BUILD:-$ROOT/build-a8w8}
MODEL=${MODEL:-$ROOT/qwen2.5-1.5b-instruct-f16.gguf}
CORPUS=${CORPUS:-$ROOT/wikitext-2-raw/wiki.test.raw}
CUSTOM_OPP=${CUSTOM_OPP:-/home/HwHiAiUser/a8w8-probe/install_a8w8_nz/vendors/customize}
OUT=${OUT:-$ROOT/accuracy-a8w8/layer-sensitivity}
BASE=${BASE:-$ROOT/accuracy-a8w8/smoke/fp16-reference.kld}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"

if [[ ! -f "$BASE" ]]; then
    echo "missing FP16 reference: $BASE" >&2
    echo "run scripts/ascend310b-a8w8-accuracy.sh smoke first" >&2
    exit 1
fi

common=("$BUILD/bin/llama-perplexity" -m "$MODEL" -f "$CORPUS" -ngl 99 -fa 0 \
        -c 128 -b 1 -ub 1 --chunks 1 --seed 1234 \
        --kl-divergence-base "$BASE" --kl-divergence)

for layers in 0-3 4-7 8-11 12-15 16-19 20-23 24-27; do
    echo "Testing FP16 fallback for layers $layers"
    env GGML_CANN_A8W8=on GGML_CANN_A8W8_DISABLE_LAYERS="$layers" \
        GGML_CANN_WEIGHT_NZ=on GGML_CANN_TIED_OUTPUT_NZ=on \
        GGML_CANN_TIED_OUTPUT_A8W8=off \
        "${common[@]}" 2>&1 | tee "$OUT/layers-$layers.log"
done

python3 - "$OUT" <<'PY'
import pathlib, re, sys
print("\nblock,mean_kld,same_top_percent")
for path in sorted(pathlib.Path(sys.argv[1]).glob("layers-*.log")):
    text = path.read_text(errors="replace")
    kld = re.search(r"Mean\s+KLD:\s+([0-9.eE+-]+)", text)
    top = re.search(r"Same top p:\s+([0-9.eE+-]+)", text)
    print(f"{path.stem.removeprefix('layers-')},{kld.group(1) if kld else 'NA'},{top.group(1) if top else 'NA'}")
PY
