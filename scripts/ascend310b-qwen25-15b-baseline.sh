#!/usr/bin/env bash
set -euo pipefail

# Run on the 310B4 board after sourcing CANN set_env.sh.
# Usage: ./scripts/ascend310b-qwen25-15b-baseline.sh MODEL.gguf [LLAMA_BENCH] [OUTPUT_DIR]
model=${1:?Usage: $0 MODEL.gguf [LLAMA_BENCH] [OUTPUT_DIR]}
bench=${2:-./build-cann85/bin/llama-bench}
output_dir=${3:-./qwen25-15b-baseline-$(date +%Y%m%d-%H%M%S)}

if [[ ! -s "$model" ]]; then
    echo "Model file does not exist or is empty: $model" >&2
    exit 1
fi
if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 1
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
model=$(cd "$(dirname "$model")" && pwd)/$(basename "$model")
bench=$(cd "$(dirname "$bench")" && pwd)/$(basename "$bench")

{
    printf 'date=%s\n' "$(date -Is)"
    printf 'model=%s\n' "$model"
    printf 'model_bytes=%s\n' "$(stat -c %s "$model")"
    printf 'bench=%s\n' "$bench"
    printf 'GGML_CANN_WEIGHT_NZ=%s\n' "${GGML_CANN_WEIGHT_NZ:-off}"
    printf 'ASCEND_CUSTOM_OPP_PATH=%s\n' "${ASCEND_CUSTOM_OPP_PATH:-unset}"
    printf 'parameters=ngl:99,fa:off,repetitions:3\n'
    npu-smi info 2>&1 || true
} > "$output_dir/environment.txt"

failures=0

run_case() {
    local name=$1
    local prompt=$2
    local generated=$3
    echo "Running $name (3 repetitions)..."
    if "$bench" -m "$model" -p "$prompt" -n "$generated" -r 3 -ngl 99 -fa off \
        2>&1 | tee "$output_dir/$name.log"; then
        printf '%s=PASS\n' "$name" >> "$output_dir/status.txt"
    else
        local bench_status=${PIPESTATUS[0]}
        printf '%s=FAIL(exit=%s)\n' "$name" "$bench_status" | tee -a "$output_dir/status.txt"
        failures=1
    fi
}

run_case pp128 128 0
run_case pp512 512 0
run_case tg64 0 64
run_case tg256 0 256

echo "Saved baseline results to $output_dir"
exit "$failures"
