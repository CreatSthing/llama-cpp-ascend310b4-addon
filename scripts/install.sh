#!/usr/bin/env bash
set -euo pipefail

target=${1:-.}
base=37b3a9e0ccba261d1cc245a971deae0b18c201ab
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
patch="$root/patches/ascend310b4-llama-cpp.patch"

git -C "$target" rev-parse --is-inside-work-tree >/dev/null

if [[ -n $(git -C "$target" status --porcelain) ]]; then
    echo "Target llama.cpp worktree is not clean." >&2
    exit 1
fi

head=$(git -C "$target" rev-parse HEAD)
if [[ "$head" != "$base" ]]; then
    echo "Unsupported llama.cpp revision: $head" >&2
    echo "Expected: $base" >&2
    exit 1
fi

git -C "$target" apply --check "$patch"
git -C "$target" apply "$patch"
echo "Ascend 310B4 add-on installed in: $(cd "$target" && pwd)"
