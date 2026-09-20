#!/usr/bin/env bash
set -euo pipefail

target=${1:-.}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
patch="$root/patches/ascend310b4-llama-cpp.patch"

git -C "$target" rev-parse --is-inside-work-tree >/dev/null
git -C "$target" apply --reverse --check "$patch"
git -C "$target" apply --reverse "$patch"
echo "Ascend 310B4 add-on removed from: $(cd "$target" && pwd)"
