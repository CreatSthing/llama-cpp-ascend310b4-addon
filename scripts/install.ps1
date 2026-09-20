param(
    [Parameter(Mandatory = $false)]
    [string]$LlamaCppPath = "."
)

$ErrorActionPreference = "Stop"
$base = "37b3a9e0ccba261d1cc245a971deae0b18c201ab"
$root = Split-Path -Parent $PSScriptRoot
$patch = Join-Path $root "patches/ascend310b4-llama-cpp.patch"
$target = (Resolve-Path -LiteralPath $LlamaCppPath).Path

git -C $target rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Target is not a Git repository: $target" }

$changes = git -C $target status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect target repository." }
if ($changes) { throw "Target llama.cpp worktree is not clean." }

$head = git -C $target rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Cannot read target revision." }
if ($head -ne $base) {
    throw "Unsupported llama.cpp revision: $head. Expected: $base"
}

git -C $target apply --check $patch
if ($LASTEXITCODE -ne 0) { throw "Patch validation failed." }
git -C $target apply $patch
if ($LASTEXITCODE -ne 0) { throw "Patch installation failed." }

Write-Host "Ascend 310B4 add-on installed in: $target"
