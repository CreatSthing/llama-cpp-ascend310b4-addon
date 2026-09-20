param(
    [Parameter(Mandatory = $false)]
    [string]$LlamaCppPath = "."
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$patch = Join-Path $root "patches/ascend310b4-llama-cpp.patch"
$target = (Resolve-Path -LiteralPath $LlamaCppPath).Path

git -C $target rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Target is not a Git repository: $target" }

git -C $target apply --reverse --check $patch
if ($LASTEXITCODE -ne 0) { throw "Reverse patch validation failed. The installed files may have changed." }
git -C $target apply --reverse $patch
if ($LASTEXITCODE -ne 0) { throw "Patch removal failed." }

Write-Host "Ascend 310B4 add-on removed from: $target"
