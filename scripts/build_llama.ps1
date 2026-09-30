# Build llama.cpp with CUDA. Windows. Fails loudly rather than half-succeeding.
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_llama.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/build_llama.ps1 -Commit b1234
#
# Requires, on PATH: cmake, nvcc, and the MSVC vcvars environment. A CPU-only or
# partially offloaded build is discarded: the gate refuses it, and there is no
# fallback.

param(
    [string]$Commit = "",
    [string]$Source = "https://github.com/ggml-org/llama.cpp"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "runtime\llama.cpp"

function Fail($message) {
    Write-Error "build fail: $message"
    exit 1
}

foreach ($tool in @("cmake", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "$tool is not on PATH. Install CMake and the VS 2022 Build Tools C++ workload."
    }
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Fail "nvcc is not on PATH. Install the CUDA Toolkit (12.x) and reopen the terminal."
}

if (-not (Test-Path $Dest)) {
    Write-Host "cloning $Source"
    git clone --filter=blob:none $Source $Dest
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
}
if ($Commit) {
    Push-Location $Dest
    git fetch --all --tags
    git checkout $Commit
    Pop-Location
}
Push-Location $Dest
$Actual = (git rev-parse HEAD).Trim()
Pop-Location

$Build = Join-Path $Dest "build"
Write-Host "configuring with GGML_CUDA=ON GGML_NATIVE=OFF at $Actual"
cmake -S $Dest -B $Build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed" }

cmake --build $Build --config Release --target llama-server -j
if ($LASTEXITCODE -ne 0) { Fail "cmake build failed" }

$Server = Join-Path $Build "bin\llama-server.exe"
if (-not (Test-Path $Server)) {
    $Server = Join-Path $Build "bin\Release\llama-server.exe"
}
if (-not (Test-Path $Server)) { Fail "llama-server.exe not found after the build" }

# A CPU-only build would still produce a binary. Refuse it here, not at the gate.
$Help = & $Server --help 2>&1 | Out-String
if ($Help -notmatch "gpu-layers") { Fail "binary has no GPU layer flag; discarded" }

Set-Content -Path (Join-Path $Root "runtime\llama.cpp-commit.txt") -Value $Actual
Write-Host "built $Server"
Write-Host "commit $Actual recorded in runtime/llama.cpp-commit.txt"
Write-Host "next: scripts/tag_run.ps1 -Model models\<model>.gguf -Mmproj models\<mmproj>.gguf"
