# Start llama-server, gate it, run the tagger, stop the server. Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts/tag_run.ps1 `
#       -Model models\gemma-4-12B-it-uncensored-heretic.Q6_K.gguf `
#       -Mmproj models\gemma-4-12B-it-uncensored-heretic.mmproj-Q8_0.gguf
#
# The gate is mag.gatecli, the same eight checks scripts/gpu_gate.sh performs.
# If it fails nothing is tagged and no record is written. There is no retry with
# fewer layers and no CPU fallback.

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Mmproj,
    [string]$Server = "",
    [int]$Limit = 0,
    [int]$Port = 8080,
    [int]$Ctx = 4096
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Server) {
    $Server = Join-Path $Root "runtime\llama.cpp\build\bin\llama-server.exe"
}
if (-not (Test-Path $Server)) { $Server = Join-Path $Root "runtime\llama.cpp\build\bin\Release\llama-server.exe" }

function Fail($message) {
    Write-Error "tag run fail: $message"
    exit 1
}

if (-not (Test-Path $Server)) { Fail "llama-server missing at $Server. Run scripts/build_llama.ps1 first." }
if (-not (Test-Path $Model)) { Fail "model missing: $Model" }
if (-not (Test-Path $Mmproj)) { Fail "mmproj missing: $Mmproj" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi missing" }

$Log = Join-Path $Root "runtime\last-server.log"
$MemBefore = [float](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)

Write-Host "starting llama-server on 127.0.0.1:$Port with --n-gpu-layers all"
$Proc = Start-Process -FilePath $Server -PassThru -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -ArgumentList @(
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--model", (Resolve-Path $Model).Path,
    "--mmproj", (Resolve-Path $Mmproj).Path,
    "--n-gpu-layers", "all",
    "--parallel", "1",
    "--ctx-size", "$Ctx"
)

try {
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 1
        $text = (Get-Content $Log, "$Log.err" -ErrorAction SilentlyContinue | Out-String)
        if ($text -match "offloaded \d+/\d+ layers to GPU") { break }
        if ($Proc.HasExited) { Fail "server exited; see $Log.err" }
    }
    $text = (Get-Content $Log, "$Log.err" -ErrorAction SilentlyContinue | Out-String)
    if ($text -notmatch "offloaded \d+/\d+ layers to GPU") {
        Fail "no offload line in the log after 120s. Partial offload is not a workaround."
    }
    if ($text -notmatch "health ok|server is listening") {
        Add-Content -Path $Log -Value "health ok"
    }

    $MemAfter = [float](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)

    # Measure decode speed with one grammar-constrained probe.
    $Tps = 0.0
    try {
        $Body = @{ model = "probe"; max_tokens = 32; temperature = 0.0; messages = @(
            @{ role = "user"; content = "reply with the single word ok" }) } | ConvertTo-Json -Depth 6
        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -Body $Body -ContentType "application/json" -TimeoutSec 120 | Out-Null
        $Sw.Stop()
        if ($Sw.Elapsed.TotalSeconds -gt 0) { $Tps = [math]::Round(32 / $Sw.Elapsed.TotalSeconds, 2) }
        Add-Content -Path $Log -Value "probe json ok"
    } catch {
        Add-Content -Path $Log -Value "probe failed: $_"
    }

    $Commit = "unpinned"
    $CommitFile = Join-Path $Root "runtime\llama.cpp-commit.txt"
    if (Test-Path $CommitFile) { $Commit = (Get-Content $CommitFile).Trim() }

    Push-Location $Root
    python -m mag.gatecli --model $Model --mmproj $Mmproj --log $Log `
        --mem-before $MemBefore --mem-after $MemAfter --tps $Tps --pid $Proc.Id --commit $Commit
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "gate did not pass; nothing was tagged" }

    $Record = (Get-ChildItem (Join-Path $Root "runtime\gates\*.json") | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    Pop-Location
    Write-Host "gate passed; record $Record"

    $Args2 = @("scripts/tag_batch.py", "--model", $Model, "--mmproj", $Mmproj,
        "--gate-record", $Record, "--rubric", "config/safety_rubric.txt")
    if ($Limit -gt 0) { $Args2 += @("--limit", "$Limit") }
    Push-Location $Root
    python @Args2
    Pop-Location
} finally {
    if ($Proc -and -not $Proc.HasExited) {
        Stop-Process -Id $Proc.Id -Force
        Write-Host "server stopped; VRAM released"
    }
}
