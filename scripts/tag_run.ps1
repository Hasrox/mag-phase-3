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
    [string]$CudaBin = "",
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

# ggml-cuda.dll is loaded at runtime and resolves cudart/cublas through PATH. If
# the CUDA bin directory is absent the backend fails to load silently, no CUDA
# device registers, and the model comes up entirely on the CPU. That is exactly
# what happened on 2026-09-30: the run looked successful and the gate stopped it.
$Nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
$CudaBin = if ($Nvcc) { Split-Path -Parent $Nvcc.Source } else { "" }
if (-not $CudaBin) {
    # nvcc is only on PATH inside a developer prompt, which this script may not
    # be running in. Find the toolkit by its marker DLL instead.
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $CudaRoot) {
        $Hit = Get-ChildItem $CudaRoot -Directory |
            Where-Object { Test-Path (Join-Path $_.FullName "bin\cudart64_12.dll") } |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($Hit) { $CudaBin = Join-Path $Hit.FullName "bin" }
    }
}
if (-not $CudaBin) { Fail "CUDA toolkit not found. Install CUDA 12.x or pass -CudaBin." }if ($env:PATH -notlike "*$CudaBin*") { $env:PATH = "$CudaBin;$env:PATH" }
if (-not (Get-ChildItem $CudaBin -Filter "cudart64*.dll" -ErrorAction SilentlyContinue)) {
    Fail "no cudart runtime DLL in $CudaBin"
}
Write-Host "CUDA bin: $CudaBin"

# Prove the backend loads before spending 20 minutes on a server that will not
# offload. This is the check that would have caught the silent CPU fallback.
$Devices = (& $Server --list-devices 2>&1 | Out-String)
if ($Devices -notmatch "CUDA\d+") {
    Write-Host $Devices
    Fail "no CUDA device is visible to llama-server. It would load the model on the CPU."
}
Write-Host ($Devices -split "`n" | Where-Object { $_ -match "CUDA\d+" } | Select-Object -First 1)

$Log = Join-Path $Root "runtime\last-server.log"
$MemBefore = [float](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)

Write-Host "starting llama-server on 127.0.0.1:$Port with --n-gpu-layers all"
# -lv 4 is required, not cosmetic. At the default verbosity of 3 this build
# never prints the "offloaded N/N layers to GPU" line, so a correct run looks
# identical to a CPU one and the gate cannot tell them apart.
$Proc = Start-Process -FilePath $Server -PassThru -NoNewWindow -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -ArgumentList @(
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--model", (Resolve-Path $Model).Path,
    "--mmproj", (Resolve-Path $Mmproj).Path,
    "--n-gpu-layers", "all",
    "--parallel", "1",
    "--ctx-size", "$Ctx",
    # The tagger sends each image as a file:// URL. Without --media-path this
    # server rejects every request with 400 "file:// URLs are not allowed".
    # The flag takes one directory, so the media root is assets/ and paths are
    # resolved relative to it. Nothing above assets/ is reachable.
    "--media-path", (Join-Path $Root "assets"),
    # Keeps the answer in message.content. Without it this chat template returns
    # the model's scratchpad there instead, and the tagger reads an empty string.
    "--skip-chat-parsing",
    "-lv", "4"
)

try {
    $Offloaded = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        $text = (Get-Content $Log, "$Log.err" -ErrorAction SilentlyContinue | Out-String)
        if ($text -match "offloaded \d+/\d+ layers to GPU") { $Offloaded = $true; break }
        if ($Proc.HasExited) { Fail "server exited; see $Log.err" }
        # A CPU run finishes loading fast and never prints an offload line.
        # Detect that and say so instead of waiting out the timeout.
        if ($text -match "listening on http://" -and $text -notmatch "using device CUDA") {
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            Fail "the server finished loading on the CPU: no CUDA device was selected. See $Log.err"
        }
    }
    if (-not $Offloaded) {
        Fail "no offload line in the log after 180s. Partial offload is not a workaround. See $Log.err"
    }

    $MemAfter = [float](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)

    # Health and probe results go in a sidecar. The server holds runtime\
    # last-server.log open for writing, so appending to it fails.
    $Probe = Join-Path $Root "runtime\last-probe.txt"
    "" | Out-File -FilePath $Probe -Encoding utf8

    $Tps = 0.0
    # The offload line appears before the context and KV cache are ready, so the
    # server answers 503 "Loading model" for a few seconds afterwards. Poll
    # /health until it reports ok rather than assuming it is up.
    $Ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
            if ($Health.status -eq "ok") { $Ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if ($Ready) {
        Add-Content -Path $Probe -Value "health ok status=ok"
    } else {
        Add-Content -Path $Probe -Value "health failed: server stayed unavailable"
    }
    try {
        $Body = @{ model = "probe"; max_tokens = 32; temperature = 0.0; messages = @(
            @{ role = "user"; content = "reply with the single word ok" }) } | ConvertTo-Json -Depth 6
        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        $Reply = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -Body $Body -ContentType "application/json" -TimeoutSec 180
        $Sw.Stop()
        if ($Sw.Elapsed.TotalSeconds -gt 0) { $Tps = [math]::Round(32 / $Sw.Elapsed.TotalSeconds, 2) }
        $Text = [string]$Reply.choices[0].message.content
        Add-Content -Path $Probe -Value "probe json ok tps=$Tps reply=$($Text.Trim())"
    } catch {
        Add-Content -Path $Probe -Value "probe failed: $_"
    }

    $Commit = "unpinned"
    $CommitFile = Join-Path $Root "runtime\llama.cpp-commit.txt"
    if (Test-Path $CommitFile) { $Commit = (Get-Content $CommitFile).Trim() }

    $ProbeText = (Get-Content $Probe -ErrorAction SilentlyContinue | Out-String)
    $HealthOk = $ProbeText -match "health ok"
    $ProbeOk = $ProbeText -match "probe json ok"
    Write-Host $ProbeText
    if (-not $HealthOk -or -not $ProbeOk) { Fail "health or probe did not pass; see $Probe" }

    Push-Location $Root
    # llama-server writes its log to stderr, not stdout. Point the gate at the
    # file that actually has the offload evidence.
    python -m mag.gatecli --model $Model --mmproj $Mmproj --log "$Log.err" `
        --mem-before $MemBefore --mem-after $MemAfter --tps $Tps --pid $Proc.Id --commit $Commit `
        --health-ok --probe-ok
    if ($LASTEXITCODE -ne 0) { Fail "gate did not pass; nothing was tagged" }

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
