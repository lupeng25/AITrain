param(
    [string]$WorkDir = ".deps\phase50-paddleocr-v5-gpu-official-chain",
    [string]$DataDir = ".deps\production-ocr-data",
    [string]$Python = "",
    [string]$PaddleOcrRepo = ".deps\repos\PaddleOCR",
    [string]$PaddleOcrRef = "",
    [string]$DetModelPreset = "PP-OCRv5_mobile_det",
    [string]$RecModelPreset = "en_PP-OCRv5_mobile_rec",
    [switch]$UseGpu,
    [switch]$AllowBlocked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow
. (Join-Path $PSScriptRoot "deps-layout.ps1")

function Resolve-RepoPath {
    param([string]$Path)
    return Resolve-AITrainRepoPath -Root $script:Root -Path $Path
}

function Resolve-Python {
    if ($Python) {
        return Resolve-RepoPath $Python
    }
    foreach ($candidate in (Get-AITrainPythonCandidates -Role OcrGpu -Root $script:Root)) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }
    return ""
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-JsonFile {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Write-BlockedSummary {
    param(
        [string]$Reason,
        [object]$Details = $null
    )
    $finishedAt = [DateTime]::UtcNow
    $summary = [ordered]@{
        ok = $false
        status = "blocked"
        reason = $Reason
        scope = "phase50_paddleocr_v5_gpu_official_chain"
        startedAt = $script:StartedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
        workDir = Resolve-RepoPath $WorkDir
        dataDir = Resolve-RepoPath $DataDir
        python = $script:PythonExe
        paddleOcrRepo = Resolve-AITrainPaddleOcrRepo -Root $script:Root -RequestedPath $PaddleOcrRepo
        detModelPreset = $DetModelPreset
        recModelPreset = $RecModelPreset
        useGpu = $script:UseGpuRequested
        blockedReason = $Reason
        details = $Details
        nextAction = "Provision a CUDA-enabled PaddlePaddle/PaddleOCR environment, then rerun this script. Do not downgrade to CPU and mark this GPU gate passed."
    }
    $summaryPath = Join-Path (Resolve-RepoPath $WorkDir) "phase50_paddleocr_v5_gpu_official_chain_summary.json"
    Write-JsonFile -Path $summaryPath -Value $summary
    Write-Host "Phase 50 PP-OCRv5 GPU official chain blocked: $Reason" -ForegroundColor Yellow
    Write-Host "Summary: $summaryPath" -ForegroundColor Cyan
    if ($AllowBlocked) {
        exit 0
    }
    exit 2
}

$workFull = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force $workFull | Out-Null
$script:UseGpuRequested = if ($PSBoundParameters.ContainsKey("UseGpu")) { $UseGpu.IsPresent } else { $true }
$script:PythonExe = Resolve-Python
if ([string]::IsNullOrWhiteSpace($script:PythonExe) -or !(Test-Path -LiteralPath $script:PythonExe)) {
    Write-BlockedSummary -Reason "python_missing" -Details ([ordered]@{ message = "No Python executable found. Provide -Python with a GPU-enabled OCR environment." })
}

if ($script:UseGpuRequested) {
    $probeScript = @"
import json
try:
    import paddle
    payload = {
        "ok": True,
        "paddleVersion": getattr(paddle, "__version__", ""),
        "compiledWithCuda": bool(paddle.device.is_compiled_with_cuda()),
        "device": paddle.device.get_device(),
    }
    if payload["compiledWithCuda"]:
        try:
            paddle.device.set_device("gpu:0")
            payload["deviceAfterSet"] = paddle.device.get_device()
        except Exception as exc:
            payload["ok"] = False
            payload["error"] = str(exc)
    print(json.dumps(payload))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
"@
    $probeScriptPath = Join-Path $workFull "phase50_gpu_probe.py"
    Set-Content -LiteralPath $probeScriptPath -Encoding UTF8 -Value $probeScript
    $probeOutput = & $script:PythonExe $probeScriptPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($probeOutput -join ""))) {
        Write-BlockedSummary -Reason "gpu_probe_failed" -Details ([ordered]@{ output = $probeOutput })
    }
    $probe = ($probeOutput | Select-Object -Last 1) | ConvertFrom-Json
    if (-not $probe.ok -or -not $probe.compiledWithCuda) {
        Write-BlockedSummary -Reason "paddle_gpu_unavailable" -Details $probe
    }
    Write-Host "Paddle GPU probe passed: $($probeOutput -join ' ')" -ForegroundColor Green
}

$chainArgs = @(
    "-WorkDir", $workFull,
    "-DataDir", (Resolve-RepoPath $DataDir),
    "-Python", $script:PythonExe,
    "-PaddleOcrRepo", (Resolve-AITrainPaddleOcrRepo -Root $script:Root -RequestedPath $PaddleOcrRepo),
    "-OcrVersion", "PP-OCRv5",
    "-DetModelPreset", $DetModelPreset,
    "-RecModelPreset", $RecModelPreset
)
if ($PaddleOcrRef) {
    $chainArgs += @("-PaddleOcrRef", $PaddleOcrRef)
}
if ($script:UseGpuRequested) {
    $chainArgs += "-UseGpu"
}
if ($AllowBlocked) {
    $chainArgs += "-AllowBlocked"
}

Write-Host "[phase50] tools\run-production-ocr-official-chain.ps1 $($chainArgs -join ' ')" -ForegroundColor Cyan
$chainCommand = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $script:Root "tools\run-production-ocr-official-chain.ps1")
) + $chainArgs
& powershell.exe @chainCommand
$chainExitCode = $LASTEXITCODE

$chainSummaryPath = Join-Path $workFull "production_ocr_official_chain_summary.json"
$phaseSummaryPath = Join-Path $workFull "phase50_paddleocr_v5_gpu_official_chain_summary.json"
$acceptanceReportPath = Join-Path $workFull "acceptance\production_ocr_acceptance_report.json"
$chainSummaryObject = Read-JsonFile -Path $chainSummaryPath
$acceptanceReportObject = Read-JsonFile -Path $acceptanceReportPath
$chainStatus = ""
$chainOk = $false
$acceptanceStatus = ""
if ($null -ne $chainSummaryObject) {
    $chainStatus = [string]$chainSummaryObject.status
    $chainOk = [bool]$chainSummaryObject.ok
}
if ($null -ne $acceptanceReportObject) {
    $acceptanceStatus = [string]$acceptanceReportObject.status
}
if ([string]::IsNullOrWhiteSpace($chainStatus) -and -not [string]::IsNullOrWhiteSpace($acceptanceStatus)) {
    $chainStatus = $acceptanceStatus
}
if ([string]::IsNullOrWhiteSpace($chainStatus)) {
    $chainStatus = if ($chainExitCode -eq 0) { "passed" } else { "blocked" }
}
$phaseOk = ($chainExitCode -eq 0 -and $chainStatus -eq "passed" -and ([string]::IsNullOrWhiteSpace($acceptanceStatus) -or $acceptanceStatus -eq "passed"))
$status = if ($phaseOk) { "passed" } else { $chainStatus }
$finished = [DateTime]::UtcNow
$summary = [ordered]@{
    ok = $phaseOk
    status = $status
    scope = "phase50_paddleocr_v5_gpu_official_chain"
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finished.ToString("o")
    elapsedSeconds = [Math]::Round(($finished - $script:StartedAt).TotalSeconds, 3)
    workDir = $workFull
    dataDir = Resolve-RepoPath $DataDir
    python = $script:PythonExe
    paddleOcrRepo = Resolve-AITrainPaddleOcrRepo -Root $script:Root -RequestedPath $PaddleOcrRepo
    detModelPreset = $DetModelPreset
    recModelPreset = $RecModelPreset
    useGpu = $script:UseGpuRequested
    productionChainSummary = $chainSummaryPath
    productionChainExitCode = $chainExitCode
    productionChainStatus = $chainStatus
    productionChainOk = $chainOk
    acceptanceStatus = $acceptanceStatus
    note = "PP-OCRv5 GPU gate uses PaddleOCR official Det, Rec, and System reports only. Public data remains workflow evidence, not customer-domain production proof."
}
Write-JsonFile -Path $phaseSummaryPath -Value $summary
Write-Host "Phase 50 PP-OCRv5 GPU official chain summary: $phaseSummaryPath" -ForegroundColor Cyan
if (!$phaseOk -and !$AllowBlocked -and $chainExitCode -eq 0) {
    exit 2
}
exit $chainExitCode
