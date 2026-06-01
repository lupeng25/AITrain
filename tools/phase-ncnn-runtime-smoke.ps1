param(
    [string]$NcnnRoot = $env:AITRAIN_NCNN_ROOT,
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [Parameter(Mandatory = $true)]
    [string]$SampleImagePath,
    [string]$OutputDir = ".deps\ncnn-runtime-smoke",
    [ValidateSet("detection", "segmentation")]
    [string]$TaskType = "detection",
    [string]$WorkerExe = ".\build-vscode\bin\aitrain_worker.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Summary {
    param([hashtable]$Summary)
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $summaryPath = Join-Path $OutputDir "ncnn_runtime_smoke_summary.json"
    $Summary.generatedAt = (Get-Date).ToUniversalTime().ToString("o")
    $Summary.outputDir = (Resolve-Path -LiteralPath $OutputDir).Path
    $Summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host ("NCNN runtime smoke summary: {0}" -f $summaryPath)
}

function Resolve-Onnx2Ncnn {
    param([string]$Root)
    if ([string]::IsNullOrWhiteSpace($Root)) {
        return $null
    }
    $candidates = @(
        (Join-Path $Root "bin\onnx2ncnn.exe"),
        (Join-Path $Root "tools\onnx\onnx2ncnn.exe"),
        (Join-Path $Root "x64\bin\onnx2ncnn.exe"),
        (Join-Path $Root "x64\tools\onnx\onnx2ncnn.exe"),
        (Join-Path $Root "onnx2ncnn.exe"),
        (Join-Path $Root "bin\onnx2ncnn"),
        (Join-Path $Root "tools\onnx\onnx2ncnn"),
        (Join-Path $Root "x64\bin\onnx2ncnn"),
        (Join-Path $Root "x64\tools\onnx\onnx2ncnn"),
        (Join-Path $Root "onnx2ncnn")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (!(Test-Path -LiteralPath $WorkerExe)) {
    Write-Summary @{
        ok = $false
        status = "blocked"
        stage = "worker"
        error = "Worker executable not found: $WorkerExe"
    }
    exit 2
}
if (!(Test-Path -LiteralPath $OnnxPath)) {
    Write-Summary @{
        ok = $false
        status = "blocked"
        stage = "input"
        error = "ONNX model not found: $OnnxPath"
    }
    exit 2
}
if (!(Test-Path -LiteralPath $SampleImagePath)) {
    Write-Summary @{
        ok = $false
        status = "blocked"
        stage = "input"
        error = "Sample image not found: $SampleImagePath"
    }
    exit 2
}

if (![string]::IsNullOrWhiteSpace($NcnnRoot)) {
    if (!(Test-Path -LiteralPath $NcnnRoot)) {
        Write-Summary @{
            ok = $false
            status = "blocked"
            stage = "ncnn-root"
            error = "NCNN SDK root not found: $NcnnRoot"
        }
        exit 2
    }
    $env:AITRAIN_NCNN_ROOT = (Resolve-Path -LiteralPath $NcnnRoot).Path
    $converter = Resolve-Onnx2Ncnn $env:AITRAIN_NCNN_ROOT
    if ($converter) {
        $env:AITRAIN_NCNN_ONNX2NCNN = $converter
    }
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $workerOutput = & $WorkerExe `
        --ncnn-smoke $OnnxPath `
        --image $SampleImagePath `
        --output $OutputDir `
        --task-type $TaskType 2>&1
    $workerExitCode = $LASTEXITCODE
} catch {
    $workerOutput = @($_.Exception.Message)
    $lastExitCodeValue = Get-Variable -Name LASTEXITCODE -ValueOnly -ErrorAction SilentlyContinue
    $workerExitCode = if ($null -ne $lastExitCodeValue) { [int]$lastExitCodeValue } else { 1 }
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
$workerText = ($workerOutput | Out-String).Trim()
$jsonLine = ($workerOutput | Where-Object { $_ -match '^\s*\{' } | Select-Object -Last 1)
$workerJson = $null
if ($jsonLine) {
    try {
        $workerJson = $jsonLine | ConvertFrom-Json
    } catch {
        $workerJson = $null
    }
}

$status = if ($workerJson -and $workerJson.status) { [string]$workerJson.status } else { "failed" }
$ok = ($workerExitCode -eq 0 -and $workerJson -and $workerJson.ok)
Write-Summary @{
    ok = [bool]$ok
    status = $status
    stage = if ($workerJson -and $workerJson.stage) { [string]$workerJson.stage } else { "worker" }
    taskType = $TaskType
    onnxPath = (Resolve-Path -LiteralPath $OnnxPath).Path
    sampleImagePath = (Resolve-Path -LiteralPath $SampleImagePath).Path
    ncnnRoot = [Environment]::GetEnvironmentVariable("AITRAIN_NCNN_ROOT")
    onnx2ncnn = [Environment]::GetEnvironmentVariable("AITRAIN_NCNN_ONNX2NCNN")
    workerExe = (Resolve-Path -LiteralPath $WorkerExe).Path
    workerExitCode = $workerExitCode
    workerResult = $workerJson
    workerOutput = $workerText
}

if ($ok) {
    exit 0
}
if ($workerExitCode -ne 0) {
    exit $workerExitCode
}
exit 1
