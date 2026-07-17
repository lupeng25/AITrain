param(
    [string]$PythonExecutable = $env:AITRAIN_PYTHON_EXECUTABLE,
    [string]$WorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# This phase is a standalone diagnostic runner, not a Worker launch.
$env:AITRAIN_STANDALONE_ADAPTER_PROTOCOL = "1"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = "python"
}
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $root ".deps\phase-anomaly-anomalib-smoke"
}

$workFull = [System.IO.Path]::GetFullPath($WorkDir)
$expectedParent = [System.IO.Path]::GetFullPath((Join-Path $root ".deps"))
if (-not $workFull.StartsWith($expectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use anomaly smoke work dir outside .deps: $workFull"
}
if (Test-Path -LiteralPath $workFull) {
    $leaf = Split-Path -Leaf $workFull
    if ($leaf -ne "phase-anomaly-anomalib-smoke") {
        throw "Refusing to remove unexpected work dir: $workFull"
    }
    Remove-Item -LiteralPath $workFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $workFull | Out-Null

$dataset = Join-Path $workFull "dataset"
foreach ($dir in @(
    "train\good",
    "test\good",
    "test\anomaly",
    "masks\test\anomaly"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $dataset $dir) | Out-Null
}

$png1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGNgYGD4DwABBAEAgh2lYQAAAABJRU5ErkJggg=="
$mask1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFgwJ/l7t2VwAAAABJRU5ErkJggg=="
[IO.File]::WriteAllBytes((Join-Path $dataset "train\good\good_001.png"), [Convert]::FromBase64String($png1x1))
[IO.File]::WriteAllBytes((Join-Path $dataset "test\good\good_002.png"), [Convert]::FromBase64String($png1x1))
[IO.File]::WriteAllBytes((Join-Path $dataset "test\anomaly\ng_001.png"), [Convert]::FromBase64String($png1x1))
[IO.File]::WriteAllBytes((Join-Path $dataset "masks\test\anomaly\ng_001.png"), [Convert]::FromBase64String($mask1x1))

$adapter = Join-Path $root "python_trainers\anomaly\anomalib_adapter.py"
$summaryPath = Join-Path $workFull "anomaly_anomalib_smoke_summary.json"
$summary = [ordered]@{
    ok = $false
    status = "blocked"
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    python = $PythonExecutable
    adapter = $adapter
    dataset = $dataset
    checks = @()
    results = @()
}

function Add-Check {
    param([string]$Name, [string]$Status, [string]$Message)
    $script:summary.checks += [ordered]@{ name = $Name; status = $Status; message = $Message }
}

& $PythonExecutable -m py_compile $adapter
if ($LASTEXITCODE -ne 0) {
    Add-Check "adapter_py_compile" "failed" "anomalib_adapter.py failed py_compile"
    $summary.status = "failed"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $summaryPath
    throw "anomalib_adapter.py py_compile failed"
}
Add-Check "adapter_py_compile" "passed" "anomalib_adapter.py compiles"

& $PythonExecutable -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('anomalib') else 3)"
if ($LASTEXITCODE -ne 0) {
    Add-Check "anomalib_available" "blocked" "Anomalib is not installed; smoke summary is blocked, not passed."
    $summary.message = "Install python_trainers/requirements-anomaly.txt to run full Anomalib smoke."
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $summaryPath
    Write-Host "Anomalib smoke blocked: dependencies missing. Summary: $summaryPath" -ForegroundColor Yellow
    exit 0
}
Add-Check "anomalib_available" "passed" "Anomalib import succeeded"

function Invoke-AnomalyAdapter {
    param(
        [string]$Mode,
        [string]$Backend,
        [string]$OutputName,
        [string]$ModelPath = "",
        [string]$ImagePath = "",
        [hashtable]$ExtraParameters = @{}
    )
    $out = Join-Path $workFull $OutputName
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    if ([string]::IsNullOrWhiteSpace($ModelPath)) {
        $ModelPath = Join-Path $out "anomaly_sidecar.json"
    }
    if ([string]::IsNullOrWhiteSpace($ImagePath)) {
        $ImagePath = Join-Path $dataset "test\anomaly\ng_001.png"
    }
    $params = [ordered]@{
        trainingBackend = $Backend
        modelPreset = if ($Backend -eq "anomalib_efficientad") { "anomalib_efficientad_s" } else { "anomalib_patchcore_wide_resnet50_2" }
        epochs = 1
        batchSize = 1
        imageSize = 64
        workers = 0
        device = "cpu"
    }
    foreach ($key in $ExtraParameters.Keys) {
        $params[$key] = $ExtraParameters[$key]
    }
    $request = [ordered]@{
        protocolVersion = 1
        mode = $Mode
        taskId = "anomaly-smoke-$OutputName"
        taskType = "anomaly_detection"
        backend = $Backend
        datasetPath = $dataset
        outputPath = $out
        parameters = $params
        modelPath = $ModelPath
        imagePath = $ImagePath
        options = @{ runtime = "anomalib_python"; iterations = 1; warmupIterations = 0 }
    }
    $requestPath = Join-Path $out "request.json"
    $request | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $requestPath
    & $PythonExecutable -u $adapter --request $requestPath --mode $Mode
    $code = $LASTEXITCODE
    $script:summary.results += [ordered]@{
        mode = $Mode
        backend = $Backend
        exitCode = $code
        outputPath = $out
    }
    return @{ ExitCode = $code; OutputPath = $out }
}

function Invoke-AnomalyBackendLifecycle {
    param(
        [string]$Backend,
        [string]$Name,
        [hashtable]$ExtraParameters = @{}
    )
    $train = Invoke-AnomalyAdapter -Mode "train" -Backend $Backend -OutputName "$Name-train" -ExtraParameters $ExtraParameters
    if ($train.ExitCode -ne 0) {
        return $train
    }
    $sidecar = Join-Path $train.OutputPath "anomaly_sidecar.json"
    $sample = Join-Path $dataset "test\anomaly\ng_001.png"
    Invoke-AnomalyAdapter -Mode "evaluate" -Backend $Backend -OutputName "$Name-evaluate" -ModelPath $sidecar -ImagePath $sample -ExtraParameters $ExtraParameters | Out-Null
    Invoke-AnomalyAdapter -Mode "infer" -Backend $Backend -OutputName "$Name-infer" -ModelPath $sidecar -ImagePath $sample -ExtraParameters $ExtraParameters | Out-Null
    Invoke-AnomalyAdapter -Mode "benchmark" -Backend $Backend -OutputName "$Name-benchmark" -ModelPath $sidecar -ImagePath $sample -ExtraParameters $ExtraParameters | Out-Null
    return $train
}

$patch = Invoke-AnomalyBackendLifecycle -Backend "anomalib_patchcore" -Name "patchcore"

$imagenet = if ($env:AITRAIN_ANOMALIB_IMAGENET_DIR) { $env:AITRAIN_ANOMALIB_IMAGENET_DIR } else { Join-Path $root ".deps\anomalib\imagenette" }
$efficientParams = @{}
if (Test-Path -LiteralPath $imagenet) {
    $efficientParams["imagenetDir"] = $imagenet
}
$efficient = Invoke-AnomalyBackendLifecycle -Backend "anomalib_efficientad" -Name "efficientad" -ExtraParameters $efficientParams
if ($efficient.ExitCode -ne 0 -and -not (Test-Path -LiteralPath $imagenet)) {
    Add-Check "efficientad_imagenet_dir" "blocked" "EfficientAD ImageNet directory missing; this is expected unless AITRAIN_ANOMALIB_IMAGENET_DIR is configured."
}

$failed = @($summary.results | Where-Object { $_.exitCode -ne 0 })
if ($failed.Count -eq 0) {
    $summary.ok = $true
    $summary.status = "passed"
} else {
    $summary.status = "blocked"
    $summary.message = "One or more Anomalib runs were blocked or failed; inspect output reports."
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $summaryPath
Write-Host "Anomalib smoke summary: $summaryPath"
