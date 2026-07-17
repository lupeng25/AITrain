param(
    [string]$WorkDir = ".deps\phase-smp-semantic-segmentation-smoke",
    [string]$Python = "",
    [switch]$SkipTraining
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# This phase is a standalone diagnostic runner, not a Worker launch.
$env:AITRAIN_STANDALONE_ADAPTER_PROTOCOL = "1"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-Python {
    if (-not [string]::IsNullOrWhiteSpace($Python)) {
        return $Python
    }
    $candidates = @(
        (Join-Path $root ".deps\python-3.13.13-embed-amd64\python.exe"),
        "python",
        "python3"
    )
    foreach ($candidate in $candidates) {
        try {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }
    throw "No usable Python executable found."
}

function Write-Json {
    param([string]$Path, [object]$Value)
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding UTF8
}

$pythonExe = Resolve-Python
$workFull = [System.IO.Path]::GetFullPath((Join-Path $root $WorkDir))
New-Item -ItemType Directory -Force -Path $workFull | Out-Null
$summaryPath = Join-Path $workFull "smp_semantic_smoke_summary.json"

Write-Host "SMP semantic smoke: generate datasets" -ForegroundColor Cyan
& $pythonExe (Join-Path $root "examples\create-minimal-datasets.py") --output $workFull --profile minimal
if ($LASTEXITCODE -ne 0) {
    throw "Minimal dataset generation failed."
}

Write-Host "SMP semantic smoke: py_compile" -ForegroundColor Cyan
& $pythonExe -m py_compile `
    (Join-Path $root "python_trainers\semantic_segmentation\smp_trainer.py") `
    (Join-Path $root "python_trainers\semantic_segmentation\smp_evaluator.py")
if ($LASTEXITCODE -ne 0) {
    throw "SMP trainer/evaluator py_compile failed."
}

$moduleProbe = @"
import importlib.util, sys
mods = ['segmentation_models_pytorch', 'torch', 'timm', 'onnx', 'onnxruntime', 'PIL', 'numpy', 'yaml']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(','.join(missing))
sys.exit(3 if missing else 0)
"@
$missing = (& $pythonExe -c $moduleProbe)
$depsOk = $LASTEXITCODE -eq 0
if ($SkipTraining -or -not $depsOk) {
    $status = if ($SkipTraining) { "skipped" } else { "blocked" }
    Write-Json $summaryPath @{
        ok = $SkipTraining.IsPresent
        status = $status
        workDir = $workFull
        python = $pythonExe
        missingModules = @($missing -split "," | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        note = "Training smoke requires segmentation-models-pytorch, torch, timm, onnx, onnxruntime, Pillow, and numpy."
    }
    if ($SkipTraining) {
        Write-Host "SMP semantic smoke skipped training: $summaryPath" -ForegroundColor Yellow
        exit 0
    }
    throw "SMP Python dependencies missing: $missing. Install python_trainers\requirements-smp.txt and rerun."
}

Write-Host "SMP semantic smoke: train" -ForegroundColor Cyan
$requestPath = Join-Path $workFull "smp_semantic_request.json"
& $pythonExe (Join-Path $root "python_trainers\semantic_segmentation\smp_trainer.py") --request $requestPath
if ($LASTEXITCODE -ne 0) {
    throw "SMP semantic trainer failed."
}

$runDir = Join-Path $workFull "runs\smp_semantic"
$required = @(
    (Join-Path $runDir "best.pt"),
    (Join-Path $runDir "best.onnx"),
    (Join-Path $runDir "smp_training_report.json"),
    (Join-Path $runDir "semantic_segmentation_sidecar.json")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing SMP smoke artifact: $path"
    }
}

Write-Host "SMP semantic smoke: evaluate" -ForegroundColor Cyan
$evalRequest = @{
    modelPath = (Join-Path $runDir "best.onnx")
    datasetPath = (Join-Path $workFull "semantic_mask")
    outputPath = (Join-Path $workFull "evaluation")
    options = @{ split = "val"; maxOverlays = 4 }
}
$evalRequestPath = Join-Path $workFull "smp_evaluation_request.json"
Write-Json $evalRequestPath $evalRequest
& $pythonExe (Join-Path $root "python_trainers\semantic_segmentation\smp_evaluator.py") --request $evalRequestPath
if ($LASTEXITCODE -ne 0) {
    throw "SMP semantic evaluator failed."
}

Write-Json $summaryPath @{
    ok = $true
    status = "passed"
    workDir = $workFull
    python = $pythonExe
    checkpointPath = (Join-Path $runDir "best.pt")
    onnxPath = (Join-Path $runDir "best.onnx")
    trainingReportPath = (Join-Path $runDir "smp_training_report.json")
    evaluationReportPath = (Join-Path $workFull "evaluation\evaluation_report.json")
}
Write-Host "SMP semantic smoke passed: $summaryPath" -ForegroundColor Green
