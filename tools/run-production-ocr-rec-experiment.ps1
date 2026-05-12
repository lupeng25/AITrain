param(
    [string]$WorkDir = ".deps\production-ocr-rec-experiments",
    [string]$DataDir = ".deps\production-ocr-data",
    [string]$Python = "",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$ExperimentName = "",
    [int]$Epochs = 2,
    [int]$BatchSize = 32,
    [int]$TrainLimit = 256,
    [int]$ValLimit = 64,
    [int]$EvalEverySteps = 8,
    [int]$ImageWidth = 320,
    [int]$ImageHeight = 48,
    [int]$MaxTextLength = 25,
    [string]$OfficialConfig = "",
    [string]$DictionaryFile = "dict.txt",
    [switch]$UseGpu,
    [switch]$UsePretrained,
    [string]$PretrainedModel = "",
    [switch]$ExportOnly,
    [switch]$PrepareOnly,
    [switch]$SkipInference
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Resolve-Python {
    if ($Python) {
        return Resolve-RepoPath $Python
    }
    $candidates = @(
        (Join-Path $script:Root ".deps\python-3.13.13-ocr-amd64\python.exe"),
        (Join-Path $script:Root ".deps\python-3.13.13-embed-amd64\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }
    throw "No Python executable found. Provide -Python or provision the OCR Python environment."
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-SubsetLabel {
    param(
        [string]$Source,
        [string]$Destination,
        [int]$Limit
    )
    if (!(Test-Path -LiteralPath $Source)) {
        throw "Missing label source: $Source"
    }
    if ($Limit -le 0) {
        Get-Content -LiteralPath $Source -Encoding UTF8 |
            Set-Content -LiteralPath $Destination -Encoding UTF8
    } else {
        Get-Content -LiteralPath $Source -Encoding UTF8 -TotalCount $Limit |
            Set-Content -LiteralPath $Destination -Encoding UTF8
    }
    $count = @(Get-Content -LiteralPath $Destination -Encoding UTF8 | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
    $minimumCount = if ($Limit -le 0) { 1 } else { [Math]::Min($Limit, 1) }
    if ($count -lt $minimumCount) {
        throw "Subset label file is empty: $Destination"
    }
    return $count
}

function Get-JsonNumber {
    param(
        [object]$Object,
        [string[]]$Names
    )
    if ($null -eq $Object) {
        return $null
    }
    $propertyNames = @($Object.PSObject.Properties | ForEach-Object { $_.Name })
    foreach ($name in $Names) {
        if ($propertyNames -contains $name) {
            $value = $Object.$name
            if ($null -ne $value -and "$value" -ne "") {
                return [double]$value
            }
        }
    }
    if ($propertyNames -contains "metrics") {
        return Get-JsonNumber -Object $Object.metrics -Names $Names
    }
    return $null
}

function Assert-PaddleGpuReady {
    param([string]$PythonPath)

    $gpuInfo = ""
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $gpuInfo = (& nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>$null) -join "; "
    }

    $probeScript = @'
import json
import sys

try:
    import paddle
    result = {
        "paddleVersion": getattr(paddle, "__version__", ""),
        "compiledWithCuda": bool(paddle.device.is_compiled_with_cuda()),
        "device": paddle.get_device(),
    }
    if result["compiledWithCuda"]:
        try:
            paddle.utils.run_check()
            result["runCheck"] = "passed"
        except Exception as exc:
            result["runCheck"] = "failed"
            result["error"] = str(exc)
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(3)
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["compiledWithCuda"] else 2)
except Exception as exc:
    print(json.dumps({"error": str(exc)}, ensure_ascii=False))
    sys.exit(4)
'@
    $probePath = Join-Path ([System.IO.Path]::GetTempPath()) ("aitrain_paddle_gpu_probe_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    Set-Content -LiteralPath $probePath -Value $probeScript -Encoding UTF8
    try {
        $probeOutput = & $PythonPath $probePath
        $exitCode = $LASTEXITCODE
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
    if ($exitCode -ne 0) {
        $details = ($probeOutput -join "`n").Trim()
        if ([string]::IsNullOrWhiteSpace($details)) {
            $details = "No Paddle GPU probe output."
        }
        if (-not [string]::IsNullOrWhiteSpace($gpuInfo)) {
            $details = "$details`nGPU: $gpuInfo"
        }
        throw "GPU training requested, but the current PaddleOCR Python environment is not GPU-ready. $details"
    }
    Write-Host "[production-ocr-rec-experiment] Paddle GPU probe passed: $($probeOutput -join ' ')" -ForegroundColor Green
    if (-not [string]::IsNullOrWhiteSpace($gpuInfo)) {
        Write-Host "[production-ocr-rec-experiment] GPU: $gpuInfo" -ForegroundColor Green
    }
}

$pythonExe = Resolve-Python
$dataFull = Resolve-RepoPath $DataDir
$repoFull = Resolve-RepoPath $PaddleOcrRepo
$recDataset = Join-Path $dataFull "rec_dataset"
if (!(Test-Path -LiteralPath $recDataset)) {
    throw "Production OCR Rec dataset is missing: $recDataset"
}
if (!(Test-Path -LiteralPath (Join-Path $repoFull "tools\train.py"))) {
    throw "PaddleOCR source checkout is missing or incomplete: $repoFull"
}
if ($UseGpu) {
    Assert-PaddleGpuReady -PythonPath $pythonExe
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($ExperimentName)) {
    $ExperimentName = "rec-exp-$timestamp"
}
$workRoot = Resolve-RepoPath $WorkDir
$experimentDir = Join-Path $workRoot $ExperimentName
$outputDir = Join-Path $experimentDir "official_rec"
New-Item -ItemType Directory -Force $experimentDir, $outputDir | Out-Null

$trainLabelName = "rec_gt_train_exp_$TrainLimit.txt"
$valLabelName = "rec_gt_val_exp_$ValLimit.txt"
$trainLabelPath = Join-Path $recDataset $trainLabelName
$valLabelPath = Join-Path $recDataset $valLabelName
$trainCount = Write-SubsetLabel -Source (Join-Path $recDataset "rec_gt_train.txt") -Destination $trainLabelPath -Limit $TrainLimit
$valCount = Write-SubsetLabel -Source (Join-Path $recDataset "rec_gt_val.txt") -Destination $valLabelPath -Limit $ValLimit

$resolvedPaddleOcrRef = (& git -C $repoFull rev-parse HEAD 2>$null).Trim()
$pretrainedValue = ""
if ($UsePretrained -or $ExportOnly) {
    if ([string]::IsNullOrWhiteSpace($PretrainedModel)) {
        throw "-UsePretrained or -ExportOnly requires -PretrainedModel. Use a checkpoint base path without .pdparams when required by PaddleOCR."
    }
    $pretrainedValue = Resolve-RepoPath $PretrainedModel
}

$requestPath = Join-Path $experimentDir "paddleocr_rec_official_request.json"
$requestParameters = [ordered]@{
    trainingBackend = "paddleocr_rec_official"
    paddleOcrRepoPath = $repoFull
    paddleOcrRef = $resolvedPaddleOcrRef
    runOfficial = -not [bool]$PrepareOnly
    prepareOnly = [bool]$PrepareOnly
    exportOnly = [bool]$ExportOnly
    epochs = $Epochs
    batchSize = $BatchSize
    imageWidth = $ImageWidth
    imageHeight = $ImageHeight
    recImageShape = "3,$ImageHeight,$ImageWidth"
    maxTextLength = $MaxTextLength
    useGpu = [bool]$UseGpu
    trainLabelFile = $trainLabelName
    valLabelFile = $valLabelName
    dictionaryFile = $DictionaryFile
    runInferenceAfterExport = -not [bool]$SkipInference
    inferenceImage = "images/test/totaltext_002700_img589_0.jpg"
    evalEverySteps = $EvalEverySteps
    acceptanceNote = "Production OCR Rec experiment; not a production acceptance pass by itself."
}
if (-not [string]::IsNullOrWhiteSpace($OfficialConfig)) {
    $requestParameters.officialConfig = $OfficialConfig.Replace("\", "/")
}
if ($UsePretrained -or $ExportOnly) {
    $requestParameters.pretrainedModel = $pretrainedValue
}
$request = [ordered]@{
    protocolVersion = 1
    taskId = $ExperimentName
    taskType = "ocr_recognition"
    datasetPath = $recDataset
    outputPath = $outputDir
    backend = "paddleocr_rec_official"
    parameters = $requestParameters
}
Write-JsonFile -Path $requestPath -Value $request

$commandLine = @(
    $pythonExe,
    (Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_official_adapter.py"),
    "--request",
    $requestPath
)
$commandLinePath = Join-Path $experimentDir "run_experiment.ps1"
@(
    ("Set-Location ""{0}""" -f $script:Root),
    (($commandLine | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " ")
) | Set-Content -LiteralPath $commandLinePath -Encoding UTF8

Write-Host "[production-ocr-rec-experiment] $($commandLine -join ' ')" -ForegroundColor Cyan
& $pythonExe (Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_official_adapter.py") --request $requestPath
$adapterExitCode = $LASTEXITCODE
$reportPath = Join-Path $outputDir "paddleocr_official_rec_report.json"
$report = $null
if (Test-Path -LiteralPath $reportPath) {
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reportPath | ConvertFrom-Json
}
$accuracy = Get-JsonNumber -Object $report -Names @("accuracy", "acc")
$cer = Get-JsonNumber -Object $report -Names @("cer", "CER")
$normalizedEditDistance = Get-JsonNumber -Object $report -Names @("normalizedEditDistance", "norm_edit_dis")
$loss = Get-JsonNumber -Object $report -Names @("loss")

$finishedAt = [DateTime]::UtcNow
$summary = [ordered]@{
    ok = ($adapterExitCode -eq 0 -and $null -ne $report -and $report.ok)
    status = if ($adapterExitCode -eq 0 -and $null -ne $report -and $report.ok) { "completed" } else { "failed" }
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    experimentName = $ExperimentName
    experimentDir = $experimentDir
    python = $pythonExe
    paddleOcrRepo = $repoFull
    paddleOcrRef = $resolvedPaddleOcrRef
    trainLimit = $TrainLimit
    valLimit = $ValLimit
    trainCount = $trainCount
    valCount = $valCount
    epochs = $Epochs
    batchSize = $BatchSize
    evalEverySteps = $EvalEverySteps
    officialConfig = $OfficialConfig
    dictionaryFile = $DictionaryFile
    useGpu = [bool]$UseGpu
    usePretrained = [bool]$UsePretrained
    exportOnly = [bool]$ExportOnly
    pretrainedModel = $pretrainedValue
    requestPath = $requestPath
    commandLinePath = $commandLinePath
    reportPath = $reportPath
    adapterExitCode = $adapterExitCode
    metrics = [ordered]@{
        accuracy = $accuracy
        cer = $cer
        normalizedEditDistance = $normalizedEditDistance
        loss = $loss
    }
    note = "This experiment is for Rec metric iteration only. It is not a production OCR acceptance pass."
}
$summaryPath = Join-Path $experimentDir "rec_experiment_summary.json"
Write-JsonFile -Path $summaryPath -Value $summary

Write-Host "Rec experiment summary: $summaryPath" -ForegroundColor Cyan
if ($adapterExitCode -ne 0) {
    exit $adapterExitCode
}
exit 0
