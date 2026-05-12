param(
    [string]$WorkDir = ".deps\production-ocr-official-chain",
    [string]$DataDir = ".deps\production-ocr-data",
    [string]$Python = "",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$PaddleOcrRef = "",
    [int]$DetEpochs = 1,
    [int]$DetBatchSize = 1,
    [int]$DetImageSize = 256,
    [int]$RecEpochs = 1,
    [int]$RecBatchSize = 8,
    [int]$RecImageWidth = 320,
    [int]$RecImageHeight = 48,
    [int]$RecMaxTextLength = 25,
    [int]$RecEvalEverySteps = 1000000,
    [int]$RecSubsetTrain = 256,
    [int]$RecSubsetVal = 64,
    [string]$RecOfficialConfig = "",
    [string]$RecDictionaryFile = "dict.txt",
    [string]$RecPretrainedModel = "",
    [switch]$UseGpu,
    [switch]$UseRecCpuSubset,
    [switch]$SkipDataPrep,
    [switch]$SkipExistingReports,
    [switch]$Force,
    [switch]$AllowBlocked,
    [switch]$RequireDetOnnxEvidence,
    [string]$OcrDetOnnxSummary = ""
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

function Invoke-CheckedCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    Write-Host "[$Name] $FilePath $($ArgumentList -join ' ')" -ForegroundColor Cyan
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Ensure-RecSubsetLabels {
    param(
        [string]$RecDatasetPath,
        [int]$TrainCount,
        [int]$ValCount
    )
    $trainSource = Join-Path $RecDatasetPath "rec_gt_train.txt"
    $valSource = Join-Path $RecDatasetPath "rec_gt_val.txt"
    if (!(Test-Path -LiteralPath $trainSource) -or !(Test-Path -LiteralPath $valSource)) {
        throw "Rec CPU subset requires rec_gt_train.txt and rec_gt_val.txt under $RecDatasetPath"
    }
    $trainSubset = Join-Path $RecDatasetPath "rec_gt_train_cpu_subset.txt"
    $valSubset = Join-Path $RecDatasetPath "rec_gt_val_cpu_subset.txt"
    Get-Content -LiteralPath $trainSource -Encoding UTF8 -TotalCount $TrainCount | Set-Content -LiteralPath $trainSubset -Encoding UTF8
    Get-Content -LiteralPath $valSource -Encoding UTF8 -TotalCount $ValCount | Set-Content -LiteralPath $valSubset -Encoding UTF8
    return @{
        train = "rec_gt_train_cpu_subset.txt"
        val = "rec_gt_val_cpu_subset.txt"
    }
}

function Invoke-AdapterIfNeeded {
    param(
        [string]$Name,
        [string]$ReportPath,
        [string]$PythonExe,
        [string]$AdapterPath,
        [string]$RequestPath
    )
    if ($SkipExistingReports -and !$Force -and (Test-Path -LiteralPath $ReportPath)) {
        Write-Host "[$Name] Skipping existing report: $ReportPath" -ForegroundColor Yellow
        return "skipped"
    }
    Invoke-CheckedCommand -Name $Name -FilePath $PythonExe -ArgumentList @($AdapterPath, "--request", $RequestPath)
    if (!(Test-Path -LiteralPath $ReportPath)) {
        throw "$Name completed but report is missing: $ReportPath"
    }
    return "completed"
}

$pythonExe = Resolve-Python
$dataFull = Resolve-RepoPath $DataDir
$workFull = Resolve-RepoPath $WorkDir
$repoFull = Resolve-RepoPath $PaddleOcrRepo
$requestsDir = Join-Path $workFull "requests"
$reportsDir = Join-Path $workFull "reports"
$acceptanceDir = Join-Path $workFull "acceptance"
New-Item -ItemType Directory -Force $requestsDir, $reportsDir, $acceptanceDir | Out-Null

$manifestPath = Join-Path $dataFull "manifests\production_ocr_data_manifest.json"
if (!$SkipDataPrep -and (!(Test-Path -LiteralPath $manifestPath) -or $Force)) {
    Invoke-CheckedCommand -Name "prepare-production-ocr-data" -FilePath (Join-Path $script:Root "tools\prepare-production-ocr-data.ps1") -ArgumentList @("-WorkDir", $dataFull, "-Python", $pythonExe)
} elseif (!(Test-Path -LiteralPath $manifestPath)) {
    throw "Production OCR data manifest is missing: $manifestPath. Remove -SkipDataPrep or run prepare-production-ocr-data.ps1 first."
} else {
    Write-Host "Using existing production OCR data manifest: $manifestPath" -ForegroundColor Cyan
}

if (!(Test-Path -LiteralPath (Join-Path $repoFull "tools\train.py"))) {
    throw "PaddleOCR source checkout is missing or incomplete: $repoFull"
}
$resolvedPaddleOcrRef = (& git -C $repoFull rev-parse HEAD 2>$null).Trim()
if ($PaddleOcrRef -and $resolvedPaddleOcrRef -ne $PaddleOcrRef) {
    Write-Host "Requested PaddleOCR ref '$PaddleOcrRef', current checkout is '$resolvedPaddleOcrRef'. The script does not checkout refs automatically." -ForegroundColor Yellow
}

$detDataset = Join-Path $dataFull "det_dataset"
$recDataset = Join-Path $dataFull "rec_dataset"
$systemImages = Join-Path $dataFull "system_images"
foreach ($required in @($detDataset, $recDataset, $systemImages)) {
    if (!(Test-Path -LiteralPath $required)) {
        throw "Required production OCR data path is missing: $required"
    }
}

$detOutput = Join-Path $reportsDir "det_official"
$recOutput = Join-Path $reportsDir ($(if ($UseRecCpuSubset) { "rec_official_cpu_subset" } else { "rec_official" }))
$systemOutput = Join-Path $reportsDir "system_official"
$detReport = Join-Path $detOutput "paddleocr_official_det_report.json"
$recReport = Join-Path $recOutput "paddleocr_official_rec_report.json"
$systemReport = Join-Path $systemOutput "paddleocr_official_system_report.json"

$detRequestPath = Join-Path $requestsDir "production_det_official_request.json"
$detRequest = [ordered]@{
    protocolVersion = 1
    taskId = "production-ocr-det-official"
    taskType = "ocr_detection"
    datasetPath = $detDataset
    outputPath = $detOutput
    backend = "paddleocr_det_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_det_official"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        runOfficial = $true
        prepareOnly = $false
        epochs = $DetEpochs
        batchSize = $DetBatchSize
        imageSize = $DetImageSize
        evalEverySteps = 1000000
        useGpu = [bool]$UseGpu
        trainLabelFile = "det_gt_train.txt"
        valLabelFile = "det_gt_val.txt"
        calMetricDuringTrain = $false
    }
}
Write-JsonFile -Path $detRequestPath -Value $detRequest

$recLabels = @{
    train = "rec_gt_train.txt"
    val = "rec_gt_val.txt"
}
if ($UseRecCpuSubset) {
    $recLabels = Ensure-RecSubsetLabels -RecDatasetPath $recDataset -TrainCount $RecSubsetTrain -ValCount $RecSubsetVal
}
$recRequestPath = Join-Path $requestsDir "production_rec_official_request.json"
$recRequest = [ordered]@{
    protocolVersion = 1
    taskId = "production-ocr-rec-official"
    taskType = "ocr_recognition"
    datasetPath = $recDataset
    outputPath = $recOutput
    backend = "paddleocr_rec_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_rec_official"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        runOfficial = $true
        prepareOnly = $false
        epochs = $RecEpochs
        batchSize = $RecBatchSize
        imageWidth = $RecImageWidth
        imageHeight = $RecImageHeight
        recImageShape = "3,$RecImageHeight,$RecImageWidth"
        maxTextLength = $RecMaxTextLength
        useGpu = [bool]$UseGpu
        trainLabelFile = $recLabels.train
        valLabelFile = $recLabels.val
        dictionaryFile = $RecDictionaryFile
        runInferenceAfterExport = $true
        inferenceImage = "images/test/totaltext_002700_img589_0.jpg"
        evalEverySteps = $RecEvalEverySteps
        acceptanceNote = $(if ($UseRecCpuSubset) { "CPU subset run from public Total-Text crops; not a production accuracy pass." } else { "Full public Total-Text Rec run; still not customer-domain production evidence." })
    }
}
if (-not [string]::IsNullOrWhiteSpace($RecOfficialConfig)) {
    $recRequest.parameters.officialConfig = $RecOfficialConfig
}
if (-not [string]::IsNullOrWhiteSpace($RecPretrainedModel)) {
    $recRequest.parameters.pretrainedModel = $RecPretrainedModel
}
Write-JsonFile -Path $recRequestPath -Value $recRequest

$systemRequestPath = Join-Path $requestsDir "production_system_official_request.json"
$systemRequest = [ordered]@{
    protocolVersion = 1
    taskId = "production-ocr-system-official"
    taskType = "ocr"
    datasetPath = $systemImages
    outputPath = $systemOutput
    backend = "paddleocr_system_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_system_official"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        detModelDir = (Join-Path $detOutput "official_inference")
        recModelDir = (Join-Path $recOutput "official_inference")
        dictionaryFile = (Join-Path $recOutput "official_data\dict.txt")
        inferenceImage = $systemImages
        dropScore = 0.0
        useGpu = [bool]$UseGpu
        acceptanceNote = "System run uses official Det and Rec model directories from this chain."
    }
}
Write-JsonFile -Path $systemRequestPath -Value $systemRequest

$steps = [ordered]@{}
$steps.det = Invoke-AdapterIfNeeded -Name "official-det" -ReportPath $detReport -PythonExe $pythonExe -AdapterPath (Join-Path $script:Root "python_trainers\ocr_det\paddleocr_det_official_adapter.py") -RequestPath $detRequestPath
$steps.rec = Invoke-AdapterIfNeeded -Name "official-rec" -ReportPath $recReport -PythonExe $pythonExe -AdapterPath (Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_official_adapter.py") -RequestPath $recRequestPath
$steps.system = Invoke-AdapterIfNeeded -Name "official-system" -ReportPath $systemReport -PythonExe $pythonExe -AdapterPath (Join-Path $script:Root "python_trainers\ocr_system\paddleocr_system_official_adapter.py") -RequestPath $systemRequestPath

$acceptanceArgs = @(
    "-WorkDir", $acceptanceDir,
    "-DetDataset", $detDataset,
    "-RecDataset", $recDataset,
    "-SystemImages", $systemImages,
    "-OfficialDetReport", $detReport,
    "-OfficialRecReport", $recReport,
    "-OfficialSystemReport", $systemReport
)
if ($OcrDetOnnxSummary) {
    $acceptanceArgs += @("-OcrDetOnnxSummary", (Resolve-RepoPath $OcrDetOnnxSummary))
}
if ($RequireDetOnnxEvidence) {
    $acceptanceArgs += "-RequireDetOnnxEvidence"
}

Write-Host "[production-ocr-acceptance] tools\production-ocr-acceptance.ps1 $($acceptanceArgs -join ' ')" -ForegroundColor Cyan
$acceptanceCommand = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $script:Root "tools\production-ocr-acceptance.ps1")
) + $acceptanceArgs
& powershell.exe @acceptanceCommand
$acceptanceExitCode = $LASTEXITCODE
$acceptanceReportPath = Join-Path $acceptanceDir "production_ocr_acceptance_report.json"
$acceptanceSummaryPath = Join-Path $acceptanceDir "production_ocr_acceptance_summary.md"
$acceptanceStatus = "unknown"
if (Test-Path -LiteralPath $acceptanceReportPath) {
    $acceptanceReport = Get-Content -Raw -Encoding UTF8 -LiteralPath $acceptanceReportPath | ConvertFrom-Json
    $acceptanceStatus = [string]$acceptanceReport.status
}

$finishedAt = [DateTime]::UtcNow
$chainSummary = [ordered]@{
    ok = ($acceptanceExitCode -eq 0)
    status = $acceptanceStatus
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    workDir = $workFull
    dataDir = $dataFull
    python = $pythonExe
    paddleOcrRepo = $repoFull
    paddleOcrRef = $resolvedPaddleOcrRef
    useGpu = [bool]$UseGpu
    useRecCpuSubset = [bool]$UseRecCpuSubset
    steps = $steps
    requests = [ordered]@{
        det = $detRequestPath
        rec = $recRequestPath
        system = $systemRequestPath
    }
    reports = [ordered]@{
        det = $detReport
        rec = $recReport
        system = $systemReport
        acceptance = $acceptanceReportPath
        acceptanceSummary = $acceptanceSummaryPath
    }
    acceptanceExitCode = $acceptanceExitCode
    note = "A blocked production OCR gate is valid evidence. Do not weaken thresholds or claim production OCR readiness from blocked results."
}
$chainSummaryPath = Join-Path $workFull "production_ocr_official_chain_summary.json"
Write-JsonFile -Path $chainSummaryPath -Value $chainSummary

Write-Host "Production OCR official chain summary: $chainSummaryPath" -ForegroundColor Cyan
if (Test-Path -LiteralPath $acceptanceSummaryPath) {
    Write-Host "Production OCR acceptance summary: $acceptanceSummaryPath" -ForegroundColor Cyan
}
if ($acceptanceExitCode -ne 0 -and !$AllowBlocked) {
    exit $acceptanceExitCode
}
exit 0
