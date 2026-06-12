param(
    [string]$WorkDir = ".deps\phase-ppocrv6-model-matrix-smoke",
    [string]$PythonDir = ".deps\python-3.13.13-ocr-amd64",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$PaddleOcrRef = "v3.7.0",
    [string]$PaddlePaddleRequirement = "paddlepaddle==3.3.1",
    [switch]$DisablePinnedConstraints,
    [switch]$SkipInstall,
    [switch]$SkipFullChain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Resolve-OcrPython {
    param([string]$PythonDirPath)
    $pythonDirFull = Resolve-RepoPath $PythonDirPath
    $venvPython = Join-Path $pythonDirFull "Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    return (Join-Path $pythonDirFull "python.exe")
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-AdapterRow {
    param(
        [string]$Name,
        [string]$Preset,
        [string]$TaskType,
        [string]$Backend,
        [string]$DatasetPath,
        [string]$OutputPath,
        [string]$AdapterPath,
        [hashtable]$ExtraParameters
    )
    $requestPath = Join-Path $requestsDir "$Name.json"
    $parameters = [ordered]@{
        trainingBackend = $Backend
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        prepareOnly = $true
        runOfficial = $false
        modelPreset = $Preset
        epochs = 1
        batchSize = 1
        validationRatio = 0.5
    }
    foreach ($key in $ExtraParameters.Keys) {
        $parameters[$key] = $ExtraParameters[$key]
    }
    $request = [ordered]@{
        protocolVersion = 1
        taskId = "phase-ppocrv6-$Name"
        taskType = $TaskType
        datasetPath = $DatasetPath
        outputPath = $OutputPath
        backend = $Backend
        parameters = $parameters
    }
    Write-JsonFile -Path $requestPath -Value $request
    Write-Host "[$Name] $Preset" -ForegroundColor Cyan
    $adapterOutput = & $pythonExe $AdapterPath "--request" $requestPath 2>&1
    $adapterExitCode = $LASTEXITCODE
    $adapterOutput | ForEach-Object { Write-Host $_ }
    if ($adapterExitCode -ne 0) {
        throw "PP-OCRv6 matrix row failed: $Name ($Preset)"
    }
    $reportName = if ($Backend -eq "paddleocr_det_official") { "paddleocr_official_det_report.json" } else { "paddleocr_official_rec_report.json" }
    $reportPath = Join-Path $OutputPath $reportName
    if (!(Test-Path -LiteralPath $reportPath)) {
        throw "PP-OCRv6 matrix row did not write report: $reportPath"
    }
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reportPath | ConvertFrom-Json
    if ($report.ocrVersion -ne "PP-OCRv6") {
        throw "Unexpected OCR version for $Preset`: $($report.ocrVersion)"
    }
    if ($report.modelPreset -ne $Preset) {
        throw "Unexpected modelPreset for $Preset`: $($report.modelPreset)"
    }
    if ([string]$report.resolvedOfficialConfig -notmatch "PP-OCRv6") {
        throw "Resolved config does not point to PP-OCRv6 for $Preset`: $($report.resolvedOfficialConfig)"
    }
    if ($Backend -eq "paddleocr_rec_official") {
        if ([string]::IsNullOrWhiteSpace([string]$report.recAlgorithm)) {
            throw "Missing recAlgorithm for $Preset"
        }
        if ($report.dictionarySource -ne "official_config") {
            throw "Expected official_config dictionary source for $Preset, got $($report.dictionarySource)"
        }
    }
    return [ordered]@{
        name = $Name
        taskType = $TaskType
        backend = $Backend
        modelPreset = $Preset
        status = "passed"
        reportPath = $reportPath
        resolvedOfficialConfig = [string]$report.resolvedOfficialConfig
        dictionarySource = if ($Backend -eq "paddleocr_rec_official") { [string]$report.dictionarySource } else { "" }
        recAlgorithm = if ($Backend -eq "paddleocr_rec_official") { [string]$report.recAlgorithm } else { "" }
    }
}

$workFull = Resolve-RepoPath $WorkDir
$repoFull = Resolve-RepoPath $PaddleOcrRepo
$fullChainWork = Join-Path $workFull "full-chain-tiny"
$matrixData = Join-Path $workFull "matrix-data"
$matrixRuns = Join-Path $workFull "matrix-runs"
$requestsDir = Join-Path $workFull "requests"
New-Item -ItemType Directory -Force $workFull, $matrixRuns, $requestsDir | Out-Null

if (!$SkipFullChain) {
    $phase31Args = @{
        WorkDir = $fullChainWork
        PythonDir = $PythonDir
        PaddleOcrRepo = $PaddleOcrRepo
        PaddleOcrRef = $PaddleOcrRef
        PaddlePaddleRequirement = $PaddlePaddleRequirement
        OcrVersion = "PP-OCRv6"
        PPOCRv6Tier = "tiny"
    }
    if ($DisablePinnedConstraints) {
        $phase31Args["DisablePinnedConstraints"] = $true
    }
    if ($SkipInstall) {
        $phase31Args["SkipInstall"] = $true
    }
    & (Join-Path $script:Root "tools\phase31-paddleocr-full-official-smoke.ps1") @phase31Args
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCRv6 tiny full-chain smoke failed."
    }
}

$pythonExe = Resolve-OcrPython -PythonDirPath $PythonDir
if (!(Test-Path -LiteralPath $pythonExe)) {
    throw "OCR Python executable is missing after setup: $pythonExe"
}
if (!(Test-Path -LiteralPath (Join-Path $repoFull "tools\train.py"))) {
    throw "PaddleOCR source checkout is missing after setup: $repoFull"
}
$resolvedPaddleOcrRef = (& git -C $repoFull rev-parse HEAD 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($resolvedPaddleOcrRef)) {
    $resolvedPaddleOcrRef = $PaddleOcrRef
}

if (!(Test-Path -LiteralPath (Join-Path $matrixData "paddleocr_det\det_gt.txt"))) {
    & $pythonExe (Join-Path $script:Root "examples\create-minimal-datasets.py") "--output" $matrixData
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate minimal matrix datasets."
    }
}

$detDatasetPath = Join-Path $matrixData "paddleocr_det"
$recDatasetPath = Join-Path $matrixData "paddleocr_rec"
$rows = @()
foreach ($tier in @("tiny", "small", "medium")) {
    $rows += Invoke-AdapterRow `
        -Name "det-$tier" `
        -Preset ("PP-OCRv6_{0}_det" -f $tier) `
        -TaskType "ocr_detection" `
        -Backend "paddleocr_det_official" `
        -DatasetPath $detDatasetPath `
        -OutputPath (Join-Path $matrixRuns "det-$tier") `
        -AdapterPath (Join-Path $script:Root "python_trainers\ocr_det\paddleocr_det_official_adapter.py") `
        -ExtraParameters @{ imageSize = 64; calMetricDuringTrain = $false }
    $rows += Invoke-AdapterRow `
        -Name "rec-$tier" `
        -Preset ("PP-OCRv6_{0}_rec" -f $tier) `
        -TaskType "ocr_recognition" `
        -Backend "paddleocr_rec_official" `
        -DatasetPath $recDatasetPath `
        -OutputPath (Join-Path $matrixRuns "rec-$tier") `
        -AdapterPath (Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_official_adapter.py") `
        -ExtraParameters @{ imageWidth = 320; imageHeight = 48; recImageShape = "3,48,320"; maxTextLength = 8 }
}

$finishedAt = [DateTime]::UtcNow
$summary = [ordered]@{
    ok = $true
    status = "passed"
    scope = "ppocrv6_model_matrix_smoke"
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    workDir = $workFull
    python = $pythonExe
    paddleOcrRepo = $repoFull
    paddleOcrRef = $resolvedPaddleOcrRef
    fullChainTiny = [ordered]@{
        skipped = [bool]$SkipFullChain
        workDir = $fullChainWork
    }
    requiredRowCount = 6
    passedRowCount = $rows.Count
    rows = $rows
    note = "PP-OCRv6 matrix validates official Det/Rec preset wiring and one tiny Det+Rec+System official-chain smoke. It is not an OCR accuracy benchmark."
}
$summaryPath = Join-Path $workFull "phase_ppocrv6_model_matrix_summary.json"
Write-JsonFile -Path $summaryPath -Value $summary
Write-Host "PP-OCRv6 model matrix smoke summary: $summaryPath" -ForegroundColor Green
