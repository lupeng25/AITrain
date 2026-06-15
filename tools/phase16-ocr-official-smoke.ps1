param(
    [string]$WorkDir = ".deps\phase16-ocr-official-smoke",
    [string]$PythonExe = "",
    [string]$PythonDir = ".deps\envs\ocr-cpu",
    [string]$PaddleOcrRepo = ".deps\repos\PaddleOCR",
    [string]$PaddleOcrRef = "f8b41a62bba991d35e578ffa712107a042b0c3b0",
    [string]$PaddlePaddleRequirement = "paddlepaddle==3.3.1",
    [switch]$DisablePinnedConstraints,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
. (Join-Path $PSScriptRoot "deps-layout.ps1")

function Resolve-RepoPath([string]$Path) {
    return Resolve-AITrainRepoPath -Root $script:Root -Path $Path
}

$pythonDirFull = Resolve-RepoPath $PythonDir
$resolvedPythonExe = Join-Path $pythonDirFull "python.exe"
$venvPython = Join-Path $pythonDirFull "Scripts\python.exe"
if (Test-Path $venvPython) {
    $resolvedPythonExe = $venvPython
}
if ($PythonExe) {
    $resolvedPythonExe = Resolve-RepoPath $PythonExe
}
$repoFull = Resolve-AITrainPaddleOcrRepo -Root $script:Root -RequestedPath $PaddleOcrRepo
$workFull = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force $workFull | Out-Null

if (!(Test-Path $resolvedPythonExe)) {
    if ($PythonExe) {
        throw "Python executable was not found: $resolvedPythonExe"
    }
    $sourceZip = Resolve-AITrainFirstExistingPath -Candidates (Get-AITrainPortablePythonZipCandidates -Root $script:Root)
    if ([string]::IsNullOrWhiteSpace($sourceZip)) {
        throw "Missing portable Python zip. Expected .deps\archives\python-3.13.13-embed-amd64.zip or legacy .deps\python-3.13.13-embed-amd64.zip."
    }
    New-Item -ItemType Directory -Force $pythonDirFull | Out-Null
    Expand-Archive -Path $sourceZip -DestinationPath $pythonDirFull -Force
    $pth = Join-Path $pythonDirFull "python313._pth"
    (Get-Content $pth) -replace "#import site", "import site" | Set-Content $pth -Encoding ASCII
}

$pipCheck = & $resolvedPythonExe -m pip --version 2>$null
if ($LASTEXITCODE -ne 0) {
    $getPip = Resolve-AITrainFirstExistingPath -Candidates (Get-AITrainGetPipCandidates -Root $script:Root)
    if ([string]::IsNullOrWhiteSpace($getPip)) {
        throw "Missing get-pip.py. Expected .deps\archives\get-pip.py or legacy .deps\get-pip.py."
    }
    & $resolvedPythonExe $getPip
}

if (!(Test-Path (Join-Path $repoFull "tools\train.py"))) {
    git clone https://github.com/PaddlePaddle/PaddleOCR.git $repoFull
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone PaddleOCR repository."
    }
}

if ($PaddleOcrRef) {
    git -C $repoFull fetch --depth 1 origin $PaddleOcrRef
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch PaddleOCR ref: $PaddleOcrRef"
    }
    git -C $repoFull checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to checkout PaddleOCR ref: $PaddleOcrRef"
    }
}
$resolvedPaddleOcrRef = (& git -C $repoFull rev-parse HEAD).Trim()
Write-Host "Using PaddleOCR ref: $resolvedPaddleOcrRef"

if (!$SkipInstall) {
    $pipArgs = @("-m", "pip", "install", "--no-warn-script-location", $PaddlePaddleRequirement, "-r", (Join-Path $repoFull "requirements.txt"))
    if (!$DisablePinnedConstraints) {
        $constraintsPath = Join-Path $workFull "phase16-ocr-constraints.txt"
        @(
            "albumentations==2.0.8",
            "lmdb==2.2.0",
            "numpy==2.4.4",
            "opencv-python==4.13.0.92",
            "paddlepaddle==3.3.1",
            "pillow==12.2.0",
            "pydantic==2.13.3",
            "PyYAML==6.0.3",
            "RapidFuzz==3.14.5",
            "shapely==2.1.2",
            "tqdm==4.67.3"
        ) | Set-Content $constraintsPath -Encoding ASCII
        $pipArgs += @("-c", $constraintsPath)
    }
    & $resolvedPythonExe @pipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install pinned OCR Python dependencies."
    }
}

& $resolvedPythonExe -c "import paddle, albumentations; print('paddle', paddle.__version__); print('albumentations', albumentations.__version__)"
& $resolvedPythonExe (Join-Path $repoFull "tools\train.py") -h | Out-Host

& $resolvedPythonExe "examples\create-minimal-datasets.py" --output $workFull

$requestPath = Join-Path $workFull "paddleocr_rec_official_request.json"
$datasetPath = Join-Path $workFull "paddleocr_rec"
$outputPath = Join-Path $workFull "runs\paddleocr_rec_official"
$request = [ordered]@{
    protocolVersion = 1
    taskId = "phase16-paddleocr-official-smoke"
    taskType = "ocr_recognition"
    datasetPath = $datasetPath
    outputPath = $outputPath
    backend = "paddleocr_rec_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_rec_official"
        modelPreset = "PP-OCRv4_mobile_rec"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        runOfficial = $true
        prepareOnly = $false
        epochs = 1
        batchSize = 1
        imageWidth = 320
        imageHeight = 48
        recImageShape = "3,48,320"
        maxTextLength = 8
        useGpu = $false
        validationRatio = 0.5
        runInferenceAfterExport = $true
        inferenceImage = "images\a.png"
    }
}
$request | ConvertTo-Json -Depth 20 | Set-Content $requestPath -Encoding UTF8

& $resolvedPythonExe "python_trainers\ocr_rec\paddleocr_official_adapter.py" --request $requestPath
if ($LASTEXITCODE -ne 0) {
    throw "Official PaddleOCR adapter failed."
}

$reportPath = Join-Path $outputPath "paddleocr_official_rec_report.json"
$inferenceYml = Join-Path $outputPath "official_inference\inference.yml"
$predictionPath = Join-Path $outputPath "official_prediction.json"
if (!(Test-Path $reportPath)) { throw "Missing report: $reportPath" }
$report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reportPath | ConvertFrom-Json
if ($report.PSObject.Properties.Name -contains "ok" -and -not $report.ok) {
    throw "Official PaddleOCR report has ok=false: $reportPath"
}
$checkpointPath = Join-Path $outputPath "official_model\best_accuracy.pdparams"
if ($report.PSObject.Properties.Name -contains "checkpointPath" -and $report.checkpointPath) {
    if ([System.IO.Path]::IsPathRooted([string]$report.checkpointPath)) {
        $checkpointPath = [string]$report.checkpointPath
    } else {
        $checkpointPath = Join-Path $outputPath ([string]$report.checkpointPath)
    }
}
if (!(Test-Path $checkpointPath)) { throw "Missing official checkpoint: $checkpointPath" }
if (!(Test-Path $inferenceYml)) { throw "Missing official inference model config: $inferenceYml" }
if (!(Test-Path $predictionPath)) { throw "Missing official prediction report: $predictionPath" }
$prediction = Get-Content -Raw -Encoding UTF8 -LiteralPath $predictionPath | ConvertFrom-Json
if ($prediction.PSObject.Properties.Name -contains "ok" -and -not $prediction.ok) {
    throw "Official PaddleOCR prediction has ok=false: $predictionPath"
}

Write-Host "Phase 16 official PaddleOCR smoke passed: $outputPath"
