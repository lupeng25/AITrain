param(
    [string]$WorkDir = ".deps\phase31-paddleocr-full-official-smoke",
    [string]$PythonDir = ".deps\python-3.13.13-ocr-amd64",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$PaddleOcrRef = "f8b41a62bba991d35e578ffa712107a042b0c3b0",
    [string]$PaddlePaddleRequirement = "paddlepaddle==3.3.1",
    [switch]$DisablePinnedConstraints,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

$pythonDirFull = Resolve-RepoPath $PythonDir
$pythonExe = Join-Path $pythonDirFull "python.exe"
$repoFull = Resolve-RepoPath $PaddleOcrRepo
$workFull = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force $workFull | Out-Null

if (!(Test-Path $pythonExe)) {
    if (!(Test-Path ".deps\python-3.13.13-embed-amd64.zip")) {
        throw "Missing .deps\python-3.13.13-embed-amd64.zip. Run the Phase 8 Python setup first."
    }
    New-Item -ItemType Directory -Force $pythonDirFull | Out-Null
    Expand-Archive -Path ".deps\python-3.13.13-embed-amd64.zip" -DestinationPath $pythonDirFull -Force
    $pth = Join-Path $pythonDirFull "python313._pth"
    (Get-Content $pth) -replace "#import site", "import site" | Set-Content $pth -Encoding ASCII
}

if (!(Test-Path (Join-Path $pythonDirFull "Lib\site-packages\pip"))) {
    if (!(Test-Path ".deps\get-pip.py")) {
        throw "Missing .deps\get-pip.py. Run the Phase 8 Python setup first."
    }
    & $pythonExe ".deps\get-pip.py"
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
        $constraintsPath = Join-Path $workFull "phase31-ocr-constraints.txt"
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
    & $pythonExe @pipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install pinned OCR Python dependencies."
    }
}

& $pythonExe -c "import paddle, yaml; print('paddle', paddle.__version__); print('yaml', yaml.__version__)"
& $pythonExe (Join-Path $repoFull "tools\train.py") -h | Out-Host
& $pythonExe (Join-Path $repoFull "tools\infer\predict_system.py") -h | Out-Host

& $pythonExe "examples\create-minimal-datasets.py" --output $workFull

$detRequestPath = Join-Path $workFull "paddleocr_det_official_request.json"
$detDatasetPath = Join-Path $workFull "paddleocr_det"
$detOutputPath = Join-Path $workFull "runs\paddleocr_det_official"
$detRequest = [ordered]@{
    protocolVersion = 1
    taskId = "phase31-paddleocr-det-official-smoke"
    taskType = "ocr_detection"
    datasetPath = $detDatasetPath
    outputPath = $detOutputPath
    backend = "paddleocr_det_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_det_official"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        runOfficial = $true
        prepareOnly = $false
        epochs = 1
        batchSize = 1
        imageSize = 64
        useGpu = $false
        validationRatio = 0.5
        calMetricDuringTrain = $true
    }
}
$detRequest | ConvertTo-Json -Depth 20 | Set-Content $detRequestPath -Encoding UTF8
& $pythonExe "python_trainers\ocr_det\paddleocr_det_official_adapter.py" --request $detRequestPath
if ($LASTEXITCODE -ne 0) {
    throw "Official PaddleOCR Det adapter failed."
}

$recRequestPath = Join-Path $workFull "paddleocr_rec_official_request.json"
$recDatasetPath = Join-Path $workFull "paddleocr_rec"
$recOutputPath = Join-Path $workFull "runs\paddleocr_rec_official"
$recRequest = [ordered]@{
    protocolVersion = 1
    taskId = "phase31-paddleocr-rec-official-smoke"
    taskType = "ocr_recognition"
    datasetPath = $recDatasetPath
    outputPath = $recOutputPath
    backend = "paddleocr_rec_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_rec_official"
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
$recRequest | ConvertTo-Json -Depth 20 | Set-Content $recRequestPath -Encoding UTF8
& $pythonExe "python_trainers\ocr_rec\paddleocr_official_adapter.py" --request $recRequestPath
if ($LASTEXITCODE -ne 0) {
    throw "Official PaddleOCR Rec adapter failed."
}

$systemRequestPath = Join-Path $workFull "paddleocr_system_official_request.json"
$systemOutputPath = Join-Path $workFull "runs\paddleocr_system_official"
$systemRequest = [ordered]@{
    protocolVersion = 1
    taskId = "phase31-paddleocr-system-official-smoke"
    taskType = "ocr"
    datasetPath = (Join-Path $detDatasetPath "images\a.png")
    outputPath = $systemOutputPath
    backend = "paddleocr_system_official"
    parameters = [ordered]@{
        trainingBackend = "paddleocr_system_official"
        paddleOcrRepoPath = $repoFull
        paddleOcrRef = $resolvedPaddleOcrRef
        detModelDir = (Join-Path $detOutputPath "official_inference")
        recModelDir = (Join-Path $recOutputPath "official_inference")
        dictionaryFile = (Join-Path $recOutputPath "official_data\dict.txt")
        inferenceImage = (Join-Path $detDatasetPath "images\a.png")
        dropScore = 0.0
        useGpu = $false
    }
}
$systemRequest | ConvertTo-Json -Depth 20 | Set-Content $systemRequestPath -Encoding UTF8
& $pythonExe "python_trainers\ocr_system\paddleocr_system_official_adapter.py" --request $systemRequestPath
if ($LASTEXITCODE -ne 0) {
    throw "Official PaddleOCR System adapter failed."
}

$requiredPaths = @(
    (Join-Path $detOutputPath "paddleocr_official_det_report.json"),
    (Join-Path $detOutputPath "official_inference\inference.yml"),
    (Join-Path $recOutputPath "paddleocr_official_rec_report.json"),
    (Join-Path $recOutputPath "official_inference\inference.yml"),
    (Join-Path $systemOutputPath "official_system_prediction.json"),
    (Join-Path $systemOutputPath "paddleocr_official_system_report.json")
)
foreach ($path in $requiredPaths) {
    if (!(Test-Path $path)) {
        throw "Missing Phase 31 smoke artifact: $path"
    }
}

$systemPrediction = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $systemOutputPath "official_system_prediction.json") | ConvertFrom-Json
if ($systemPrediction.PSObject.Properties.Name -contains "ok" -and -not $systemPrediction.ok) {
    throw "Official PaddleOCR system prediction has ok=false."
}

Write-Host "Phase 31 official PaddleOCR full smoke passed: $workFull" -ForegroundColor Green
