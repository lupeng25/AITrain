param(
    [string]$WorkDir = ".deps\phase16-ocr-official-smoke",
    [string]$PythonDir = ".deps\python-3.13.13-ocr-amd64",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
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
    git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git $repoFull
}

if (!$SkipInstall) {
    & $pythonExe -m pip install --no-warn-script-location "paddlepaddle>=3.3,<4" -r (Join-Path $repoFull "requirements.txt")
}

& $pythonExe -c "import paddle, albumentations; print('paddle', paddle.__version__); print('albumentations', albumentations.__version__)"
& $pythonExe (Join-Path $repoFull "tools\train.py") -h | Out-Host

New-Item -ItemType Directory -Force $workFull | Out-Null
& $pythonExe "examples\create-minimal-datasets.py" --output $workFull

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
        paddleOcrRepoPath = $repoFull
        runOfficial = $true
        prepareOnly = $false
        epochs = 1
        batchSize = 1
        imageWidth = 320
        imageHeight = 48
        maxTextLength = 8
        useGpu = $false
        validationRatio = 0.5
    }
}
$request | ConvertTo-Json -Depth 20 | Set-Content $requestPath -Encoding UTF8

& $pythonExe "python_trainers\ocr_rec\paddleocr_official_adapter.py" --request $requestPath

$reportPath = Join-Path $outputPath "paddleocr_official_rec_report.json"
$checkpointPath = Join-Path $outputPath "official_model\best_accuracy.pdparams"
$inferenceYml = Join-Path $outputPath "official_inference\inference.yml"
if (!(Test-Path $reportPath)) { throw "Missing report: $reportPath" }
if (!(Test-Path $checkpointPath)) { throw "Missing official checkpoint: $checkpointPath" }
if (!(Test-Path $inferenceYml)) { throw "Missing official inference model config: $inferenceYml" }

Write-Host "Phase 16 official PaddleOCR smoke passed: $outputPath"
