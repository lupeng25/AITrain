param(
    [string]$WorkDir = ".deps\phase45-yolo-model-matrix",
    [string]$PythonExe = "",
    [string]$BuildDir = "build-vscode",
    [int]$Epochs = 1,
    [int]$ImageSize = 64,
    [int]$Batch = 1,
    [switch]$IncludeYolo12,
    [switch]$SkipCtest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow

function Write-Step {
    param([string]$Message)
    Write-Host "Phase45 YOLO matrix: $Message" -ForegroundColor Cyan
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Resolve-PythonExe {
    if ($PythonExe) {
        $resolved = Resolve-RepoPath $PythonExe
        if (!(Test-Path $resolved)) {
            throw "Python executable was not found: $resolved"
        }
        return $resolved
    }

    $candidates = @(
        (Join-Path $script:Root ".deps\python-3.13.13-embed-amd64\python.exe"),
        (Join-Path $script:Root ".deps\python-3.13.13-ocr-amd64\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }
    throw "Python was not found. Install Python or pass -PythonExe."
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $script:Root
    )

    Write-Step ("{0} {1}" -f $FilePath, ($Arguments -join " "))
    Push-Location $WorkingDirectory
    try {
        if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
            $commandOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
        } else {
            $commandOutput = & $FilePath @Arguments 2>&1
        }
        foreach ($line in $commandOutput) {
            Write-Host $line
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Description
    )
    if (!(Test-Path $Path)) {
        throw "Missing $Description`: $Path"
    }
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force $parent | Out-Null
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-UltralyticsVersion {
    param([string]$Python)
    $code = "import ultralytics; print(getattr(ultralytics, '__version__', 'unknown'))"
    $output = & $Python -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Ultralytics is not available. Install python_trainers\requirements-yolo.txt or pass -PythonExe."
    }
    return [string]($output | Select-Object -Last 1)
}

function Test-UltralyticsModel {
    param(
        [string]$Python,
        [string]$Model
    )
    $code = "import sys; from ultralytics import YOLO; model=YOLO(sys.argv[1]); print(getattr(model, 'task', 'unknown'))"
    $output = & $Python -c $code $Model
    if ($LASTEXITCODE -ne 0) {
        throw "Ultralytics could not resolve model $Model"
    }
    return [string]($output | Select-Object -Last 1)
}

function Get-TrainingReportSummary {
    param(
        [string]$ReportPath,
        [string]$ExpectedBackend,
        [string]$ExpectedModel
    )

    Assert-PathExists $ReportPath "training report"
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $ReportPath | ConvertFrom-Json
    if (($report.PSObject.Properties.Name -contains "ok") -and -not $report.ok) {
        throw "Training report reports ok=false: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "backend") -or [string]$report.backend -ne $ExpectedBackend) {
        throw "Report backend mismatch for $ExpectedModel`: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "model") -or [string]$report.model -ne $ExpectedModel) {
        throw "Report model mismatch for $ExpectedModel`: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "metrics")) {
        throw "Report is missing metrics: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "checkpointPath") -or ![string]$report.checkpointPath) {
        throw "Report is missing checkpointPath: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "onnxPath") -or ![string]$report.onnxPath) {
        throw "Report is missing onnxPath: $ReportPath"
    }

    $checkpointPath = [string]$report.checkpointPath
    $onnxPath = [string]$report.onnxPath
    Assert-PathExists $checkpointPath "checkpointPath"
    Assert-PathExists $onnxPath "onnxPath"

    return [pscustomobject][ordered]@{
        reportPath = [System.IO.Path]::GetFullPath($ReportPath)
        checkpointPath = [System.IO.Path]::GetFullPath($checkpointPath)
        onnxPath = [System.IO.Path]::GetFullPath($onnxPath)
        metrics = $report.metrics
    }
}

function New-YoloRequest {
    param(
        [string]$Task,
        [string]$Backend,
        [string]$Model,
        [string]$DatasetPath,
        [string]$OutputPath,
        [string]$RunName
    )

    return [ordered]@{
        protocolVersion = 1
        taskId = "phase45-$Task-$($Model -replace '[^A-Za-z0-9]+', '-')"
        taskType = $Task
        datasetPath = $DatasetPath
        outputPath = $OutputPath
        backend = $Backend
        parameters = [ordered]@{
            trainingBackend = $Backend
            model = $Model
            epochs = $Epochs
            batchSize = $Batch
            imageSize = $ImageSize
            device = "cpu"
            workers = 0
            runName = $RunName
            compactEvents = $true
        }
    }
}

function Invoke-MatrixCase {
    param(
        [string]$Python,
        [object]$Case,
        [string]$GeneratedRoot,
        [string]$RunsRoot
    )

    $started = [DateTime]::UtcNow
    $caseName = [string]$Case.name
    $model = [string]$Case.model
    $task = [string]$Case.task
    $backend = [string]$Case.backend
    $required = [bool]$Case.required

    Write-Step "probe $model"
    $resolvedTask = Test-UltralyticsModel -Python $Python -Model $model

    $datasetPath = if ($task -eq "segmentation") {
        Join-Path $GeneratedRoot "yolo_segment"
    } else {
        Join-Path $GeneratedRoot "yolo_detect"
    }
    $trainer = if ($task -eq "segmentation") {
        Join-Path $script:Root "python_trainers\segmentation\ultralytics_trainer.py"
    } else {
        Join-Path $script:Root "python_trainers\detection\ultralytics_trainer.py"
    }
    $outputPath = Join-Path $RunsRoot $caseName
    $requestPath = Join-Path $RunsRoot "$caseName-request.json"
    $requestTask = if ($task -eq "segmentation") { "segmentation" } else { "detection" }
    $request = New-YoloRequest -Task $requestTask -Backend $backend -Model $model -DatasetPath $datasetPath -OutputPath $outputPath -RunName $caseName
    Write-JsonFile -Path $requestPath -Value $request

    Write-Step "train/export $model"
    Invoke-Checked -FilePath $Python -Arguments @($trainer, "--request", $requestPath)

    $reportPath = Join-Path $outputPath "ultralytics_training_report.json"
    $reportSummary = Get-TrainingReportSummary -ReportPath $reportPath -ExpectedBackend $backend -ExpectedModel $model
    $finished = [DateTime]::UtcNow

    Write-Host ("  [ok] {0}: checkpoint and ONNX verified" -f $model)
    return [ordered]@{
        name = $caseName
        task = $task
        backend = $backend
        model = $model
        required = $required
        status = "passed"
        resolvedUltralyticsTask = $resolvedTask
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        artifacts = $reportSummary
        failureCategory = ""
        failure = ""
    }
}

function Invoke-CtestForWorkDir {
    param([string]$WorkRoot)

    if ($SkipCtest) {
        Write-Host "  [skip] CTest skipped by -SkipCtest"
        return "skipped"
    }

    $ctestFile = Join-Path $script:Root "$BuildDir\CTestTestfile.cmake"
    if (!(Test-Path $ctestFile)) {
        Write-Host "  [warn] CTest build directory not found; skipping C++ ONNX inference regression check." -ForegroundColor Yellow
        return "skipped-build-dir-missing"
    }

    $previousAcceptanceSmokeRoot = $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT
    $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $WorkRoot
    try {
        Invoke-Checked -FilePath "ctest" -Arguments @("--test-dir", (Join-Path $script:Root $BuildDir), "--output-on-failure", "--timeout", "360")
        return "passed"
    } finally {
        if ($null -eq $previousAcceptanceSmokeRoot) {
            Remove-Item Env:\AITRAIN_ACCEPTANCE_SMOKE_ROOT -ErrorAction SilentlyContinue
        } else {
            $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $previousAcceptanceSmokeRoot
        }
    }
}

$python = Resolve-PythonExe
$work = Resolve-RepoPath $WorkDir
$generated = Join-Path $work "generated"
$runs = Join-Path $work "runs"
$summaryPath = Join-Path $work "yolo_model_matrix_summary.json"
New-Item -ItemType Directory -Force $work | Out-Null

$ultralyticsVersion = Get-UltralyticsVersion -Python $python
Write-Host ("  [ok] Ultralytics version={0}" -f $ultralyticsVersion)

$generator = Join-Path $script:Root "examples\create-minimal-datasets.py"
Assert-PathExists $generator "minimal dataset generator"
Invoke-Checked -FilePath $python -Arguments @($generator, "--output", $generated)

$cases = @(
    [pscustomobject]@{ name = "yolo11n-detect"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolo11n.yaml"; required = $true },
    [pscustomobject]@{ name = "yolo11n-segment"; task = "segmentation"; backend = "ultralytics_yolo_segment"; model = "yolo11n-seg.yaml"; required = $true }
)
if ($IncludeYolo12) {
    $cases += @(
        [pscustomobject]@{ name = "yolo12n-detect"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolo12n.yaml"; required = $true },
        [pscustomobject]@{ name = "yolo12n-segment"; task = "segmentation"; backend = "ultralytics_yolo_segment"; model = "yolo12n-seg.yaml"; required = $true }
    )
}

$results = @()
foreach ($case in $cases) {
    try {
        $results += Invoke-MatrixCase -Python $python -Case $case -GeneratedRoot $generated -RunsRoot $runs
    } catch {
        $finished = [DateTime]::UtcNow
        $message = $_.Exception.Message
        Write-Host ("  [fail] {0}: {1}" -f $case.model, $message) -ForegroundColor Red
        $results += [pscustomobject][ordered]@{
            name = [string]$case.name
            task = [string]$case.task
            backend = [string]$case.backend
            model = [string]$case.model
            required = [bool]$case.required
            status = "failed"
            resolvedUltralyticsTask = ""
            startedAt = ""
            finishedAt = $finished.ToString("o")
            elapsedSeconds = 0
            artifacts = [ordered]@{}
            failureCategory = "model-matrix-smoke-failed"
            failure = $message
        }
    }
}

$requiredFailures = @($results | Where-Object { $_.required -and $_.status -ne "passed" })
$ctestStatus = if ($requiredFailures.Count -eq 0) {
    Invoke-CtestForWorkDir -WorkRoot $work
} else {
    "skipped-required-case-failed"
}
$finishedAt = [DateTime]::UtcNow
$status = if ($requiredFailures.Count -eq 0 -and $ctestStatus -ne "failed") { "passed" } else { "failed" }

$summary = [ordered]@{
    ok = ($status -eq "passed")
    phase = "45"
    status = $status
    workDir = $work
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    ultralyticsVersion = $ultralyticsVersion
    parameters = [ordered]@{
        epochs = $Epochs
        batchSize = $Batch
        imageSize = $ImageSize
        device = "cpu"
        includeYolo12 = [bool]$IncludeYolo12
    }
    ctestStatus = $ctestStatus
    results = $results
    note = "Phase 45 validates Ultralytics YOLO detection/segmentation model-family wiring and artifacts; it is not an accuracy benchmark and does not change TensorRT external acceptance."
}
$summary | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host ("  [ok] summary={0}" -f $summaryPath)

if ($requiredFailures.Count -gt 0) {
    throw "Phase 45 YOLO model matrix failed for required models: $($requiredFailures.model -join ', ')"
}
Write-Host "Phase45 YOLO matrix passed" -ForegroundColor Green
