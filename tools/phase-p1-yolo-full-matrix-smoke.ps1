param(
    [string]$WorkDir = ".deps\phase-p1-yolo-full-matrix",
    [string]$PythonExe = "",
    [string]$BuildDir = "build-vscode",
    [string]$WorkerExe = "",
    [int]$Epochs = 1,
    [int]$ImageSize = 64,
    [int]$Batch = 1,
    [string]$Device = "cpu",
    [switch]$SkipCtest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow
. (Join-Path $PSScriptRoot "toolchain-env.ps1")
Set-AITrainQtRuntimeEnvironment

function Write-Step {
    param([string]$Message)
    Write-Host "P1 YOLO full matrix: $Message" -ForegroundColor Cyan
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

function Resolve-WorkerExe {
    if ($WorkerExe) {
        $resolved = Resolve-RepoPath $WorkerExe
        if (!(Test-Path $resolved)) {
            throw "Worker executable was not found: $resolved"
        }
        return $resolved
    }

    $candidates = @(
        (Join-Path $script:Root "$BuildDir\bin\aitrain_worker.exe"),
        (Join-Path $script:Root "aitrain_worker.exe"),
        (Join-Path $script:Root "$BuildDir\package-smoke\aitrain_worker.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "aitrain_worker.exe was not found. Build first or pass -WorkerExe."
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $script:Root
    )

    Write-Step ("{0} {1}" -f $FilePath, ($Arguments -join " "))
    Push-Location $WorkingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
            $commandOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
        } else {
            $commandOutput = & $FilePath @Arguments 2>&1
        }
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        foreach ($line in $commandOutput) {
            Write-Host $line
        }
        if ($exitCode -ne 0) {
            throw "Command failed with exit code $exitCode`: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
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
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-Utf8NoBomEncoding {
    return (New-Object System.Text.UTF8Encoding $false)
}

function ConvertTo-ProtocolLine {
    param(
        [string]$Type,
        [object]$Payload
    )

    return ([ordered]@{
        type = $Type
        payload = $Payload
    } | ConvertTo-Json -Depth 50 -Compress)
}

function Invoke-WorkerTraining {
    param(
        [string]$ResolvedWorkerExe,
        [object]$Request,
        [string]$EventsPath
    )

    $eventsParent = Split-Path -Parent $EventsPath
    New-Item -ItemType Directory -Force $eventsParent | Out-Null
    if (Test-Path -LiteralPath $EventsPath) {
        Remove-Item -LiteralPath $EventsPath -Force
    }

    $pipeName = "aitrain_p1_" + ([guid]::NewGuid().ToString("N"))
    $pipe = $null
    $worker = $null
    $reader = $null
    $writer = $null
    $terminalEvent = $null
    $startSent = $false
    $utf8 = New-Utf8NoBomEncoding

    try {
        $pipe = [System.IO.Pipes.NamedPipeServerStream]::new(
            $pipeName,
            [System.IO.Pipes.PipeDirection]::InOut,
            1,
            [System.IO.Pipes.PipeTransmissionMode]::Byte,
            [System.IO.Pipes.PipeOptions]::Asynchronous)

        $worker = Start-Process `
            -FilePath $ResolvedWorkerExe `
            -ArgumentList @("--server", $pipeName) `
            -WorkingDirectory (Split-Path -Parent $ResolvedWorkerExe) `
            -WindowStyle Hidden `
            -PassThru

        $connectWait = $pipe.BeginWaitForConnection($null, $null)
        if (-not $connectWait.AsyncWaitHandle.WaitOne(15000)) {
            throw "Worker did not connect to command pipe within 15 seconds."
        }
        $pipe.EndWaitForConnection($connectWait)
        if ($pipe.CanTimeout) {
            $pipe.ReadTimeout = 1000
            $pipe.WriteTimeout = 1000
        }

        $reader = [System.IO.StreamReader]::new($pipe, [System.Text.Encoding]::UTF8, $false, 4096, $true)
        $writer = [System.IO.StreamWriter]::new($pipe, $utf8, 4096, $true)
        $writer.AutoFlush = $true

        $requestLine = ConvertTo-ProtocolLine -Type "startTrain" -Payload $Request
        while ($true) {
            $line = $null
            try {
                $line = $reader.ReadLine()
            } catch [System.IO.IOException] {
                if ($worker.HasExited) {
                    throw "Worker exited before emitting a terminal training event. ExitCode=$($worker.ExitCode)"
                }
                continue
            }
            if ($null -eq $line) {
                if ($worker.HasExited) {
                    throw "Worker pipe closed before emitting a terminal training event. ExitCode=$($worker.ExitCode)"
                }
                continue
            }
            [System.IO.File]::AppendAllText($EventsPath, $line + [Environment]::NewLine, $utf8)

            try {
                $event = $line | ConvertFrom-Json
            } catch {
                continue
            }
            $eventType = [string]$event.type
            if (-not $startSent -and $eventType -eq "ready") {
                $writer.WriteLine($requestLine)
                $startSent = $true
                continue
            }
            if ($eventType -in @("completed", "failed", "canceled")) {
                $terminalEvent = $event
                break
            }
        }

        if (-not $startSent) {
            throw "Worker connected but did not emit a ready event."
        }
        if ($null -eq $terminalEvent) {
            throw "Worker did not emit a terminal training event."
        }
        if ([string]$terminalEvent.type -ne "completed") {
            $message = ""
            if ($terminalEvent.PSObject.Properties.Name -contains "payload") {
                $message = [string]$terminalEvent.payload.message
            }
            throw "Worker training ended with $($terminalEvent.type): $message"
        }

        if (-not $worker.WaitForExit(10000)) {
            Stop-Process -Id $worker.Id -Force
            throw "Worker emitted completed but did not exit cleanly."
        }
        if ($worker.ExitCode -ne 0) {
            throw "Worker exited with code $($worker.ExitCode) after completed event."
        }
        return $terminalEvent
    } finally {
        if ($writer) { $writer.Dispose() }
        if ($reader) { $reader.Dispose() }
        if ($pipe) { $pipe.Dispose() }
        if ($worker -and -not $worker.HasExited) {
            Stop-Process -Id $worker.Id -Force
        }
    }
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

function New-P1Cases {
    $families = @("yolov8", "yolo11", "yolo12")
    $scales = @("n", "s", "m", "l", "x")
    $sourceTypes = @("yaml", "pt")
    $cases = @()
    foreach ($family in $families) {
        foreach ($scale in $scales) {
            foreach ($sourceType in $sourceTypes) {
                $model = "$family$scale.$sourceType"
                $cases += [pscustomobject]@{
                    name = "$family$scale-detect-$sourceType"
                    task = "detection"
                    backend = "ultralytics_yolo_detect"
                    model = $model
                    sourceType = $sourceType
                    required = $true
                }
                $segModel = "$family$scale-seg.$sourceType"
                $cases += [pscustomobject]@{
                    name = "$family$scale-segment-$sourceType"
                    task = "segmentation"
                    backend = "ultralytics_yolo_segment"
                    model = $segModel
                    sourceType = $sourceType
                    required = $true
                }
            }
        }
    }
    foreach ($scale in $scales) {
        foreach ($variant in @("p2", "p6")) {
            $model = "yolov8$scale-$variant.yaml"
            $cases += [pscustomobject]@{
                name = "yolov8$scale-$variant-detect-yaml"
                task = "detection"
                backend = "ultralytics_yolo_detect"
                model = $model
                sourceType = "yaml"
                required = $true
            }
        }
    }
    return $cases
}

function New-YoloRequest {
    param(
        [object]$Case,
        [string]$Python,
        [string]$DatasetPath,
        [string]$OutputPath
    )

    return [ordered]@{
        protocolVersion = 1
        taskId = "p1-$($Case.name)"
        taskType = [string]$Case.task
        datasetPath = $DatasetPath
        outputPath = $OutputPath
        backend = [string]$Case.backend
        parameters = [ordered]@{
            trainingBackend = [string]$Case.backend
            model = [string]$Case.model
            modelPreset = [string]$Case.model
            epochs = $Epochs
            batchSize = $Batch
            imageSize = $ImageSize
            device = $Device
            pythonExecutable = $Python
            workers = 0
            runName = [string]$Case.name
            compactEvents = $true
            ultralyticsExportArgs = [ordered]@{
                format = "onnx"
                dynamic = $false
                half = $false
                int8 = $false
                imgsz = $ImageSize
                batch = $Batch
                device = $Device
            }
        }
    }
}

function Get-TrainingReportSummary {
    param(
        [string]$ReportPath,
        [string]$ExpectedBackend,
        [string]$ExpectedModel,
        [string]$SourceType
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
    if ($SourceType -eq "pt" -and -not ([string]$report.model).EndsWith(".pt")) {
        throw "Report model does not preserve .pt source path for $ExpectedModel`: $ReportPath"
    }
    foreach ($property in @("metrics", "checkpointPath", "onnxPath", "ultralyticsExportArgs")) {
        if (!($report.PSObject.Properties.Name -contains $property)) {
            throw "Report is missing $property`: $ReportPath"
        }
    }

    $checkpointPath = [string]$report.checkpointPath
    $onnxPath = [string]$report.onnxPath
    Assert-PathExists $checkpointPath "checkpointPath"
    Assert-PathExists $onnxPath "onnxPath"

    return [pscustomobject][ordered]@{
        reportPath = [System.IO.Path]::GetFullPath($ReportPath)
        checkpointPath = [System.IO.Path]::GetFullPath($checkpointPath)
        onnxPath = [System.IO.Path]::GetFullPath($onnxPath)
        ultralyticsExportArgs = $report.ultralyticsExportArgs
        metrics = $report.metrics
    }
}

function Invoke-MatrixCase {
    param(
        [string]$Python,
        [string]$ResolvedWorkerExe,
        [object]$Case,
        [string]$GeneratedRoot,
        [string]$RunsRoot
    )

    $started = [DateTime]::UtcNow
    $model = [string]$Case.model
    $task = [string]$Case.task
    $backend = [string]$Case.backend

    Write-Step "probe $model"
    $resolvedTask = Test-UltralyticsModel -Python $Python -Model $model

    $datasetPath = if ($task -eq "segmentation") {
        Join-Path $GeneratedRoot "yolo_segment"
    } else {
        Join-Path $GeneratedRoot "yolo_detect"
    }
    $outputPath = Join-Path $RunsRoot ([string]$Case.name)
    $requestPath = Join-Path $RunsRoot "$($Case.name)-request.json"
    $eventsPath = Join-Path $RunsRoot "$($Case.name)-worker-events.jsonl"
    $request = New-YoloRequest -Case $Case -Python $Python -DatasetPath $datasetPath -OutputPath $outputPath
    Write-JsonFile -Path $requestPath -Value $request

    Write-Step "worker train/export $model"
    $completedEvent = Invoke-WorkerTraining -ResolvedWorkerExe $ResolvedWorkerExe -Request $request -EventsPath $eventsPath

    $reportPath = Join-Path $outputPath "ultralytics_training_report.json"
    $reportSummary = Get-TrainingReportSummary -ReportPath $reportPath -ExpectedBackend $backend -ExpectedModel $model -SourceType ([string]$Case.sourceType)
    $finished = [DateTime]::UtcNow

    Write-Host ("  [ok] {0}: checkpoint and ONNX verified" -f $model)
    return [ordered]@{
        name = [string]$Case.name
        task = $task
        backend = $backend
        model = $model
        sourceType = [string]$Case.sourceType
        required = [bool]$Case.required
        status = "passed"
        resolvedUltralyticsTask = $resolvedTask
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        artifacts = $reportSummary
        workerEventsPath = [System.IO.Path]::GetFullPath($eventsPath)
        completedEvent = $completedEvent
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
$workerExeResolved = Resolve-WorkerExe
$work = Resolve-RepoPath $WorkDir
$generated = Join-Path $work "generated"
$runs = Join-Path $work "runs"
$summaryPath = Join-Path $work "p1_yolo_full_matrix_summary.json"
New-Item -ItemType Directory -Force $work | Out-Null

$ultralyticsVersion = Get-UltralyticsVersion -Python $python
Write-Host ("  [ok] Ultralytics version={0}" -f $ultralyticsVersion)

$generator = Join-Path $script:Root "examples\create-minimal-datasets.py"
Assert-PathExists $generator "minimal dataset generator"
Invoke-Checked -FilePath $python -Arguments @($generator, "--output", $generated)

$cases = New-P1Cases
$results = @()
foreach ($case in $cases) {
    try {
        $results += Invoke-MatrixCase -Python $python -ResolvedWorkerExe $workerExeResolved -Case $case -GeneratedRoot $generated -RunsRoot $runs
    } catch {
        $finished = [DateTime]::UtcNow
        $message = $_.Exception.Message
        Write-Host ("  [fail] {0}: {1}" -f $case.model, $message) -ForegroundColor Red
        $results += [pscustomobject][ordered]@{
            name = [string]$case.name
            task = [string]$case.task
            backend = [string]$case.backend
            model = [string]$case.model
            sourceType = [string]$case.sourceType
            required = [bool]$case.required
            status = "failed"
            resolvedUltralyticsTask = ""
            startedAt = ""
            finishedAt = $finished.ToString("o")
            elapsedSeconds = 0
            artifacts = [ordered]@{}
            workerEventsPath = ""
            completedEvent = $null
            failureCategory = "p1-yolo-full-matrix-failed"
            failure = $message
        }
    }
}

$requiredFailures = @($results | Where-Object { $_.required -and $_.status -ne "passed" })
$ctestFailure = ""
if ($requiredFailures.Count -eq 0) {
    try {
        $ctestStatus = Invoke-CtestForWorkDir -WorkRoot $work
    } catch {
        $ctestStatus = "failed"
        $ctestFailure = $_.Exception.Message
        Write-Host ("  [fail] CTest: {0}" -f $ctestFailure) -ForegroundColor Red
    }
} else {
    $ctestStatus = "skipped-required-case-failed"
}
$finishedAt = [DateTime]::UtcNow
$status = if ($requiredFailures.Count -eq 0 -and $ctestStatus -ne "failed") { "passed" } else { "failed" }

$summary = [ordered]@{
    ok = ($status -eq "passed")
    phase = "p1-yolo-full-matrix"
    status = $status
    workDir = $work
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    ultralyticsVersion = $ultralyticsVersion
    workerExe = $workerExeResolved
    parameters = [ordered]@{
        epochs = $Epochs
        batchSize = $Batch
        imageSize = $ImageSize
        device = $Device
        requiredCaseCount = $cases.Count
    }
    ctestStatus = $ctestStatus
    ctestFailure = $ctestFailure
    results = $results
    note = "P1 validates full YOLOv8/YOLO11/YOLO12 detection and instance-segmentation preset productization, including YAML and .pt fine-tuning entries plus YOLOv8 P2/P6 detection YAML architectures. It is not an accuracy benchmark and does not add YOLO26, semantic segmentation, tracking, YOLOE, classification, pose, or OBB."
}
$summary | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host ("  [ok] summary={0}" -f $summaryPath)

if ($requiredFailures.Count -gt 0) {
    throw "P1 YOLO full matrix failed for required models: $($requiredFailures.model -join ', ')"
}
if ($ctestStatus -eq "failed") {
    throw "P1 YOLO full matrix CTest step failed: $ctestFailure"
}
Write-Host "P1 YOLO full matrix passed" -ForegroundColor Green
