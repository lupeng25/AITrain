param(
    [string]$WorkDir = ".deps\phase-yolo26-model-matrix",
    [string]$PythonExe = "",
    [string]$Yolo26PythonDir = ".deps\yolo26\env",
    [string]$UltralyticsRequirement = "ultralytics",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$BuildDir = "build-vscode",
    [string]$WorkerExe = "",
    [int]$Epochs = 1,
    [int]$ImageSize = 64,
    [int]$Batch = 1,
    [string]$Device = "cpu",
    [string[]]$DeploymentTargets = @("onnx", "tensorrt"),
    [switch]$PrepareEnvironment,
    [switch]$ProbeOnly,
    [switch]$Focused,
    [switch]$Full,
    [string[]]$CaseName = @(),
    [switch]$SkipTorchInstall,
    [switch]$SkipCtest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow
. (Join-Path $PSScriptRoot "toolchain-env.ps1")
Set-AITrainQtRuntimeEnvironment
$env:YOLO_AUTOINSTALL = "false"

function Write-Step {
    param([string]$Message)
    Write-Host "YOLO26 matrix: $Message" -ForegroundColor Cyan
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Test-DeviceRequiresCuda {
    param([string]$DeviceValue)
    $normalized = $DeviceValue.Trim().ToLowerInvariant()
    return ($normalized -ne "cpu" -and $normalized -ne "-1")
}

function Resolve-BasePythonForVenv {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @($py.Source, "-3.12")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }
    throw "No base Python was found to create the isolated YOLO26 environment."
}

function Ensure-Yolo26Environment {
    $envDir = Resolve-RepoPath $Yolo26PythonDir
    $venvPython = Join-Path $envDir "Scripts\python.exe"
    if (!(Test-Path -LiteralPath $venvPython)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $envDir) | Out-Null
        $base = Resolve-BasePythonForVenv
        $baseExe = [string]$base[0]
        $baseArgs = @($base | Select-Object -Skip 1) + @("-m", "venv", $envDir)
        Invoke-Checked -FilePath $baseExe -Arguments $baseArgs
    }
    if (!(Test-Path -LiteralPath $venvPython)) {
        throw "YOLO26 environment creation did not produce python.exe: $venvPython"
    }
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    if (!$SkipTorchInstall -and $TorchIndexUrl) {
        Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--index-url", $TorchIndexUrl, "torch", "torchvision", "torchaudio")
    }
    Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", $UltralyticsRequirement, "onnx", "onnxruntime", "opencv-python")
    return [System.IO.Path]::GetFullPath($venvPython)
}

function Resolve-PythonExe {
    if ($PythonExe) {
        $resolved = Resolve-RepoPath $PythonExe
        if (!(Test-Path $resolved)) {
            throw "Python executable was not found: $resolved"
        }
        return $resolved
    }

    $isolated = Join-Path (Resolve-RepoPath $Yolo26PythonDir) "Scripts\python.exe"
    if (Test-Path -LiteralPath $isolated) {
        return [System.IO.Path]::GetFullPath($isolated)
    }
    if ($PrepareEnvironment) {
        return Ensure-Yolo26Environment
    }

    $candidates = @(
        (Join-Path $script:Root ".deps\yolo26\env\Scripts\python.exe"),
        (Join-Path $script:Root ".deps\rtx4090-validation\python-yolo-cuda\Scripts\python.exe"),
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

function Invoke-ProcessCapture {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$LogPath,
        [string]$WorkingDirectory = $script:Root,
        [switch]$AllowFailure
    )

    New-Item -ItemType Directory -Force (Split-Path -Parent $LogPath) | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
            $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
        } else {
            Push-Location $WorkingDirectory
            try {
                $output = & $FilePath @Arguments 2>&1
            } finally {
                Pop-Location
            }
        }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $text = ([string[]]$output -join [Environment]::NewLine).Trim()
    $text | Set-Content -LiteralPath $LogPath -Encoding UTF8
    if ($exitCode -ne 0 -and !$AllowFailure) {
        throw "Command failed with exit code $exitCode`: $FilePath $($Arguments -join ' ')"
    }
    return [pscustomobject][ordered]@{
        exitCode = $exitCode
        text = $text
        json = Get-LastJsonObjectFromText -Text $text
        logPath = $LogPath
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
    $Value | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $Path -Encoding UTF8
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

function Invoke-WorkerCommand {
    param(
        [string]$ResolvedWorkerExe,
        [string]$CommandType,
        [object]$Request,
        [string]$EventsPath
    )

    $eventsParent = Split-Path -Parent $EventsPath
    New-Item -ItemType Directory -Force $eventsParent | Out-Null
    if (Test-Path -LiteralPath $EventsPath) {
        Remove-Item -LiteralPath $EventsPath -Force
    }

    $pipeName = "aitrain_yolo26_" + ([guid]::NewGuid().ToString("N"))
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
        $requestLine = ConvertTo-ProtocolLine -Type $CommandType -Payload $Request

        while ($true) {
            $line = $null
            try {
                $line = $reader.ReadLine()
            } catch [System.IO.IOException] {
                if ($worker.HasExited) {
                    throw "Worker exited before emitting a terminal event. ExitCode=$($worker.ExitCode)"
                }
                continue
            }
            if ($null -eq $line) {
                if ($worker.HasExited) {
                    throw "Worker pipe closed before emitting a terminal event. ExitCode=$($worker.ExitCode)"
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
            throw "Worker did not emit a terminal event."
        }
        if ([string]$terminalEvent.type -ne "completed") {
            $message = ""
            if ($terminalEvent.PSObject.Properties.Name -contains "payload") {
                $message = [string]$terminalEvent.payload.message
            }
            throw "Worker $CommandType ended with $($terminalEvent.type): $message"
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

function Get-LastJsonObjectFromText {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }
    $lines = $Text -split "(`r`n|`n|`r)"
    for ($index = $lines.Count - 1; $index -ge 0; $index--) {
        $line = [string]$lines[$index]
        if ($line -notmatch '^\s*\{') {
            continue
        }
        try {
            return $line | ConvertFrom-Json
        } catch {
            continue
        }
    }
    return $null
}

function Invoke-Yolo26EnvironmentProbe {
    param(
        [string]$Python,
        [bool]$RequireCuda
    )

    $probeModels = @("yolo26n.pt", "yolo26n-seg.pt", "yolo26n.yaml", "yolo26n-seg.yaml")
    $code = @'
import json
import sys
from pathlib import Path

models = sys.argv[1:]
result = {
    "ok": False,
    "python": sys.executable,
    "ultralyticsVersion": "",
    "ultralyticsModule": "",
    "torchVersion": "",
    "torchCudaAvailable": False,
    "torchCudaDeviceCount": 0,
    "cfgModels26Exists": False,
    "modelProbes": [],
    "errors": [],
}
try:
    import torch
    result["torchVersion"] = getattr(torch, "__version__", "unknown")
    result["torchCudaAvailable"] = bool(torch.cuda.is_available())
    result["torchCudaDeviceCount"] = int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0
except Exception as exc:
    result["errors"].append({"stage": "torch_import", "message": str(exc)})
try:
    import ultralytics
    from ultralytics import YOLO
    result["ultralyticsVersion"] = getattr(ultralytics, "__version__", "unknown")
    result["ultralyticsModule"] = str(getattr(ultralytics, "__file__", ""))
    cfg_root = Path(result["ultralyticsModule"]).resolve().parent / "cfg" / "models" / "26"
    result["cfgModels26Exists"] = cfg_root.exists()
    for model_name in models:
        probe = {"model": model_name, "ok": False, "task": "", "end2end": None, "error": ""}
        try:
            model = YOLO(model_name)
            probe["task"] = str(getattr(model, "task", ""))
            yaml = getattr(getattr(model, "model", None), "yaml", {})
            if isinstance(yaml, dict) and "end2end" in yaml:
                probe["end2end"] = bool(yaml.get("end2end"))
            probe["ok"] = True
        except Exception as exc:
            probe["error"] = str(exc)
        result["modelProbes"].append(probe)
except Exception as exc:
    result["errors"].append({"stage": "ultralytics_import_or_probe", "message": str(exc)})

result["ok"] = (
    bool(result["ultralyticsVersion"])
    and bool(result["cfgModels26Exists"])
    and all(item.get("ok") for item in result["modelProbes"])
)
print(json.dumps(result, ensure_ascii=False))
'@
    $previousErrorActionPreference = $ErrorActionPreference
    $probeScript = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
    try {
        Set-Content -LiteralPath $probeScript -Encoding UTF8 -Value $code
        $ErrorActionPreference = "Continue"
        $output = & $Python $probeScript @probeModels 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Remove-Item -LiteralPath $probeScript -Force -ErrorAction SilentlyContinue
    }
    $text = ([string[]]$output -join [Environment]::NewLine)
    $probe = Get-LastJsonObjectFromText -Text $text
    if ($null -eq $probe) {
        $probe = [pscustomobject][ordered]@{
            ok = $false
            python = $Python
            exitCode = $exitCode
            errors = @([pscustomobject]@{ stage = "probe_output"; message = $text })
            modelProbes = @()
        }
    } else {
        Add-Member -InputObject $probe -NotePropertyName "exitCode" -NotePropertyValue $exitCode -Force
        Add-Member -InputObject $probe -NotePropertyName "rawOutputTail" -NotePropertyValue ($text.Substring([Math]::Max(0, $text.Length - 4096))) -Force
    }
    $cudaOk = (-not $RequireCuda) -or [bool]$probe.torchCudaAvailable
    Add-Member -InputObject $probe -NotePropertyName "requiresCuda" -NotePropertyValue $RequireCuda -Force
    if (-not $cudaOk) {
        Add-Member -InputObject $probe -NotePropertyName "ok" -NotePropertyValue $false -Force
        $errors = @($probe.errors)
        $errors += [pscustomobject]@{ stage = "torch_cuda"; message = "Requested GPU device but torch.cuda.is_available() is false." }
        Add-Member -InputObject $probe -NotePropertyName "errors" -NotePropertyValue $errors -Force
    }
    return $probe
}

function New-Yolo26Cases {
    $scales = if ($Focused -and !$Full) { @("n") } else { @("n", "s", "m", "l", "x") }
    $sourceTypes = @("yaml", "pt")
    $cases = @()
    foreach ($scale in $scales) {
        foreach ($sourceType in $sourceTypes) {
            $cases += [pscustomobject]@{
                name = "yolo26$scale-detect-$sourceType"
                task = "detection"
                backend = "ultralytics_yolo_detect"
                model = "yolo26$scale.$sourceType"
                sourceType = $sourceType
                end2end = "auto"
                required = $true
            }
            $cases += [pscustomobject]@{
                name = "yolo26$scale-segment-$sourceType"
                task = "segmentation"
                backend = "ultralytics_yolo_segment"
                model = "yolo26$scale-seg.$sourceType"
                sourceType = $sourceType
                end2end = "auto"
                required = $true
            }
        }
    }
    return $cases
}

function Convert-EndToEndValue {
    param([object]$Value)
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    if ($text -in @("true", "1", "yes", "on")) {
        return $true
    }
    if ($text -in @("false", "0", "no", "off", "auto", "")) {
        return $false
    }
    throw "Invalid end2end value: $Value"
}

function Expected-EndToEnd {
    param([object]$Case)
    $raw = [string]$Case.end2end
    if ($raw -eq "auto") {
        return $null
    }
    return Convert-EndToEndValue $raw
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
        taskId = "yolo26-$($Case.name)"
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
                end2end = [string]$Case.end2end
                imgsz = $ImageSize
                batch = $Batch
                device = $Device
            }
        }
    }
}

function New-InferenceRequest {
    param(
        [object]$Case,
        [string]$OnnxPath,
        [string]$ImagePath,
        [string]$OutputPath
    )

    return [ordered]@{
        taskId = "yolo26-infer-$($Case.name)"
        checkpointPath = $OnnxPath
        imagePath = $ImagePath
        outputPath = $OutputPath
        confidenceThreshold = 0.01
        iouThreshold = 0.45
        maxDetections = 300
    }
}

function Get-TrainingReportSummary {
    param(
        [string]$ReportPath,
        [object]$Case
    )

    Assert-PathExists $ReportPath "training report"
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $ReportPath | ConvertFrom-Json
    if (($report.PSObject.Properties.Name -contains "ok") -and -not $report.ok) {
        throw "Training report reports ok=false: $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "backend") -or [string]$report.backend -ne [string]$Case.backend) {
        throw "Report backend mismatch for $($Case.model): $ReportPath"
    }
    if (!($report.PSObject.Properties.Name -contains "model") -or [string]$report.model -ne [string]$Case.model) {
        throw "Report model mismatch for $($Case.model): $ReportPath"
    }
    if ([string]$Case.sourceType -eq "pt" -and -not ([string]$report.model).EndsWith(".pt")) {
        throw "Report model does not preserve .pt source path for $($Case.model): $ReportPath"
    }
    foreach ($property in @("metrics", "checkpointPath", "onnxPath", "ultralyticsExportArgs")) {
        if (!($report.PSObject.Properties.Name -contains $property)) {
            throw "Report is missing $property`: $ReportPath"
        }
    }
    if (!($report.ultralyticsExportArgs.PSObject.Properties.Name -contains "end2end")) {
        throw "Report is missing ultralyticsExportArgs.end2end: $ReportPath"
    }
    $actualEndToEnd = Convert-EndToEndValue $report.ultralyticsExportArgs.end2end
    $expectedEndToEnd = Expected-EndToEnd $Case
    if ($null -ne $expectedEndToEnd -and $actualEndToEnd -ne $expectedEndToEnd) {
        throw "Report end2end mismatch for $($Case.model): expected=$expectedEndToEnd actual=$actualEndToEnd"
    }

    $checkpointPath = [string]$report.checkpointPath
    $onnxPath = [string]$report.onnxPath
    Assert-PathExists $checkpointPath "checkpointPath"
    Assert-PathExists $onnxPath "onnxPath"

    return [pscustomobject][ordered]@{
        reportPath = [System.IO.Path]::GetFullPath($ReportPath)
        checkpointPath = [System.IO.Path]::GetFullPath($checkpointPath)
        onnxPath = [System.IO.Path]::GetFullPath($onnxPath)
        onnxSidecarPath = if ($report.PSObject.Properties.Name -contains "onnxSidecarPath") { [string]$report.onnxSidecarPath } else { "" }
        ultralyticsExportArgs = $report.ultralyticsExportArgs
        modelFamily = if ($report.PSObject.Properties.Name -contains "modelFamily") { [string]$report.modelFamily } else { "" }
        modelSeries = if ($report.PSObject.Properties.Name -contains "modelSeries") { [string]$report.modelSeries } else { "" }
        outputShapes = if ($report.PSObject.Properties.Name -contains "outputShapes") { $report.outputShapes } else { $null }
        metrics = $report.metrics
    }
}

function Get-InferenceSummary {
    param([string]$OutputPath)

    $predictionsPath = Join-Path $OutputPath "inference_predictions.json"
    $overlayPath = Join-Path $OutputPath "inference_overlay.png"
    Assert-PathExists $predictionsPath "inference predictions"
    Assert-PathExists $overlayPath "inference overlay"
    $predictions = Get-Content -Raw -Encoding UTF8 -LiteralPath $predictionsPath | ConvertFrom-Json
    return [pscustomobject][ordered]@{
        outputPath = [System.IO.Path]::GetFullPath($OutputPath)
        predictionsPath = [System.IO.Path]::GetFullPath($predictionsPath)
        overlayPath = [System.IO.Path]::GetFullPath($overlayPath)
        taskType = [string]$predictions.taskType
        predictionCount = @($predictions.predictions).Count
        runtime = [string]$predictions.runtime
    }
}

function Get-DeploymentStatusFromCapture {
    param([object]$Capture)
    if ($Capture.exitCode -eq 0 -and $Capture.json -and $Capture.json.ok) {
        return "passed"
    }
    $status = if ($Capture.json -and $Capture.json.PSObject.Properties.Name -contains "status") { [string]$Capture.json.status } else { "" }
    if ($status -in @("blocked", "hardware-blocked")) {
        return "blocked"
    }
    if ($Capture.text -match "hardware|unavailable|not found|missing|requires|blocked|sdk_missing|sample_missing") {
        return "blocked"
    }
    return "failed"
}

function Invoke-Yolo26Deployments {
    param(
        [object]$Case,
        [string]$Python,
        [string]$ResolvedWorkerExe,
        [string]$CheckpointPath,
        [string]$OnnxPath,
        [string]$SampleImage,
        [object]$ReportSummary,
        [string]$CaseOutputPath
    )

    $deployments = [ordered]@{}
    $targets = @($DeploymentTargets | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
    if ($targets -contains "onnx") {
        $deployments.onnx = [ordered]@{
            status = "passed"
            source = "aitrain_cpp_onnx_inference"
            onnxPath = $OnnxPath
        }
    }

    if ($targets -contains "tensorrt") {
        $trtOutput = Join-Path $CaseOutputPath "deployment\tensorrt"
        New-Item -ItemType Directory -Force $trtOutput | Out-Null
        $trt = Invoke-ProcessCapture `
            -FilePath $ResolvedWorkerExe `
            -Arguments @("--tensorrt-smoke", $OnnxPath) `
            -LogPath (Join-Path $trtOutput "tensorrt_smoke.log") `
            -AllowFailure
        $deployments.tensorrt = [ordered]@{
            status = Get-DeploymentStatusFromCapture -Capture $trt
            exitCode = $trt.exitCode
            result = $trt.json
            logPath = $trt.logPath
        }
        Write-JsonFile -Path (Join-Path $trtOutput "tensorrt_smoke_summary.json") -Value $deployments.tensorrt
    }

    return $deployments
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
    $sampleImage = if ($task -eq "segmentation") {
        Join-Path $datasetPath "images\val\b.png"
    } else {
        Join-Path $datasetPath "images\val\b.png"
    }
    Assert-PathExists $sampleImage "inference sample image"

    $outputPath = Join-Path $RunsRoot ([string]$Case.name)
    $requestPath = Join-Path $RunsRoot "$($Case.name)-request.json"
    $eventsPath = Join-Path $RunsRoot "$($Case.name)-worker-events.jsonl"
    $request = New-YoloRequest -Case $Case -Python $Python -DatasetPath $datasetPath -OutputPath $outputPath
    Write-JsonFile -Path $requestPath -Value $request

    Write-Step "worker train/export $model"
    $completedEvent = Invoke-WorkerCommand -ResolvedWorkerExe $ResolvedWorkerExe -CommandType "startTrain" -Request $request -EventsPath $eventsPath

    $reportPath = Join-Path $outputPath "ultralytics_training_report.json"
    $reportSummary = Get-TrainingReportSummary -ReportPath $reportPath -Case $Case

    $inferenceOutput = Join-Path $outputPath "aitrain_inference"
    $inferenceEventsPath = Join-Path $RunsRoot "$($Case.name)-inference-events.jsonl"
    $inferenceRequestPath = Join-Path $RunsRoot "$($Case.name)-inference-request.json"
    $inferenceRequest = New-InferenceRequest -Case $Case -OnnxPath $reportSummary.onnxPath -ImagePath $sampleImage -OutputPath $inferenceOutput
    Write-JsonFile -Path $inferenceRequestPath -Value $inferenceRequest

    Write-Step "worker inference $model"
    $inferenceCompletedEvent = Invoke-WorkerCommand -ResolvedWorkerExe $ResolvedWorkerExe -CommandType "infer" -Request $inferenceRequest -EventsPath $inferenceEventsPath
    $inferenceSummary = Get-InferenceSummary -OutputPath $inferenceOutput
    $deploymentSummary = Invoke-Yolo26Deployments `
        -Case $Case `
        -Python $Python `
        -ResolvedWorkerExe $ResolvedWorkerExe `
        -CheckpointPath $reportSummary.checkpointPath `
        -OnnxPath $reportSummary.onnxPath `
        -SampleImage $sampleImage `
        -ReportSummary $reportSummary `
        -CaseOutputPath $outputPath
    $finished = [DateTime]::UtcNow

    Write-Host ("  [ok] {0}: checkpoint, ONNX, report, predictions, and overlay verified" -f $model)
    return [ordered]@{
        name = [string]$Case.name
        task = $task
        backend = $backend
        model = $model
        sourceType = [string]$Case.sourceType
        requestedEndToEnd = [string]$Case.end2end
        expectedEndToEnd = (Expected-EndToEnd $Case)
        required = [bool]$Case.required
        status = "passed"
        resolvedUltralyticsTask = $resolvedTask
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        artifacts = $reportSummary
        inference = $inferenceSummary
        deployments = $deploymentSummary
        workerEventsPath = [System.IO.Path]::GetFullPath($eventsPath)
        inferenceEventsPath = [System.IO.Path]::GetFullPath($inferenceEventsPath)
        completedEvent = $completedEvent
        inferenceCompletedEvent = $inferenceCompletedEvent
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
        Write-Host "  [warn] CTest build directory not found; skipping C++ regression check." -ForegroundColor Yellow
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

if ($Focused -and $Full) {
    throw "Use either -Focused or -Full, not both."
}
$targets = @($DeploymentTargets | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
foreach ($target in $targets) {
    if ($target -eq "ncnn") {
        throw "YOLO26 NCNN deployment is not supported by AITrain; use ONNX or TensorRT for YOLO26."
    }
    if ($target -notin @("onnx", "tensorrt")) {
        throw "Unsupported deployment target: $target"
    }
}

$python = Resolve-PythonExe
$work = Resolve-RepoPath $WorkDir
$generated = Join-Path $work "generated"
$runs = Join-Path $work "runs"
$summaryPath = Join-Path $work "yolo26_model_matrix_summary.json"
$environmentPath = Join-Path $work "yolo26_environment_self_check.json"
New-Item -ItemType Directory -Force $work | Out-Null

$requireCuda = Test-DeviceRequiresCuda -DeviceValue $Device
$environmentProbe = Invoke-Yolo26EnvironmentProbe -Python $python -RequireCuda $requireCuda
Write-JsonFile -Path $environmentPath -Value $environmentProbe
$ultralyticsVersion = if ($environmentProbe.PSObject.Properties.Name -contains "ultralyticsVersion") { [string]$environmentProbe.ultralyticsVersion } else { "" }
if ($ultralyticsVersion) {
    Write-Host ("  [ok] Ultralytics version={0}" -f $ultralyticsVersion)
}

$cases = New-Yolo26Cases
if ($ProbeOnly -or -not [bool]$environmentProbe.ok) {
    $finishedAt = [DateTime]::UtcNow
    $status = if ([bool]$environmentProbe.ok) { "probe_passed" } else { "blocked" }
    $summary = [ordered]@{
        ok = [bool]$environmentProbe.ok
        phase = "yolo26-model-matrix"
        mode = "probe"
        status = $status
        workDir = $work
        startedAt = $script:StartedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
        ultralyticsVersion = $ultralyticsVersion
        yolo26Python = $python
        environmentReport = $environmentPath
        parameters = [ordered]@{
            epochs = $Epochs
            batchSize = $Batch
            imageSize = $ImageSize
            device = $Device
            deploymentTargets = $targets
            requiredCaseCount = @($cases | Where-Object { $_.required }).Count
            caseName = $CaseName
        }
        probe = $environmentProbe
        results = @()
        note = "Probe mode validates isolated YOLO26 environment, CUDA availability when requested, cfg/models/26 presence, and nano detection/segmentation yaml/pt loading before training."
    }
    Write-JsonFile -Path $summaryPath -Value $summary
    Write-Host ("  [ok] summary={0}" -f $summaryPath)
    if (![bool]$environmentProbe.ok) {
        Write-Host "YOLO26 probe blocked. See $environmentPath" -ForegroundColor Yellow
        exit 2
    }
    Write-Host "YOLO26 probe passed" -ForegroundColor Green
    exit 0
}

$workerExeResolved = Resolve-WorkerExe

$generator = Join-Path $script:Root "examples\create-minimal-datasets.py"
Assert-PathExists $generator "minimal dataset generator"
Invoke-Checked -FilePath $python -Arguments @($generator, "--output", $generated)

if ($CaseName.Count -gt 0) {
    $requestedCaseNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $CaseName) {
        [void]$requestedCaseNames.Add($name)
    }
    $cases = @($cases | Where-Object { $requestedCaseNames.Contains([string]$_.name) })
    if ($cases.Count -eq 0) {
        throw "No YOLO26 cases matched -CaseName: $($CaseName -join ', ')"
    }
}
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
            requestedEndToEnd = [string]$case.end2end
            expectedEndToEnd = (Expected-EndToEnd $case)
            required = [bool]$case.required
            status = "failed"
            resolvedUltralyticsTask = ""
            startedAt = ""
            finishedAt = $finished.ToString("o")
            elapsedSeconds = 0
            artifacts = [ordered]@{}
            inference = [ordered]@{}
            workerEventsPath = ""
            inferenceEventsPath = ""
            completedEvent = $null
            inferenceCompletedEvent = $null
            failureCategory = "yolo26-model-matrix-failed"
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
    phase = "yolo26-model-matrix"
    mode = if ($Focused) { "focused" } else { "full" }
    status = $status
    workDir = $work
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    ultralyticsVersion = $ultralyticsVersion
    yolo26Python = $python
    environmentReport = $environmentPath
    workerExe = $workerExeResolved
    parameters = [ordered]@{
        epochs = $Epochs
        batchSize = $Batch
        imageSize = $ImageSize
        device = $Device
        deploymentTargets = $targets
        requiredCaseCount = @($cases | Where-Object { $_.required }).Count
        caseName = $CaseName
    }
    ctestStatus = $ctestStatus
    ctestFailure = $ctestFailure
    results = $results
    note = "YOLO26 is tracked as a separate compatibility phase. Probe mode validates the isolated environment before training. Full mode validates detection and instance segmentation n/s/m/l/x yaml/pt presets. Focused mode validates the four nano yaml/pt lifecycle rows. Semantic segmentation, classification, pose, OBB, tracking, and YOLOE-26 remain outside this matrix."
}
$summary | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host ("  [ok] summary={0}" -f $summaryPath)

if ($requiredFailures.Count -gt 0) {
    throw "YOLO26 matrix failed for required models: $($requiredFailures.model -join ', ')"
}
if ($ctestStatus -eq "failed") {
    throw "YOLO26 matrix CTest step failed: $ctestFailure"
}
Write-Host "YOLO26 model matrix passed" -ForegroundColor Green
