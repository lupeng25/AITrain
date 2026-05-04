param(
    [switch]$LocalBaseline,
    [switch]$Package,
    [switch]$PublicDatasets,
    [switch]$CpuTrainingSmoke,
    [switch]$TensorRT,
    [switch]$SkipBuild,
    [switch]$SkipOfficialOcr,
    [switch]$RequirePublicDatasets,
    [switch]$InstallMissingPythonPackages,
    [string]$BuildDir = "build-vscode",
    [string]$WorkDir = ".deps\acceptance-smoke",
    [string]$PythonExe = "",
    [string]$PackagedRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:AcceptanceStartedAt = [DateTime]::UtcNow
$script:AcceptanceModes = @()
$script:AcceptanceStatus = "running"
$script:AcceptanceFailure = ""
$script:AcceptanceHardwareBlocked = ""

function Write-Step {
    param([string]$Message)
    Write-Host "Acceptance: $Message" -ForegroundColor Cyan
}

function Resolve-AcceptancePath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Write-AcceptanceSummary {
    param(
        [string]$Status,
        [string]$Failure = "",
        [string]$HardwareBlocked = ""
    )
    $work = Resolve-AcceptancePath $WorkDir
    New-Item -ItemType Directory -Force $work | Out-Null
    $summary = [ordered]@{
        modes = $script:AcceptanceModes
        status = $Status
        workDir = $work
        startedAt = $script:AcceptanceStartedAt.ToString("o")
        finishedAt = ([DateTime]::UtcNow).ToString("o")
        failure = $Failure
        hardwareBlocked = $HardwareBlocked
    }
    $summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $work "acceptance_summary.json") -Encoding UTF8
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Description
    )
    if (!(Test-Path $Path)) {
        throw "Missing $Description`: $Path"
    }
    Write-Host "  [ok] $Description"
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
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments
        } else {
            & $FilePath @Arguments
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Resolve-WorkerExe {
    $candidates = @()
    if ($PackagedRoot) {
        $packaged = Resolve-AcceptancePath $PackagedRoot
        $candidates += (Join-Path $packaged "aitrain_worker.exe")
    }
    $candidates += @(
        (Join-Path $script:Root "aitrain_worker.exe"),
        (Join-Path $script:Root "$BuildDir\bin\aitrain_worker.exe"),
        (Join-Path $script:Root "$BuildDir\package-smoke\aitrain_worker.exe"),
        (Join-Path $script:Root "build\bin\aitrain_worker.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "Could not find aitrain_worker.exe. Build first or pass -PackagedRoot."
}

function Resolve-PluginDir {
    param([string]$WorkerExe)
    $workerDir = Split-Path -Parent $WorkerExe
    $candidates = @(
        (Join-Path $workerDir "plugins\models"),
        (Join-Path $script:Root "plugins\models"),
        (Join-Path $script:Root "$BuildDir\plugins\models"),
        (Join-Path $script:Root "$BuildDir\package-smoke\plugins\models")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "Could not find plugins\models directory for worker: $WorkerExe"
}

function Resolve-PythonExe {
    if ($PythonExe) {
        $resolved = Resolve-AcceptancePath $PythonExe
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

function Test-PythonModules {
    param(
        [string]$Python,
        [string[]]$Modules
    )
    $code = "import importlib.util, sys; missing=[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]; print(','.join(missing)); sys.exit(1 if missing else 0)"
    $output = & $Python -c $code @Modules
    if ($LASTEXITCODE -ne 0) {
        return ($output | Select-Object -Last 1)
    }
    return ""
}

function Ensure-PythonModules {
    param(
        [string]$Python,
        [string[]]$Modules,
        [string]$RequirementsFile,
        [string]$CapabilityName
    )
    $missing = Test-PythonModules -Python $Python -Modules $Modules
    if (!$missing) {
        Write-Host "  [ok] Python modules for $CapabilityName"
        return
    }
    if (!$InstallMissingPythonPackages) {
        throw "Missing Python modules for $CapabilityName`: $missing. Re-run with -InstallMissingPythonPackages or install $RequirementsFile."
    }
    $requirements = Resolve-AcceptancePath $RequirementsFile
    Assert-PathExists $requirements "$CapabilityName requirements"
    Invoke-Checked -FilePath $Python -Arguments @("-m", "pip", "install", "-r", $requirements)
}

function Invoke-WorkerSelfCheck {
    param([string]$WorkerExe)
    Write-Step "worker self-check"
    $output = & $WorkerExe --self-check
    if ($LASTEXITCODE -ne 0) {
        throw "Worker self-check failed with exit code $LASTEXITCODE"
    }
    $json = $output | Select-Object -Last 1 | ConvertFrom-Json
    if (-not $json.ok) {
        throw "Worker self-check reported ok=false"
    }
    Write-Host ("  [ok] worker self-check status={0}" -f $json.status)
    return $json
}

function Invoke-PluginSmoke {
    param(
        [string]$WorkerExe,
        [string]$PluginDir
    )
    Write-Step "plugin smoke"
    $output = & $WorkerExe --plugin-smoke $PluginDir
    if ($LASTEXITCODE -ne 0) {
        throw "Plugin smoke failed with exit code $LASTEXITCODE"
    }
    $json = $output | Select-Object -Last 1 | ConvertFrom-Json
    if (-not $json.ok) {
        throw "Plugin smoke reported ok=false"
    }
    if ($json.pluginCount -lt 3) {
        throw "Expected at least 3 plugins, found $($json.pluginCount)"
    }
    Write-Host ("  [ok] plugin count={0}" -f $json.pluginCount)
}

function Invoke-LocalBaseline {
    $harness = Join-Path $script:Root "tools\harness-check.ps1"
    if (!(Test-Path $harness)) {
        $worker = Resolve-WorkerExe
        $plugins = Resolve-PluginDir -WorkerExe $worker
        Invoke-WorkerSelfCheck -WorkerExe $worker | Out-Null
        Invoke-PluginSmoke -WorkerExe $worker -PluginDir $plugins
        return
    }
    Invoke-Checked -FilePath $harness
}

function Invoke-PackageAcceptance {
    $packageSmoke = Join-Path $script:Root "tools\package-smoke.ps1"
    if ((Test-Path $packageSmoke) -and (Test-Path (Join-Path $script:Root "CMakeLists.txt"))) {
        $args = @()
        if ($SkipBuild) {
            $args += "-SkipBuild"
        }
        $args += @("-BuildDir", $BuildDir)
        Invoke-Checked -FilePath $packageSmoke -Arguments $args
        return
    }

    $worker = Resolve-WorkerExe
    $packageRoot = if ($PackagedRoot) { Resolve-AcceptancePath $PackagedRoot } else { Split-Path -Parent $worker }
    Assert-PathExists (Join-Path $packageRoot "AITrainStudio.exe") "AITrain Studio executable"
    Assert-PathExists (Join-Path $packageRoot "aitrain_worker.exe") "Worker executable"
    Assert-PathExists (Join-Path $packageRoot "docs\acceptance-runbook.md") "acceptance runbook"
    Assert-PathExists (Join-Path $packageRoot "tools\acceptance-smoke.ps1") "acceptance smoke script"
    Assert-PathExists (Join-Path $packageRoot "examples\create-minimal-datasets.py") "minimal dataset generator"
    Assert-PathExists (Join-Path $packageRoot "python_trainers\requirements-yolo.txt") "YOLO requirements"
    Assert-PathExists (Join-Path $packageRoot "python_trainers\requirements-ocr.txt") "OCR requirements"
    $plugins = Resolve-PluginDir -WorkerExe $worker
    Invoke-WorkerSelfCheck -WorkerExe $worker | Out-Null
    Invoke-PluginSmoke -WorkerExe $worker -PluginDir $plugins
}

function Materialize-UltralyticsDataset {
    param(
        [string]$Python,
        [string]$YamlName,
        [string]$Destination,
        [string]$Fallback
    )
    $destinationFull = Resolve-AcceptancePath $Destination
    $resultPath = Join-Path (Split-Path -Parent $destinationFull) ("materialize-{0}.json" -f ($YamlName -replace "[^A-Za-z0-9_.-]", "_"))
    $materializeScript = Join-Path $script:Root "tools\materialize-ultralytics-dataset.py"
    Assert-PathExists $materializeScript "Ultralytics dataset materializer"
    if (Test-Path $resultPath) {
        Remove-Item -LiteralPath $resultPath -Force
    }
    $downloads = Resolve-AcceptancePath ".deps\datasets\downloads"
    $materializedRoot = Resolve-AcceptancePath ".deps\datasets\materialized"
    $output = & $Python $materializeScript --yaml $YamlName --destination $destinationFull --downloads $downloads --materialized-root $materializedRoot --report $resultPath 2>&1
    $last = if (Test-Path $resultPath) { Get-Content -Raw -Encoding UTF8 -LiteralPath $resultPath } else { $output | Select-Object -Last 1 }
    try {
        $json = $last | ConvertFrom-Json
    } catch {
        Write-Host ($output -join [Environment]::NewLine) -ForegroundColor Yellow
        if ($RequirePublicDatasets) {
            throw "Could not parse Ultralytics dataset materialization output for $YamlName."
        }
        Write-Host "  [warn] Could not parse Ultralytics dataset materialization output; using fallback." -ForegroundColor Yellow
        return (Resolve-AcceptancePath $Fallback)
    }
    $hasOk = $null -ne ($json | Get-Member -Name "ok" -MemberType NoteProperty -ErrorAction SilentlyContinue)
    if ($hasOk -and $json.ok) {
        Write-Host ("  [ok] materialized {0} from {1}" -f $YamlName, $json.downloadUrl)
        return [string]$json.path
    }
    $hasError = $null -ne ($json | Get-Member -Name "error" -MemberType NoteProperty -ErrorAction SilentlyContinue)
    $reason = if ($hasOk -and $hasError) {
        $json.error
    } elseif ($last) {
        "unexpected output: $last"
    } else {
        "no JSON materialization result was produced"
    }
    if ($RequirePublicDatasets) {
        throw ("{0} public dataset materialization failed: {1}" -f $YamlName, $reason)
    }
    $fallbackReport = [ordered]@{
        ok = $true
        fallback = $true
        yamlName = $YamlName
        reason = $reason
        path = (Resolve-AcceptancePath $Fallback)
        materializationReport = $resultPath
    }
    $fallbackReport | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $destinationFull) ("materialize-{0}-fallback.json" -f ($YamlName -replace "[^A-Za-z0-9_.-]", "_"))) -Encoding UTF8
    Write-Host ("  [warn] {0} unavailable: {1}; using generated fallback." -f $YamlName, $reason) -ForegroundColor Yellow
    return (Resolve-AcceptancePath $Fallback)
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

function Assert-TrainingReport {
    param(
        [string]$ReportPath,
        [string[]]$ArtifactProperties
    )
    Assert-PathExists $ReportPath "training report"
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $ReportPath | ConvertFrom-Json
    if ($report.PSObject.Properties.Name -contains "ok" -and -not $report.ok) {
        throw "Training report reports ok=false: $ReportPath"
    }
    foreach ($property in $ArtifactProperties) {
        if (!($report.PSObject.Properties.Name -contains $property)) {
            throw "Report is missing $property`: $ReportPath"
        }
        $artifactPath = [string]$report.$property
        if (!$artifactPath) {
            throw "Report property $property is empty: $ReportPath"
        }
        Assert-PathExists $artifactPath $property
    }
}

function Get-TrainingReportSummary {
    param([string]$ReportPath)
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $ReportPath | ConvertFrom-Json
    $summary = [ordered]@{
        reportPath = [System.IO.Path]::GetFullPath($ReportPath)
        checkpointPath = if ($report.PSObject.Properties.Name -contains "checkpointPath") { [string]$report.checkpointPath } else { "" }
        onnxPath = if ($report.PSObject.Properties.Name -contains "onnxPath") { [string]$report.onnxPath } else { "" }
        dictPath = if ($report.PSObject.Properties.Name -contains "dictPath") { [string]$report.dictPath } else { "" }
        metrics = if ($report.PSObject.Properties.Name -contains "metrics") { $report.metrics } else { [ordered]@{} }
    }
    return $summary
}

function Count-Files {
    param(
        [string]$Path,
        [string]$Filter = "*"
    )
    if (!(Test-Path $Path)) {
        return 0
    }
    return @((Get-ChildItem -LiteralPath $Path -Filter $Filter -File -ErrorAction SilentlyContinue)).Count
}

function Invoke-CtestForAcceptanceWorkDir {
    param(
        [string]$WorkRoot,
        [int]$TimeoutSeconds = 240
    )
    $ctestFile = Join-Path $script:Root "$BuildDir\CTestTestfile.cmake"
    if (Test-Path $ctestFile) {
        $previousAcceptanceSmokeRoot = $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT
        $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $WorkRoot
        try {
            try {
                Invoke-Checked -FilePath "ctest" -Arguments @("--test-dir", (Join-Path $script:Root $BuildDir), "--output-on-failure", "--timeout", "$TimeoutSeconds")
            } catch {
                Write-Host "  [warn] CTest failed immediately after smoke training; retrying once after cleanup delay." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
                Invoke-Checked -FilePath "ctest" -Arguments @("--test-dir", (Join-Path $script:Root $BuildDir), "--output-on-failure", "--timeout", "$TimeoutSeconds")
            }
        } finally {
            if ($null -eq $previousAcceptanceSmokeRoot) {
                Remove-Item Env:\AITRAIN_ACCEPTANCE_SMOKE_ROOT -ErrorAction SilentlyContinue
            } else {
                $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $previousAcceptanceSmokeRoot
            }
        }
    } else {
        Write-Host "  [warn] CTest build directory not found; skipping C++ ONNX inference regression check." -ForegroundColor Yellow
    }
}

function Invoke-CpuTrainingSmoke {
    $python = Resolve-PythonExe
    Ensure-PythonModules -Python $python -Modules @("ultralytics") -RequirementsFile "python_trainers\requirements-yolo.txt" -CapabilityName "Ultralytics YOLO"
    Ensure-PythonModules -Python $python -Modules @("paddle", "onnx", "PIL", "numpy") -RequirementsFile "python_trainers\requirements-ocr.txt" -CapabilityName "PaddlePaddle OCR Rec"

    $baseWork = Resolve-AcceptancePath $WorkDir
    $work = Join-Path $baseWork "cpu-training-smoke"
    New-Item -ItemType Directory -Force $work | Out-Null

    $startedAt = [DateTime]::UtcNow
    $generator = Join-Path $script:Root "examples\create-minimal-datasets.py"
    Assert-PathExists $generator "minimal dataset generator"
    Invoke-Checked -FilePath $python -Arguments @($generator, "--output", (Join-Path $work "generated"), "--profile", "cpu-smoke")

    $generated = Join-Path $work "generated"
    $datasetSummary = [ordered]@{
        yoloDetectTrain = Count-Files -Path (Join-Path $generated "yolo_detect\images\train") -Filter "*.png"
        yoloDetectVal = Count-Files -Path (Join-Path $generated "yolo_detect\images\val") -Filter "*.png"
        yoloSegmentTrain = Count-Files -Path (Join-Path $generated "yolo_segment\images\train") -Filter "*.png"
        yoloSegmentVal = Count-Files -Path (Join-Path $generated "yolo_segment\images\val") -Filter "*.png"
        ocrRecSamples = if (Test-Path (Join-Path $generated "paddleocr_rec\rec_gt.txt")) {
            @((Get-Content -Encoding UTF8 -LiteralPath (Join-Path $generated "paddleocr_rec\rec_gt.txt") | Where-Object { $_.Trim() })).Count
        } else {
            0
        }
    }

    $detectRequestPath = Join-Path $generated "yolo_detect_request.json"
    $segmentRequestPath = Join-Path $generated "yolo_segment_request.json"
    $ocrRequestPath = Join-Path $generated "paddleocr_rec_request.json"

    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\detection\ultralytics_trainer.py"), "--request", $detectRequestPath)
    $detectReportPath = Join-Path $generated "runs\yolo_detect\ultralytics_training_report.json"
    Assert-TrainingReport -ReportPath $detectReportPath -ArtifactProperties @("checkpointPath", "onnxPath")

    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\segmentation\ultralytics_trainer.py"), "--request", $segmentRequestPath)
    $segmentReportPath = Join-Path $generated "runs\yolo_segment\ultralytics_training_report.json"
    Assert-TrainingReport -ReportPath $segmentReportPath -ArtifactProperties @("checkpointPath", "onnxPath")

    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_trainer.py"), "--request", $ocrRequestPath)
    $ocrReportPath = Join-Path $generated "runs\paddleocr_rec\paddleocr_rec_training_report.json"
    Assert-TrainingReport -ReportPath $ocrReportPath -ArtifactProperties @("checkpointPath", "onnxPath", "dictPath")

    Invoke-CtestForAcceptanceWorkDir -WorkRoot $work -TimeoutSeconds 360

    $finishedAt = [DateTime]::UtcNow
    $summary = [ordered]@{
        ok = $true
        mode = "CpuTrainingSmoke"
        workDir = $work
        startedAt = $startedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        elapsedSeconds = [Math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)
        note = "CPU smoke validates training/export/inference wiring on generated small/medium data; it is not an accuracy benchmark."
        datasets = $datasetSummary
        parameters = [ordered]@{
            yolo = [ordered]@{ epochs = 3; batchSize = 2; imageSize = 128; device = "cpu"; workers = 0 }
            ocrRec = [ordered]@{ epochs = 8; batchSize = 8; imageWidth = 128; imageHeight = 32; maxTextLength = 10 }
        }
        reports = [ordered]@{
            detection = Get-TrainingReportSummary -ReportPath $detectReportPath
            segmentation = Get-TrainingReportSummary -ReportPath $segmentReportPath
            ocrRecognition = Get-TrainingReportSummary -ReportPath $ocrReportPath
        }
    }
    $summaryPath = Join-Path $work "cpu_training_smoke_summary.json"
    $summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host ("  [ok] CPU training smoke summary={0}" -f $summaryPath)
}

function Invoke-PublicDatasetSmoke {
    $python = Resolve-PythonExe
    Ensure-PythonModules -Python $python -Modules @("ultralytics") -RequirementsFile "python_trainers\requirements-yolo.txt" -CapabilityName "Ultralytics YOLO"
    Ensure-PythonModules -Python $python -Modules @("paddle", "onnx", "PIL", "numpy") -RequirementsFile "python_trainers\requirements-ocr.txt" -CapabilityName "PaddlePaddle OCR Rec"

    $work = Resolve-AcceptancePath $WorkDir
    New-Item -ItemType Directory -Force $work | Out-Null

    $generator = Join-Path $script:Root "examples\create-minimal-datasets.py"
    Assert-PathExists $generator "minimal dataset generator"
    Invoke-Checked -FilePath $python -Arguments @($generator, "--output", (Join-Path $work "generated"))

    $detectDataset = Materialize-UltralyticsDataset -Python $python -YamlName "coco8.yaml" -Destination (Join-Path $work "coco8") -Fallback (Join-Path $work "generated\yolo_detect")
    $segmentDataset = Materialize-UltralyticsDataset -Python $python -YamlName "coco8-seg.yaml" -Destination (Join-Path $work "coco8-seg") -Fallback (Join-Path $work "generated\yolo_segment")

    $detectRequestPath = Join-Path $work "yolo_detect_acceptance_request.json"
    $detectOutput = Join-Path $work "runs\yolo_detect"
    Write-JsonFile -Path $detectRequestPath -Value ([ordered]@{
        protocolVersion = 1
        taskId = "acceptance-yolo-detect"
        taskType = "detection"
        datasetPath = $detectDataset
        outputPath = $detectOutput
        backend = "ultralytics_yolo_detect"
        parameters = [ordered]@{
            trainingBackend = "ultralytics_yolo_detect"
            model = "yolov8n.yaml"
            epochs = 1
            batchSize = 1
            imageSize = 64
            device = "cpu"
            workers = 0
            runName = "acceptance-yolo-detect"
            compactEvents = $true
        }
    })
    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\detection\ultralytics_trainer.py"), "--request", $detectRequestPath)
    Assert-TrainingReport -ReportPath (Join-Path $detectOutput "ultralytics_training_report.json") -ArtifactProperties @("checkpointPath", "onnxPath")

    $segmentRequestPath = Join-Path $work "yolo_segment_acceptance_request.json"
    $segmentOutput = Join-Path $work "runs\yolo_segment"
    Write-JsonFile -Path $segmentRequestPath -Value ([ordered]@{
        protocolVersion = 1
        taskId = "acceptance-yolo-segment"
        taskType = "segmentation"
        datasetPath = $segmentDataset
        outputPath = $segmentOutput
        backend = "ultralytics_yolo_segment"
        parameters = [ordered]@{
            trainingBackend = "ultralytics_yolo_segment"
            model = "yolov8n-seg.yaml"
            epochs = 1
            batchSize = 1
            imageSize = 64
            device = "cpu"
            workers = 0
            runName = "acceptance-yolo-segment"
            compactEvents = $true
        }
    })
    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\segmentation\ultralytics_trainer.py"), "--request", $segmentRequestPath)
    Assert-TrainingReport -ReportPath (Join-Path $segmentOutput "ultralytics_training_report.json") -ArtifactProperties @("checkpointPath", "onnxPath")

    $ocrRequestPath = Join-Path $work "generated\paddleocr_rec_request.json"
    Invoke-Checked -FilePath $python -Arguments @((Join-Path $script:Root "python_trainers\ocr_rec\paddleocr_trainer.py"), "--request", $ocrRequestPath)
    Assert-TrainingReport -ReportPath (Join-Path $work "generated\runs\paddleocr_rec\paddleocr_rec_training_report.json") -ArtifactProperties @("checkpointPath", "onnxPath", "dictPath")

    Invoke-CtestForAcceptanceWorkDir -WorkRoot $work -TimeoutSeconds 240

    if (!$SkipOfficialOcr) {
        $phase16 = Join-Path $script:Root "tools\phase16-ocr-official-smoke.ps1"
        if (Test-Path $phase16) {
            Invoke-Checked -FilePath $phase16
        } else {
            Write-Host "  [warn] phase16-ocr-official-smoke.ps1 not found; skipping official PaddleOCR train/export smoke." -ForegroundColor Yellow
        }
    }
}

function Get-GpuComputeCapability {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (!$nvidiaSmi) {
        return $null
    }
    $output = & $nvidiaSmi.Source --query-gpu=name,compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or !$output) {
        return $null
    }
    $best = $null
    foreach ($line in $output) {
        $parts = $line -split ","
        if ($parts.Count -lt 2) {
            continue
        }
        $capText = $parts[$parts.Count - 1].Trim()
        $cap = 0.0
        if ([double]::TryParse($capText, [ref]$cap)) {
            if ($null -eq $best -or $cap -gt $best) {
                $best = $cap
            }
        }
    }
    return $best
}

function Invoke-TensorRtAcceptance {
    $worker = Resolve-WorkerExe
    $selfCheck = Invoke-WorkerSelfCheck -WorkerExe $worker
    if (-not $selfCheck.tensorRtBackend.inferenceAvailable) {
        throw "TensorRT backend is unavailable: $($selfCheck.tensorRtBackend.message)"
    }

    $computeCapability = Get-GpuComputeCapability
    if ($null -ne $computeCapability -and $computeCapability -lt 7.5) {
        throw "hardware-blocked: detected GPU compute capability $computeCapability. TensorRT 10 acceptance requires RTX / SM 75+."
    }
    if ($null -eq $computeCapability) {
        Write-Host "  [warn] Could not query GPU compute capability; running TensorRT smoke and relying on Worker diagnostics." -ForegroundColor Yellow
    } else {
        Write-Host ("  [ok] GPU compute capability={0}" -f $computeCapability)
    }

    $tensorRtWork = Join-Path (Resolve-AcceptancePath $WorkDir) "tensorrt"
    New-Item -ItemType Directory -Force $tensorRtWork | Out-Null
    Write-Step "TensorRT worker smoke"
    $output = & $worker --tensorrt-smoke $tensorRtWork 2>&1
    if ($LASTEXITCODE -ne 0) {
        $text = $output -join [Environment]::NewLine
        if ($text -match "SM 61|not supported|unsupported") {
            throw "hardware-blocked: TensorRT smoke failed on unsupported GPU. Output: $text"
        }
        throw "TensorRT smoke failed with exit code $LASTEXITCODE. Output: $text"
    }
    $json = $output | Select-Object -Last 1 | ConvertFrom-Json
    if (-not $json.ok) {
        throw "TensorRT smoke reported ok=false: $($output -join [Environment]::NewLine)"
    }
    Write-Host ("  [ok] TensorRT engine={0}" -f $json.enginePath)
}

try {
    if (-not ($LocalBaseline -or $Package -or $PublicDatasets -or $CpuTrainingSmoke -or $TensorRT)) {
        throw "Select at least one mode: -LocalBaseline, -Package, -PublicDatasets, -CpuTrainingSmoke, or -TensorRT."
    }

    if ($LocalBaseline) {
        $script:AcceptanceModes += "LocalBaseline"
        Invoke-LocalBaseline
    }
    if ($Package) {
        $script:AcceptanceModes += "Package"
        Invoke-PackageAcceptance
    }
    if ($PublicDatasets) {
        $script:AcceptanceModes += "PublicDatasets"
        Invoke-PublicDatasetSmoke
    }
    if ($CpuTrainingSmoke) {
        $script:AcceptanceModes += "CpuTrainingSmoke"
        Invoke-CpuTrainingSmoke
    }
    if ($TensorRT) {
        $script:AcceptanceModes += "TensorRT"
        Invoke-TensorRtAcceptance
    }

    Write-AcceptanceSummary -Status "passed"
    Write-Host "Acceptance smoke completed." -ForegroundColor Green
    exit 0
} catch {
    $message = $_.Exception.Message
    $hardwareBlocked = if ($message -like "hardware-blocked:*") { $message } else { "" }
    Write-AcceptanceSummary -Status "failed" -Failure $message -HardwareBlocked $hardwareBlocked
    Write-Host ("Acceptance smoke failed: {0}" -f $message) -ForegroundColor Red
    exit 1
}
