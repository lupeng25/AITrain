param(
    [string]$WorkDir = ".deps\smp-realtest\gpu-4090d",
    [string]$Python = ".deps\envs\smp-gpu\Scripts\python.exe",
    [string]$BasePython = "python",
    [string]$WorkerExe = "build-vscode\bin\aitrain_worker.exe",
    [int]$MainEpochs = 20,
    [int]$MatrixEpochs = 1,
    [int]$ImageSize = 256,
    [int]$BatchSize = 8,
    [switch]$SkipEnvironmentInstall,
    [switch]$SkipMatrix,
    [switch]$SkipProductRuntime
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$script:cudaTorchProvider = $null

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Write-Json {
    param([string]$Path, [object]$Value)
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $Value | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$LogPath
    )
    $dir = Split-Path -Parent $LogPath
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& $File @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $lines = @($output | ForEach-Object { [string]$_ })
    $text = $lines -join [Environment]::NewLine
    Set-Content -LiteralPath $LogPath -Value $text -Encoding UTF8
    foreach ($line in $lines) {
        Write-Host $line
    }
    return @{
        exitCode = $exitCode
        logPath = $LogPath
        text = $text
    }
}

function Invoke-PipInstall {
    param(
        [string]$PythonExe,
        [string[]]$PipArguments,
        [string]$Description,
        [int]$Attempts = 3
    )
    for ($attempt = 1; $attempt -le $Attempts; ++$attempt) {
        Write-Host ("SMP GPU realtest: {0} attempt {1}/{2}" -f $Description, $attempt, $Attempts) -ForegroundColor Cyan
        & $PythonExe -m pip @PipArguments 2>&1 | Out-Host
        if ($LASTEXITCODE -eq 0) {
            return
        }
        if ($attempt -lt $Attempts) {
            Start-Sleep -Seconds ([Math]::Min(30, 5 * $attempt))
        }
    }
    throw "$Description failed after $Attempts attempts."
}

function Read-LastJsonLine {
    param([string]$Text)
    $lines = @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    for ($index = $lines.Count - 1; $index -ge 0; --$index) {
        $line = $lines[$index].Trim()
        if ($line.StartsWith("{") -and $line.EndsWith("}")) {
            try {
                return ($line | ConvertFrom-Json)
            } catch {
            }
        }
    }
    return $null
}

function Probe-SmpEnvironment {
    param([string]$PythonExe)
    $probe = @"
import importlib.util, json, sys
missing = [m for m in ['segmentation_models_pytorch','torch','torchvision','timm','onnx','onnxruntime','PIL','numpy','yaml'] if importlib.util.find_spec(m) is None]
info = {'python': sys.executable, 'missing': missing, 'torch': None, 'cudaAvailable': False, 'cudaVersion': None, 'deviceCount': 0, 'deviceName': ''}
try:
    import torch
    info['torch'] = getattr(torch, '__version__', None)
    info['cudaAvailable'] = bool(torch.cuda.is_available())
    info['cudaVersion'] = getattr(torch.version, 'cuda', None)
    info['deviceCount'] = int(torch.cuda.device_count())
    if info['cudaAvailable']:
        info['deviceName'] = torch.cuda.get_device_name(0)
except Exception as exc:
    info['torchError'] = str(exc)
print(json.dumps(info, ensure_ascii=False))
"@
    $output = & $PythonExe -c $probe
    if ($LASTEXITCODE -ne 0) {
        throw "SMP environment probe failed for $PythonExe"
    }
    return ($output | Select-Object -Last 1 | ConvertFrom-Json)
}

function Find-CudaTorchProvider {
    param([string]$CurrentPythonExe)
    $candidates = @(
        (Resolve-RepoPath ".deps\envs\yolo-cuda\Scripts\python.exe"),
        (Resolve-RepoPath ".deps\envs\yolo26\Scripts\python.exe"),
        "python"
    )
    $probe = @"
import json, site, sys
info = {'python': sys.executable, 'torch': None, 'cudaAvailable': False, 'cudaVersion': None, 'deviceName': '', 'sitePackages': []}
try:
    info['sitePackages'] = site.getsitepackages()
except Exception:
    pass
try:
    import torch
    info['torch'] = getattr(torch, '__version__', None)
    info['cudaAvailable'] = bool(torch.cuda.is_available())
    info['cudaVersion'] = getattr(torch.version, 'cuda', None)
    if info['cudaAvailable']:
        info['deviceName'] = torch.cuda.get_device_name(0)
except Exception as exc:
    info['error'] = str(exc)
print(json.dumps(info, ensure_ascii=False))
"@
    foreach ($candidate in $candidates) {
        try {
            if ($candidate -ne "python" -and -not (Test-Path -LiteralPath $candidate)) {
                continue
            }
            $resolvedCandidate = if ($candidate -eq "python") { "python" } else { [System.IO.Path]::GetFullPath($candidate) }
            if ($resolvedCandidate -ne "python" -and [System.IO.Path]::GetFullPath($CurrentPythonExe) -eq $resolvedCandidate) {
                continue
            }
            $output = & $resolvedCandidate -c $probe
            if ($LASTEXITCODE -ne 0) {
                continue
            }
            $info = $output | Select-Object -Last 1 | ConvertFrom-Json
            if (-not $info.cudaAvailable) {
                continue
            }
            $sitePackages = @($info.sitePackages | Where-Object {
                -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath (Join-Path $_ "torch"))
            })
            if ($sitePackages.Count -eq 0) {
                continue
            }
            return @{
                python = $info.python
                torch = $info.torch
                cudaVersion = $info.cudaVersion
                deviceName = $info.deviceName
                sitePackages = $sitePackages[0]
            }
        } catch {
        }
    }
    return $null
}

function Enable-CudaTorchProvider {
    param(
        [string]$PythonExe,
        [object]$Provider
    )
    $sitePath = (& $PythonExe -c "import site; print(site.getsitepackages()[0])") | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($sitePath)) {
        throw "Cannot locate SMP GPU venv site-packages for CUDA Torch provider bridge."
    }
    New-Item -ItemType Directory -Force -Path $sitePath | Out-Null
    $pthPath = Join-Path $sitePath "aitrain_cuda_torch_provider.pth"
    Set-Content -LiteralPath $pthPath -Value $Provider.sitePackages -Encoding ASCII
    $script:cudaTorchProvider = @{
        python = $Provider.python
        torch = $Provider.torch
        cudaVersion = $Provider.cudaVersion
        deviceName = $Provider.deviceName
        sitePackages = $Provider.sitePackages
        pthPath = $pthPath
    }
}

function Ensure-SmpEnvironment {
    param([string]$PythonExe)
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $venvScripts = Split-Path -Parent $PythonExe
        $venvRoot = Split-Path -Parent $venvScripts
        Write-Host "SMP GPU realtest: create venv $venvRoot" -ForegroundColor Cyan
        & $BasePython -m venv $venvRoot 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create SMP GPU venv: $venvRoot"
        }
    }

    $info = Probe-SmpEnvironment $PythonExe
    $missing = @($info.missing)
    $needsTorch = $missing -contains "torch" -or -not $info.cudaAvailable
    $needsSmp = $missing.Count -gt 0
    if (($needsTorch -or $needsSmp) -and $SkipEnvironmentInstall) {
        throw "SMP GPU environment is incomplete and -SkipEnvironmentInstall was set. Missing: $($missing -join ', '); cudaAvailable=$($info.cudaAvailable)"
    }
    if ($needsTorch -or $needsSmp) {
        Write-Host "SMP GPU realtest: install/repair Python dependencies" -ForegroundColor Cyan
        Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "--upgrade", "pip") "pip upgrade" 2
        if ($needsTorch) {
            try {
                Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "--index-url", "https://download.pytorch.org/whl/cu126", "torch", "torchvision") "CUDA PyTorch install" 3
            } catch {
                Write-Host "SMP GPU realtest: CUDA PyTorch download failed; probing local CUDA Torch providers" -ForegroundColor Yellow
                $provider = Find-CudaTorchProvider $PythonExe
                if ($null -eq $provider) {
                    throw
                }
                Enable-CudaTorchProvider $PythonExe $provider
                Write-Host ("SMP GPU realtest: using CUDA Torch provider {0} torch={1}" -f $provider.python, $provider.torch) -ForegroundColor Yellow
            }
        }
        Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "segmentation-models-pytorch>=0.5,<0.6", "timm", "numpy", "Pillow", "PyYAML", "onnx", "onnxruntime") "SMP dependency install" 3
        $info = Probe-SmpEnvironment $PythonExe
    }
    $missing = @($info.missing)
    if ($missing.Count -gt 0) {
        throw "SMP GPU environment still misses modules: $($missing -join ', ')"
    }
    if (-not $info.cudaAvailable) {
        throw "SMP GPU realtest requires CUDA PyTorch. CPU fallback is not accepted for this 4090D gate."
    }
    return $info
}

function New-SmpRequest {
    param(
        [string]$DatasetPath,
        [string]$OutputPath,
        [string]$RequestPath,
        [string]$TaskId,
        [string]$Preset,
        [int]$Epochs,
        [int]$Batch
    )
    $request = @{
        protocolVersion = 1
        taskId = $TaskId
        taskType = "semantic_segmentation"
        datasetPath = $DatasetPath
        outputPath = $OutputPath
        backend = "smp_semantic_segmentation"
        parameters = @{
            trainingBackend = "smp_semantic_segmentation"
            datasetFormat = "semantic_segmentation_mask"
            model = $Preset
            modelPreset = $Preset
            epochs = $Epochs
            batchSize = $Batch
            imageSize = $ImageSize
            device = "0"
            workers = 0
            learningRate = 0.001
            optimizer = "adamw"
            loss = "dice_ce"
            encoderWeights = "none"
            ignoreIndex = 255
            seed = 42
        }
    }
    Write-Json $RequestPath $request
}

function Test-SmpArtifacts {
    param([string]$RunDir)
    $required = @(
        (Join-Path $RunDir "best.pt"),
        (Join-Path $RunDir "best.onnx"),
        (Join-Path $RunDir "smp_training_report.json"),
        (Join-Path $RunDir "semantic_segmentation_sidecar.json")
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_) })
    return @{
        ok = ($missing.Count -eq 0)
        missing = $missing
        checkpointPath = (Join-Path $RunDir "best.pt")
        onnxPath = (Join-Path $RunDir "best.onnx")
        trainingReportPath = (Join-Path $RunDir "smp_training_report.json")
        sidecarPath = (Join-Path $RunDir "semantic_segmentation_sidecar.json")
    }
}

function Invoke-SmpTrainingRow {
    param(
        [string]$Preset,
        [int]$Epochs,
        [int]$Batch,
        [string]$DatasetPath,
        [string]$OutputRoot,
        [string]$RequestRoot,
        [string]$LogRoot,
        [string]$TaskPrefix
    )
    $safePreset = $Preset -replace '[^A-Za-z0-9_.-]', '_'
    $runDir = Join-Path $OutputRoot $safePreset
    $requestPath = Join-Path $RequestRoot "$safePreset.json"
    $logPath = Join-Path $LogRoot "$safePreset.log"
    New-SmpRequest `
        -DatasetPath $DatasetPath `
        -OutputPath $runDir `
        -RequestPath $requestPath `
        -TaskId "$TaskPrefix-$safePreset" `
        -Preset $Preset `
        -Epochs $Epochs `
        -Batch $Batch
    Write-Host "SMP GPU realtest: train $Preset epochs=$Epochs batch=$Batch imageSize=$ImageSize" -ForegroundColor Cyan
    $result = Invoke-Logged $pythonExe @((Join-Path $root "python_trainers\semantic_segmentation\smp_trainer.py"), "--request", $requestPath) $logPath
    $usedBatch = $Batch
    if ($result.exitCode -ne 0 -and $Batch -gt 4 -and $result.text -match "out of memory|CUDA out of memory") {
        $usedBatch = 4
        $runDir = Join-Path $OutputRoot "$safePreset-batch4"
        $requestPath = Join-Path $RequestRoot "$safePreset-batch4.json"
        $logPath = Join-Path $LogRoot "$safePreset-batch4.log"
        New-SmpRequest `
            -DatasetPath $DatasetPath `
            -OutputPath $runDir `
            -RequestPath $requestPath `
            -TaskId "$TaskPrefix-$safePreset-batch4" `
            -Preset $Preset `
            -Epochs $Epochs `
            -Batch $usedBatch
        Write-Host "SMP GPU realtest: retry $Preset with batch=4 after CUDA OOM" -ForegroundColor Yellow
        $result = Invoke-Logged $pythonExe @((Join-Path $root "python_trainers\semantic_segmentation\smp_trainer.py"), "--request", $requestPath) $logPath
    }
    $artifacts = Test-SmpArtifacts $runDir
    $status = if ($result.exitCode -eq 0 -and $artifacts.ok) {
        "passed"
    } elseif ($result.text -match "no available encoder|Unsupported SMP preset|architecture .* is unavailable") {
        "blocked"
    } else {
        "failed"
    }
    return @{
        preset = $Preset
        epochs = $Epochs
        batchSize = $usedBatch
        imageSize = $ImageSize
        device = "0"
        status = $status
        ok = ($status -eq "passed")
        exitCode = $result.exitCode
        runDir = $runDir
        requestPath = $requestPath
        logPath = $logPath
        artifacts = $artifacts
    }
}

$pythonExe = Resolve-RepoPath $Python
$workerPath = Resolve-RepoPath $WorkerExe
$workFull = Resolve-RepoPath $WorkDir
$summaryPath = Join-Path $workFull "smp_4090d_gpu_realtest_summary.json"
New-Item -ItemType Directory -Force -Path $workFull | Out-Null

$nvidiaSmi = ""
try {
    $nvidiaSmi = (& nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader) -join "`n"
} catch {
    throw "nvidia-smi is required for the 4090D SMP GPU realtest."
}

$envInfo = Ensure-SmpEnvironment $pythonExe
Write-Host ("SMP GPU realtest: CUDA ready torch={0} device={1}" -f $envInfo.torch, $envInfo.deviceName) -ForegroundColor Green

Write-Host "SMP GPU realtest: generate dataset" -ForegroundColor Cyan
$datasetLog = Join-Path $workFull "logs\generate-dataset.log"
$datasetResult = Invoke-Logged $pythonExe @((Join-Path $root "examples\create-minimal-datasets.py"), "--output", $workFull, "--profile", "cpu-smoke") $datasetLog
if ($datasetResult.exitCode -ne 0) {
    throw "SMP GPU realtest dataset generation failed. See $datasetLog"
}
$datasetPath = Join-Path $workFull "semantic_mask"
$sampleImagePath = Join-Path $datasetPath "images\val\val_00.png"

$requestsRoot = Join-Path $workFull "requests"
$logsRoot = Join-Path $workFull "logs"
$runsRoot = Join-Path $workFull "runs"
$mainRow = Invoke-SmpTrainingRow `
    -Preset "smp_unet_resnet34" `
    -Epochs $MainEpochs `
    -Batch $BatchSize `
    -DatasetPath $datasetPath `
    -OutputRoot (Join-Path $runsRoot "main") `
    -RequestRoot $requestsRoot `
    -LogRoot $logsRoot `
    -TaskPrefix "smp-4090d-main"
if ($mainRow.status -ne "passed") {
    Write-Json $summaryPath @{
        ok = $false
        status = "failed"
        stage = "main-training"
        nvidiaSmi = $nvidiaSmi
        environment = $envInfo
        main = $mainRow
    }
    throw "SMP 4090D main training failed. See $($mainRow.logPath)"
}

Write-Host "SMP GPU realtest: evaluate main ONNX" -ForegroundColor Cyan
$mainOnnx = $mainRow.artifacts.onnxPath
$evaluationPath = Join-Path $workFull "evaluation\main"
$evaluationRequestPath = Join-Path $requestsRoot "main_evaluation.json"
$evaluationLogPath = Join-Path $logsRoot "main_evaluation.log"
Write-Json $evaluationRequestPath @{
    modelPath = $mainOnnx
    datasetPath = $datasetPath
    outputPath = $evaluationPath
    options = @{
        split = "val"
        maxOverlays = 8
        ignoreIndex = 255
        lowQualityThreshold = 0.5
    }
}
$evaluationResult = Invoke-Logged $pythonExe @((Join-Path $root "python_trainers\semantic_segmentation\smp_evaluator.py"), "--request", $evaluationRequestPath) $evaluationLogPath
if ($evaluationResult.exitCode -ne 0) {
    throw "SMP 4090D evaluation failed. See $evaluationLogPath"
}
$evaluationReportPath = Join-Path $evaluationPath "evaluation_report.json"
if (-not (Test-Path -LiteralPath $evaluationReportPath)) {
    throw "SMP 4090D evaluation did not write evaluation_report.json"
}

$productRuntime = @{
    skipped = $SkipProductRuntime.IsPresent
}
if (-not $SkipProductRuntime) {
    if (-not (Test-Path -LiteralPath $workerPath)) {
        throw "Worker executable not found for product runtime smoke: $workerPath. Build first with tools\harness-check.ps1 or cmake --build."
    }
    Write-Host "SMP GPU realtest: C++ semantic ONNX runtime smoke" -ForegroundColor Cyan
    $productLogPath = Join-Path $logsRoot "semantic_onnx_smoke.log"
    $productOutputPath = Join-Path $workFull "product-runtime"
    $productResult = Invoke-Logged $workerPath @("--semantic-onnx-smoke", $mainOnnx, "--image", $sampleImagePath, "--output", $productOutputPath) $productLogPath
    $productJson = Read-LastJsonLine $productResult.text
    $productRuntime = @{
        skipped = $false
        exitCode = $productResult.exitCode
        logPath = $productLogPath
        result = $productJson
        ok = ($productResult.exitCode -eq 0 -and $null -ne $productJson -and $productJson.ok)
    }
    if (-not $productRuntime.ok) {
        throw "SMP C++ semantic ONNX runtime smoke failed. See $productLogPath"
    }

    Write-Host "SMP GPU realtest: record ONNX Runtime-only deployment scope" -ForegroundColor Cyan
    $productRuntime.deploymentScope = @{
        ok = $true
        status = "not_required"
        scope = "onnx_runtime_only"
        onnxRuntime = "required"
        ncnn = "not_required"
        tensorrt = "not_required"
        note = "SMP semantic segmentation does not require NCNN/TensorRT export. This GPU lane validates ONNX Runtime inference, overlay, benchmark, and deployment only."
    }
}

$matrixRows = @()
if (-not $SkipMatrix) {
    $presets = @(
        "smp_unetplusplus_resnet34",
        "smp_fpn_resnet34",
        "smp_deeplabv3plus_resnet50",
        "smp_segformer_mit_b0"
    )
    foreach ($preset in $presets) {
        $matrixRows += Invoke-SmpTrainingRow `
            -Preset $preset `
            -Epochs $MatrixEpochs `
            -Batch $BatchSize `
            -DatasetPath $datasetPath `
            -OutputRoot (Join-Path $runsRoot "matrix") `
            -RequestRoot $requestsRoot `
            -LogRoot $logsRoot `
            -TaskPrefix "smp-4090d-matrix"
    }
}

$failedRows = @($matrixRows | Where-Object { $_.status -eq "failed" })
$blockedRows = @($matrixRows | Where-Object { $_.status -eq "blocked" })
$ok = $mainRow.ok -and ($evaluationResult.exitCode -eq 0) -and ($SkipProductRuntime -or $productRuntime.ok) -and ($failedRows.Count -eq 0)
$status = if ($ok -and $blockedRows.Count -eq 0) {
    "passed"
} elseif ($ok) {
    "passed_with_blocked_matrix_rows"
} else {
    "failed"
}

$summary = @{
    ok = $ok
    status = $status
    workDir = $workFull
    python = $pythonExe
    worker = $workerPath
    nvidiaSmi = $nvidiaSmi
    environment = $envInfo
    cudaTorchProvider = $script:cudaTorchProvider
    datasetPath = $datasetPath
    sampleImagePath = $sampleImagePath
    main = $mainRow
    evaluation = @{
        ok = ($evaluationResult.exitCode -eq 0)
        reportPath = $evaluationReportPath
        logPath = $evaluationLogPath
    }
    productRuntime = $productRuntime
    matrix = $matrixRows
    failedMatrixRows = $failedRows
    blockedMatrixRows = $blockedRows
    note = "Synthetic data validates SMP engineering lifecycle on RTX 4090D; it is not industrial accuracy evidence."
}
Write-Json $summaryPath $summary
if (-not $ok) {
    throw "SMP 4090D GPU realtest failed: $summaryPath"
}
Write-Host "SMP 4090D GPU realtest passed: $summaryPath" -ForegroundColor Green
