param(
    [string]$WorkDir = ".deps\smp-quality\oxford-pets",
    [string]$Python = ".deps\envs\smp-gpu\Scripts\python.exe",
    [string]$BasePython = "python",
    [string]$WorkerExe = "build-vscode\bin\aitrain_worker.exe",
    [string]$Device = "0",
    [int]$Epochs = 30,
    [int]$ImageSize = 256,
    [int]$BatchSize = 8,
    [switch]$NoPretrained,
    [switch]$SkipDownload,
    [int]$MaxSamplesPerSplit = 0,
    [switch]$SkipProductRuntime,
    [switch]$SkipEnvironmentInstall
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
    $Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-Content -LiteralPath $Path -Encoding UTF8 -Raw | ConvertFrom-Json)
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
        Write-Host ("SMP Oxford Pets matrix: {0} attempt {1}/{2}" -f $Description, $attempt, $Attempts) -ForegroundColor Cyan
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
    param(
        [string]$PythonExe,
        [string]$RequestedDevice
    )
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $venvScripts = Split-Path -Parent $PythonExe
        $venvRoot = Split-Path -Parent $venvScripts
        Write-Host "SMP Oxford Pets matrix: create venv $venvRoot" -ForegroundColor Cyan
        & $BasePython -m venv $venvRoot 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create SMP venv: $venvRoot"
        }
    }

    $info = Probe-SmpEnvironment $PythonExe
    $missing = @($info.missing)
    $requiresCuda = -not [string]::Equals($RequestedDevice, "cpu", [System.StringComparison]::OrdinalIgnoreCase)
    $needsTorch = $missing -contains "torch" -or ($requiresCuda -and -not $info.cudaAvailable)
    $needsSmp = $missing.Count -gt 0
    if (($needsTorch -or $needsSmp) -and $SkipEnvironmentInstall) {
        throw "SMP environment is incomplete and -SkipEnvironmentInstall was set. Missing: $($missing -join ', '); cudaAvailable=$($info.cudaAvailable)"
    }
    if ($needsTorch -or $needsSmp) {
        Write-Host "SMP Oxford Pets matrix: install/repair Python dependencies" -ForegroundColor Cyan
        Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "--upgrade", "pip") "pip upgrade" 2
        if ($needsTorch) {
            try {
                if ($requiresCuda) {
                    Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "--index-url", "https://download.pytorch.org/whl/cu126", "torch", "torchvision") "CUDA PyTorch install" 3
                } else {
                    Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "torch", "torchvision") "PyTorch install" 3
                }
            } catch {
                if (-not $requiresCuda) {
                    throw
                }
                Write-Host "SMP Oxford Pets matrix: CUDA PyTorch download failed; probing local CUDA Torch providers" -ForegroundColor Yellow
                $provider = Find-CudaTorchProvider $PythonExe
                if ($null -eq $provider) {
                    throw
                }
                Enable-CudaTorchProvider $PythonExe $provider
                Write-Host ("SMP Oxford Pets matrix: using CUDA Torch provider {0} torch={1}" -f $provider.python, $provider.torch) -ForegroundColor Yellow
            }
        }
        Invoke-PipInstall $PythonExe @("install", "--retries", "10", "--timeout", "120", "segmentation-models-pytorch>=0.5,<0.6", "timm", "numpy", "Pillow", "PyYAML", "onnx", "onnxruntime") "SMP dependency install" 3
        $info = Probe-SmpEnvironment $PythonExe
    }
    $missing = @($info.missing)
    if ($missing.Count -gt 0) {
        throw "SMP environment still misses modules: $($missing -join ', ')"
    }
    if ($requiresCuda -and -not $info.cudaAvailable) {
        throw "SMP Oxford Pets matrix requires CUDA PyTorch for Device=$RequestedDevice. Use -Device cpu only for an explicit CPU downgrade."
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
        [string]$EncoderWeights
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
            batchSize = $BatchSize
            imageSize = $ImageSize
            device = $Device
            workers = 0
            learningRate = 0.001
            optimizer = "adamw"
            loss = "dice_ce"
            encoderWeights = $EncoderWeights
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

function Test-PretrainedBlocker {
    param([string]$Text)
    return ($Text -match "pretrain|pre-trained|encoder weights|download|HTTP|URL|checkpoint|SSL|timed out|Connection|model weights")
}

function Get-Number {
    param(
        [object]$Value,
        [double]$Default = 0.0
    )
    if ($null -eq $Value) {
        return $Default
    }
    $parsed = 0.0
    if ([double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        return $parsed
    }
    return $Default
}

function Get-OptionalValue {
    param(
        [object]$Object,
        [string]$Name
    )
    if ($null -eq $Object) {
        return $null
    }
    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) {
            return $Object[$Name]
        }
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-ProductRuntimeP95 {
    param([object]$ProductRuntime)
    $value = Get-OptionalValue $ProductRuntime "p95Ms"
    if ($null -eq $value) {
        return $null
    }
    return [double]$value
}

function Invoke-SmpTrainingRow {
    param(
        [string]$Preset,
        [string]$DatasetPath,
        [string]$RunsRoot,
        [string]$RequestsRoot,
        [string]$LogsRoot,
        [string]$EncoderWeights
    )
    $safePreset = $Preset -replace '[^A-Za-z0-9_.-]', '_'
    $runDir = Join-Path $RunsRoot $safePreset
    $requestPath = Join-Path $RequestsRoot "$safePreset-train.json"
    $logPath = Join-Path $LogsRoot "$safePreset-train.log"
    New-SmpRequest `
        -DatasetPath $DatasetPath `
        -OutputPath $runDir `
        -RequestPath $requestPath `
        -TaskId "smp-oxford-pets-$safePreset" `
        -Preset $Preset `
        -EncoderWeights $EncoderWeights
    Write-Host "SMP Oxford Pets matrix: train $Preset epochs=$Epochs batch=$BatchSize imageSize=$ImageSize weights=$EncoderWeights" -ForegroundColor Cyan
    $result = Invoke-Logged $pythonExe @((Join-Path $root "python_trainers\semantic_segmentation\smp_trainer.py"), "--request", $requestPath) $logPath
    $artifacts = Test-SmpArtifacts $runDir
    $status = if ($result.exitCode -eq 0 -and $artifacts.ok) {
        "passed"
    } elseif ((-not $NoPretrained) -and (Test-PretrainedBlocker $result.text)) {
        "blocked_pretrained_weights"
    } elseif ($result.text -match "no available encoder|Unsupported SMP preset|architecture .* is unavailable") {
        "blocked"
    } else {
        "failed"
    }
    $trainingReport = Read-JsonFile (Join-Path $runDir "smp_training_report.json")
    return @{
        preset = $Preset
        status = $status
        ok = ($status -eq "passed")
        exitCode = $result.exitCode
        runDir = $runDir
        requestPath = $requestPath
        logPath = $logPath
        artifacts = $artifacts
        trainingReport = $trainingReport
        architecture = if ($trainingReport) { $trainingReport.architecture } else { "" }
        encoder = if ($trainingReport) { $trainingReport.encoder } else { "" }
        encoderWeights = $EncoderWeights
        epochs = $Epochs
        imageSize = $ImageSize
        batchSize = $BatchSize
        device = $Device
    }
}

function Invoke-SmpEvaluation {
    param(
        [object]$Row,
        [string]$Split,
        [string]$DatasetPath,
        [string]$EvaluationRoot,
        [string]$RequestsRoot,
        [string]$LogsRoot
    )
    if (-not $Row.ok) {
        return @{
            ok = $false
            skipped = $true
            split = $Split
        }
    }
    $safePreset = $Row.preset -replace '[^A-Za-z0-9_.-]', '_'
    $outputPath = Join-Path $EvaluationRoot (Join-Path $safePreset $Split)
    $requestPath = Join-Path $RequestsRoot "$safePreset-eval-$Split.json"
    $logPath = Join-Path $LogsRoot "$safePreset-eval-$Split.log"
    Write-Json $requestPath @{
        modelPath = $Row.artifacts.onnxPath
        datasetPath = $DatasetPath
        outputPath = $outputPath
        options = @{
            split = $Split
            maxOverlays = 8
            ignoreIndex = 255
            lowQualityThreshold = 0.5
        }
    }
    Write-Host "SMP Oxford Pets matrix: evaluate $($Row.preset) split=$Split" -ForegroundColor Cyan
    $result = Invoke-Logged $pythonExe @((Join-Path $root "python_trainers\semantic_segmentation\smp_evaluator.py"), "--request", $requestPath) $logPath
    $reportPath = Join-Path $outputPath "evaluation_report.json"
    $report = Read-JsonFile $reportPath
    return @{
        ok = ($result.exitCode -eq 0 -and $null -ne $report -and $report.ok)
        split = $Split
        exitCode = $result.exitCode
        outputPath = $outputPath
        requestPath = $requestPath
        logPath = $logPath
        reportPath = $reportPath
        report = $report
        metrics = if ($report) { $report.metrics } else { $null }
    }
}

function Get-TestSamples {
    param([string]$DatasetPath)
    $imageRoot = Join-Path $DatasetPath "images\test"
    if (-not (Test-Path -LiteralPath $imageRoot)) {
        return @()
    }
    return @(Get-ChildItem -LiteralPath $imageRoot -File |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp)$' } |
        Sort-Object Name |
        Select-Object -First 3 |
        ForEach-Object { $_.FullName })
}

function Convert-ProductRuntimeSamples {
    param(
        [string[]]$Samples,
        [string]$OutputRoot
    )
    if ($Samples.Count -eq 0) {
        return @()
    }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    try {
        Add-Type -AssemblyName System.Drawing -ErrorAction Stop
    } catch {
        throw "Unable to load System.Drawing for product-runtime PNG sample conversion: $($_.Exception.Message)"
    }
    $converted = @()
    foreach ($sample in $Samples) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($sample)
        $target = Join-Path $OutputRoot "$stem.png"
        $image = $null
        try {
            $image = [System.Drawing.Image]::FromFile($sample)
            $image.Save($target, [System.Drawing.Imaging.ImageFormat]::Png)
        } finally {
            if ($null -ne $image) {
                $image.Dispose()
            }
        }
        $converted += $target
    }
    return $converted
}

function Get-BenchmarkP95 {
    param([object]$SmokeJson)
    if ($null -eq $SmokeJson) {
        return $null
    }
    $benchmark = Get-OptionalValue $SmokeJson "benchmark"
    $p95 = Get-OptionalValue $benchmark "p95Ms"
    if ($null -ne $p95) {
        return [double]$p95
    }
    $latency = Get-OptionalValue $benchmark "latency"
    $nestedP95 = Get-OptionalValue $latency "p95Ms"
    if ($null -ne $nestedP95) {
        return [double]$nestedP95
    }
    return $null
}

function Invoke-ProductRuntime {
    param(
        [object]$Row,
        [string[]]$Samples,
        [string]$ProductRoot,
        [string]$LogsRoot
    )
    if ($SkipProductRuntime) {
        return @{
            skipped = $true
            ok = $true
            status = "skipped"
            p95Ms = $null
            smokes = @()
        }
    }
    if (-not $Row.ok) {
        return @{
            skipped = $true
            ok = $false
            status = "skipped_training_not_passed"
            p95Ms = $null
            smokes = @()
        }
    }
    if (-not (Test-Path -LiteralPath $workerPath)) {
        return @{
            skipped = $true
            ok = $false
            status = "skipped_worker_missing"
            worker = $workerPath
            p95Ms = $null
            smokes = @()
        }
    }
    if ($Samples.Count -eq 0) {
        return @{
            skipped = $true
            ok = $false
            status = "skipped_sample_missing"
            p95Ms = $null
            smokes = @()
        }
    }
    $safePreset = $Row.preset -replace '[^A-Za-z0-9_.-]', '_'
    $smokes = @()
    $allOk = $true
    foreach ($sample in $Samples) {
        $sampleStem = [System.IO.Path]::GetFileNameWithoutExtension($sample)
        $outputPath = Join-Path $ProductRoot (Join-Path $safePreset $sampleStem)
        $logPath = Join-Path $LogsRoot "$safePreset-product-$sampleStem.log"
        Write-Host "SMP Oxford Pets matrix: product runtime $($Row.preset) sample=$sampleStem" -ForegroundColor Cyan
        $result = Invoke-Logged $workerPath @("--semantic-onnx-smoke", $Row.artifacts.onnxPath, "--image", $sample, "--output", $outputPath) $logPath
        $json = Read-LastJsonLine $result.text
        $ok = ($result.exitCode -eq 0 -and $null -ne $json -and $json.ok)
        if (-not $ok) {
            $allOk = $false
        }
        $smokes += @{
            sampleImagePath = $sample
            outputPath = $outputPath
            logPath = $logPath
            exitCode = $result.exitCode
            ok = $ok
            result = $json
            p95Ms = Get-BenchmarkP95 $json
            overlayPath = [string](Get-OptionalValue $json "overlayPath")
        }
    }
    $p95Values = @($smokes | ForEach-Object { $_.p95Ms } | Where-Object { $null -ne $_ })
    $p95 = if ($p95Values.Count -gt 0) { ($p95Values | Measure-Object -Average).Average } else { $null }
    return @{
        skipped = $false
        ok = $allOk
        status = if ($allOk) { "passed" } else { "failed" }
        smokes = $smokes
        p95Ms = $p95
    }
}

function Get-SmpDeploymentScope {
    param(
        [object]$BestRow
    )
    $bestPreset = ""
    if ($BestRow) {
        $bestPreset = $BestRow.preset
    }
    return @{
        ok = $true
        status = "not_required"
        scope = "onnx_runtime_only"
        bestPreset = $bestPreset
        onnxRuntime = "required"
        ncnn = "not_required"
        tensorrt = "not_required"
        note = "SMP semantic segmentation does not require NCNN/TensorRT export. This matrix validates ONNX Runtime inference, overlay, benchmark, and deployment only."
    }
}

function New-ContactSheet {
    param(
        [object[]]$Rows,
        [string]$OutputPath
    )
    $items = @()
    foreach ($row in $Rows) {
        $overlay = ""
        if ($row.productRuntime -and $row.productRuntime.smokes -and $row.productRuntime.smokes.Count -gt 0) {
            $overlay = [string]$row.productRuntime.smokes[0].overlayPath
        }
        if ([string]::IsNullOrWhiteSpace($overlay) -and $row.testEvaluation -and $row.testEvaluation.report -and $row.testEvaluation.report.lowQualitySamples) {
            foreach ($sample in @($row.testEvaluation.report.lowQualitySamples)) {
                if (-not [string]::IsNullOrWhiteSpace([string]$sample.overlayPath)) {
                    $overlay = [string]$sample.overlayPath
                    break
                }
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($overlay) -and (Test-Path -LiteralPath $overlay)) {
            $items += @{
                label = $row.preset
                path = $overlay
            }
        }
    }
    if ($items.Count -eq 0) {
        return @{
            ok = $false
            skipped = $true
            reason = "no overlay artifacts available"
            path = $OutputPath
        }
    }
    $payloadPath = Join-Path (Split-Path -Parent $OutputPath) "contact_sheet_items.json"
    Write-Json $payloadPath @{ items = $items; outputPath = $OutputPath }
    $script = @"
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
items = payload['items']
thumb_w, thumb_h = 256, 256
label_h = 34
cols = min(5, max(1, len(items)))
rows = (len(items) + cols - 1) // cols
sheet = Image.new('RGB', (cols * thumb_w, rows * (thumb_h + label_h)), (248, 248, 248))
draw = ImageDraw.Draw(sheet)
for idx, item in enumerate(items):
    col = idx % cols
    row = idx // cols
    x = col * thumb_w
    y = row * (thumb_h + label_h)
    image = Image.open(item['path']).convert('RGB')
    image.thumbnail((thumb_w, thumb_h))
    ox = x + (thumb_w - image.width) // 2
    oy = y + (thumb_h - image.height) // 2
    sheet.paste(image, (ox, oy))
    draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(238, 238, 238))
    draw.text((x + 8, y + thumb_h + 9), item['label'][:38], fill=(20, 20, 20))
out = Path(payload['outputPath'])
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(json.dumps({'ok': True, 'path': str(out)}, ensure_ascii=False))
"@
    $output = & $pythonExe -c $script $payloadPath
    if ($LASTEXITCODE -ne 0) {
        return @{
            ok = $false
            skipped = $false
            path = $OutputPath
            error = ($output -join "`n")
        }
    }
    return @{
        ok = $true
        path = $OutputPath
        itemCount = $items.Count
    }
}

function Export-ComparisonCsv {
    param(
        [object[]]$Rows,
        [string]$CsvPath
    )
    $records = @()
    foreach ($row in $Rows) {
        $valMetrics = if ($row.valEvaluation) { $row.valEvaluation.metrics } else { $null }
        $testMetrics = if ($row.testEvaluation) { $row.testEvaluation.metrics } else { $null }
        $records += [pscustomobject]@{
            preset = $row.preset
            status = $row.status
            architecture = $row.architecture
            encoder = $row.encoder
            epochs = $row.epochs
            imageSize = $row.imageSize
            batchSize = $row.batchSize
            encoderWeights = $row.encoderWeights
            val_mIoU = Get-Number (Get-OptionalValue $valMetrics "mIoU")
            val_meanDice = Get-Number (Get-OptionalValue $valMetrics "meanDice")
            val_pixelAccuracy = Get-Number (Get-OptionalValue $valMetrics "pixelAccuracy")
            test_mIoU = Get-Number (Get-OptionalValue $testMetrics "mIoU")
            test_meanDice = Get-Number (Get-OptionalValue $testMetrics "meanDice")
            test_pixelAccuracy = Get-Number (Get-OptionalValue $testMetrics "pixelAccuracy")
            p95Ms = Get-ProductRuntimeP95 $row.productRuntime
            onnxPath = $row.artifacts.onnxPath
            trainingReportPath = $row.artifacts.trainingReportPath
            testEvaluationReportPath = if ($row.testEvaluation) { $row.testEvaluation.reportPath } else { "" }
        }
    }
    $records | ConvertTo-Csv -NoTypeInformation | Set-Content -LiteralPath $CsvPath -Encoding UTF8
}

function Export-ComparisonMarkdown {
    param(
        [object[]]$RankedRows,
        [object[]]$Rows,
        [object]$BestRow,
        [string]$MarkdownPath,
        [string]$ContactSheetPath
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# SMP Oxford Pets Model Comparison")
    $lines.Add("")
    $lines.Add("Oxford-IIIT Pet is used here as a public semantic segmentation quality comparison dataset. It is not industrial defect or customer-domain production precision evidence.")
    $lines.Add("")
    $lines.Add("## Ranking")
    $lines.Add("")
    $lines.Add("| Rank | Preset | Architecture | Encoder | Test mIoU | Test meanDice | Test pixelAccuracy | p95Ms | Status |")
    $lines.Add("|---:|---|---|---|---:|---:|---:|---:|---|")
    $rank = 1
    foreach ($row in $RankedRows) {
        $testMetrics = $row.testEvaluation.metrics
        $p95Value = Get-ProductRuntimeP95 $row.productRuntime
        $p95 = if ($null -ne $p95Value) { "{0:N2}" -f $p95Value } else { "" }
        $lines.Add(("| {0} | {1} | {2} | {3} | {4:N6} | {5:N6} | {6:N6} | {7} | {8} |" -f `
            $rank, $row.preset, $row.architecture, $row.encoder, `
            (Get-Number $testMetrics.mIoU), (Get-Number $testMetrics.meanDice), (Get-Number $testMetrics.pixelAccuracy), $p95, $row.status))
        $rank += 1
    }
    if ($RankedRows.Count -eq 0) {
        $lines.Add("|  |  |  |  |  |  |  |  | no passed rows |")
    }
    $lines.Add("")
    $lines.Add("## Recommendation")
    $lines.Add("")
    if ($BestRow) {
        $lines.Add(('Best model by test mIoU, meanDice, pixelAccuracy, then p95 latency: `{0}`.' -f $BestRow.preset))
        $lines.Add(('ONNX artifact: `{0}`' -f $BestRow.artifacts.onnxPath))
    } else {
        $lines.Add("No passed model row is available for recommendation.")
    }
    $lines.Add("")
    $lines.Add("## Row Status")
    $lines.Add("")
    foreach ($row in $Rows) {
        $lines.Add(('- `{0}`: {1}' -f $row.preset, $row.status))
    }
    $lines.Add("")
    $lines.Add("## Deployment Notes")
    $lines.Add("")
    $lines.Add("- Product runtime smoke uses AITrain C++ ONNX Runtime inference, overlay, benchmark, and deployment validation.")
    $lines.Add("- SMP deployment scope is ONNX Runtime only; NCNN/TensorRT export is not required and is not run by this matrix.")
    $lines.Add('- Default pretrained encoder weights are ImageNet; use `-NoPretrained` for explicit offline runs.')
    if (Test-Path -LiteralPath $ContactSheetPath) {
        $lines.Add(('- Contact sheet: `{0}`' -f $ContactSheetPath))
    }
    $MarkdownPath | Split-Path -Parent | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
    Set-Content -LiteralPath $MarkdownPath -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
}

$pythonExe = Resolve-RepoPath $Python
$workerPath = Resolve-RepoPath $WorkerExe
$workFull = Resolve-RepoPath $WorkDir
$summaryPath = Join-Path $workFull "smp_oxford_pets_quality_matrix_summary.json"
$csvPath = Join-Path $workFull "smp_oxford_pets_model_comparison.csv"
$markdownPath = Join-Path $workFull "smp_oxford_pets_model_comparison.md"
$contactSheetPath = Join-Path $workFull "overlays\contact_sheet.png"
New-Item -ItemType Directory -Force -Path $workFull | Out-Null

$envInfo = Ensure-SmpEnvironment $pythonExe $Device
Write-Host ("SMP Oxford Pets matrix: environment ready torch={0} cuda={1} device={2}" -f $envInfo.torch, $envInfo.cudaAvailable, $envInfo.deviceName) -ForegroundColor Green

$requestsRoot = Join-Path $workFull "requests"
$logsRoot = Join-Path $workFull "logs"
$runsRoot = Join-Path $workFull "runs"
$evaluationRoot = Join-Path $workFull "evaluations"
$productRoot = Join-Path $workFull "product-runtime"
New-Item -ItemType Directory -Force -Path $requestsRoot, $logsRoot, $runsRoot, $evaluationRoot, $productRoot | Out-Null

Write-Host "SMP Oxford Pets matrix: materialize public dataset" -ForegroundColor Cyan
$materializerArgs = @(
    (Join-Path $root "tools\materialize-oxford-pets-semantic.py"),
    "--work-dir", $workFull,
    "--seed", "42"
)
if ($SkipDownload) {
    $materializerArgs += "--skip-download"
}
if ($MaxSamplesPerSplit -gt 0) {
    $materializerArgs += @("--max-samples-per-split", [string]$MaxSamplesPerSplit)
}
$datasetLogPath = Join-Path $logsRoot "materialize-oxford-pets.log"
$datasetResult = Invoke-Logged $pythonExe $materializerArgs $datasetLogPath
if ($datasetResult.exitCode -ne 0) {
    Write-Json $summaryPath @{
        ok = $false
        status = "failed"
        stage = "dataset_materialization"
        workDir = $workFull
        python = $pythonExe
        materializeLogPath = $datasetLogPath
    }
    throw "Oxford Pets materialization failed. See $datasetLogPath"
}
$datasetPath = Join-Path $workFull "semantic_mask_oxford_pets"
$datasetManifestPath = Join-Path $datasetPath "dataset_manifest.json"
$datasetManifest = Read-JsonFile $datasetManifestPath
$testSamples = @(Get-TestSamples $datasetPath)
if ($testSamples.Count -eq 0) {
    throw "Oxford Pets materialization produced no test sample images: $datasetPath"
}
$productSamples = if ($SkipProductRuntime) {
    $testSamples
} else {
    @(Convert-ProductRuntimeSamples -Samples $testSamples -OutputRoot (Join-Path $productRoot "input-samples"))
}

$encoderWeights = if ($NoPretrained) { "none" } else { "imagenet" }
$presets = @(
    "smp_unet_resnet34",
    "smp_unetplusplus_resnet34",
    "smp_fpn_resnet34",
    "smp_deeplabv3plus_resnet50",
    "smp_segformer_mit_b0"
)

$rows = @()
foreach ($preset in $presets) {
    $row = Invoke-SmpTrainingRow `
        -Preset $preset `
        -DatasetPath $datasetPath `
        -RunsRoot $runsRoot `
        -RequestsRoot $requestsRoot `
        -LogsRoot $logsRoot `
        -EncoderWeights $encoderWeights
    $row.valEvaluation = Invoke-SmpEvaluation -Row $row -Split "val" -DatasetPath $datasetPath -EvaluationRoot $evaluationRoot -RequestsRoot $requestsRoot -LogsRoot $logsRoot
    $row.testEvaluation = Invoke-SmpEvaluation -Row $row -Split "test" -DatasetPath $datasetPath -EvaluationRoot $evaluationRoot -RequestsRoot $requestsRoot -LogsRoot $logsRoot
    if ($row.ok -and (-not $row.valEvaluation.ok -or -not $row.testEvaluation.ok)) {
        $row.status = "failed"
        $row.ok = $false
    }
    $row.productRuntime = Invoke-ProductRuntime -Row $row -Samples $productSamples -ProductRoot $productRoot -LogsRoot $logsRoot
    if ($row.ok -and -not $SkipProductRuntime -and -not $row.productRuntime.ok) {
        $row.status = "failed"
        $row.ok = $false
    }
    $rows += $row
}

$passedRows = @($rows | Where-Object { $_.ok -and $_.testEvaluation -and $_.testEvaluation.ok })
$rankedRows = @($passedRows | Sort-Object `
    @{ Expression = { -1.0 * (Get-Number $_.testEvaluation.metrics.mIoU) } }, `
    @{ Expression = { -1.0 * (Get-Number $_.testEvaluation.metrics.meanDice) } }, `
    @{ Expression = { -1.0 * (Get-Number $_.testEvaluation.metrics.pixelAccuracy) } }, `
    @{ Expression = {
        $p95 = Get-ProductRuntimeP95 $_.productRuntime
        if ($null -ne $p95) { $p95 } else { [double]::PositiveInfinity }
    } })
$bestRow = if ($rankedRows.Count -gt 0) { $rankedRows[0] } else { $null }

$deploymentScope = Get-SmpDeploymentScope -BestRow $bestRow

$contactSheet = New-ContactSheet -Rows $rows -OutputPath $contactSheetPath
Export-ComparisonCsv -Rows $rows -CsvPath $csvPath
Export-ComparisonMarkdown -RankedRows $rankedRows -Rows $rows -BestRow $bestRow -MarkdownPath $markdownPath -ContactSheetPath $contactSheetPath

$failedRows = @($rows | Where-Object { $_.status -eq "failed" })
$blockedRows = @($rows | Where-Object { $_.status -like "blocked*" })
$productRuntimePassedCount = @($rows | Where-Object { $_.productRuntime -and $_.productRuntime.ok -and -not $_.productRuntime.skipped }).Count
$productRuntimeOk = $SkipProductRuntime -or $productRuntimePassedCount -ge 1
$ok = ($failedRows.Count -eq 0 -and $blockedRows.Count -eq 0 -and $rankedRows.Count -gt 0 -and $productRuntimeOk)
$status = if ($ok) {
    "passed"
} elseif ($blockedRows.Count -gt 0 -and $failedRows.Count -eq 0) {
    "blocked"
} else {
    "failed"
}

$summary = @{
    ok = $ok
    status = $status
    workDir = $workFull
    python = $pythonExe
    worker = $workerPath
    device = $Device
    epochs = $Epochs
    imageSize = $ImageSize
    batchSize = $BatchSize
    optimizer = "adamw"
    loss = "dice_ce"
    seed = 42
    encoderWeights = $encoderWeights
    noPretrained = $NoPretrained.IsPresent
    maxSamplesPerSplit = $MaxSamplesPerSplit
    environment = $envInfo
    cudaTorchProvider = $script:cudaTorchProvider
    datasetPath = $datasetPath
    datasetManifestPath = $datasetManifestPath
    datasetManifest = $datasetManifest
    sourceDataset = @{
        name = "Oxford-IIIT Pet"
        sourcePage = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
        imageUrl = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        annotationUrl = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        license = "CC BY-SA 4.0"
        note = "Public dataset quality comparison only; not industrial defect/customer-domain production precision evidence."
    }
    rows = $rows
    rankedPresets = @($rankedRows | ForEach-Object { $_.preset })
    bestPreset = if ($bestRow) { $bestRow.preset } else { "" }
    failedRows = $failedRows
    blockedRows = $blockedRows
    productRuntimePassedCount = $productRuntimePassedCount
    deploymentScope = $deploymentScope
    reports = @{
        summary = $summaryPath
        csv = $csvPath
        markdown = $markdownPath
        contactSheet = $contactSheetPath
        contactSheetResult = $contactSheet
    }
}
Write-Json $summaryPath $summary

if (-not $ok) {
    throw "SMP Oxford Pets quality matrix did not pass: $summaryPath"
}
Write-Host "SMP Oxford Pets quality matrix passed: $summaryPath" -ForegroundColor Green
