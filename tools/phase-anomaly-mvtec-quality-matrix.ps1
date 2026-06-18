param(
    [string]$PythonExecutable = "",
    [string]$CondaExecutable = "conda",
    [string]$EnvPath = ".deps\envs\anomalib",
    [string]$WorkDir = ".deps\anomaly-mvtec-quality-matrix",
    [string]$MvtecRoot = ".deps\datasets\materialized\mvtec-ad",
    [string]$MvtecArchivePath = ".deps\datasets\downloads\mvtec_ad\mvtec_anomaly_detection.tar.xz",
    [string]$MvtecArchiveUrl = "",
    [string]$ImagenetteDir = ".deps\anomalib\imagenette",
    [string]$ImagenetteUrl = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    [string[]]$Categories = @("bottle", "hazelnut", "leather"),
    [string[]]$Backends = @("anomalib_patchcore", "anomalib_efficientad"),
    [string]$Device = "0",
    [int]$Epochs = 5,
    [int]$ImageSize = 256,
    [int]$BatchSize = 2,
    [int]$Workers = 4,
    [int]$BenchmarkWarmup = 10,
    [int]$BenchmarkIterations = 50,
    [switch]$SkipEnvironmentSetup,
    [switch]$SkipDownloads,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = Split-Path -Parent $PSScriptRoot
Set-Location $script:Root

$script:Summary = [ordered]@{
    ok = $false
    status = "blocked"
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    runtime = "anomalib_python"
    dataset = "MVTec AD public subset"
    categories = @($Categories)
    backends = @($Backends)
    parameters = [ordered]@{
        device = $Device
        epochs = $Epochs
        imageSize = $ImageSize
        batchSize = $BatchSize
        workers = $Workers
        benchmarkWarmup = $BenchmarkWarmup
        benchmarkIterations = $BenchmarkIterations
    }
    checks = @()
    categoryPreflight = @()
    rows = @()
    notes = @(
        "Public MVTec evidence is workflow/quality evidence, not customer-domain production precision.",
        "Anomaly v1 runtime is Worker-managed Python/Anomalib artifacts, not AITrain C++ ONNX/TensorRT/NCNN."
    )
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Resolve-Executable {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $Path
    }
    if ((-not [System.IO.Path]::IsPathRooted($Path)) -and ($Path.IndexOfAny([char[]]@('\', '/')) -lt 0)) {
        return $Path
    }
    return Resolve-RepoPath $Path
}

function Test-IsBareCommand {
    param([string]$Path)
    return -not [string]::IsNullOrWhiteSpace($Path) `
        -and -not [System.IO.Path]::IsPathRooted($Path) `
        -and $Path.IndexOfAny([char[]]@('\', '/')) -lt 0
}

function Assert-UnderDeps {
    param([string]$Path, [string]$Description)
    $full = [System.IO.Path]::GetFullPath($Path)
    $deps = [System.IO.Path]::GetFullPath((Join-Path $script:Root ".deps"))
    if (-not $full.StartsWith($deps, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use $Description outside .deps: $full"
    }
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 -LiteralPath $Path
}

function Add-Check {
    param([string]$Name, [string]$Status, [string]$Message, [hashtable]$Details = @{})
    $item = [ordered]@{
        name = $Name
        status = $Status
        message = $Message
    }
    foreach ($key in $Details.Keys) {
        $item[$key] = $Details[$key]
    }
    $script:Summary.checks += $item
}

function ConvertTo-ProcessArgument {
    param([string]$Argument)
    if ($null -eq $Argument -or $Argument.Length -eq 0) {
        return '""'
    }
    if ($Argument.IndexOfAny([char[]]@(' ', "`t", "`r", "`n", '"')) -lt 0) {
        return $Argument
    }
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($char in $Argument.ToCharArray()) {
        if ($char -eq [char]92) {
            $backslashes += 1
        } elseif ($char -eq '"') {
            if ($backslashes -gt 0) {
                [void]$builder.Append([String]::new([char]92, $backslashes * 2))
                $backslashes = 0
            }
            [void]$builder.Append('\"')
        } else {
            if ($backslashes -gt 0) {
                [void]$builder.Append([String]::new([char]92, $backslashes))
                $backslashes = 0
            }
            [void]$builder.Append($char)
        }
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append([String]::new([char]92, $backslashes * 2))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-Logged {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$LogPath,
        [hashtable]$Environment = @{}
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $psi.WorkingDirectory = $script:Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($key in $Environment.Keys) {
        $psi.Environment[$key] = [string]$Environment[$key]
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $started = Get-Date
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $ended = Get-Date
    $log = @()
    $log += "name=$Name"
    $log += "command=$FilePath $($Arguments -join ' ')"
    $log += "startedAt=$($started.ToUniversalTime().ToString('o'))"
    $log += "finishedAt=$($ended.ToUniversalTime().ToString('o'))"
    $log += "exitCode=$($process.ExitCode)"
    $log += ""
    $log += "----- stdout -----"
    $log += $stdout
    $log += ""
    $log += "----- stderr -----"
    $log += $stderr
    $log -join "`n" | Set-Content -Encoding UTF8 -LiteralPath $LogPath
    return [ordered]@{
        name = $Name
        exitCode = $process.ExitCode
        logPath = $LogPath
        stdout = $stdout
        stderr = $stderr
        elapsedSeconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
    }
}

function Read-JsonObject {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -Encoding UTF8 -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Test-LogLooksLikeOom {
    param([string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }
    $text = Get-Content -Encoding UTF8 -LiteralPath $LogPath -Raw
    return $text -match "out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED"
}

function Ensure-CondaEnvironment {
    param([string]$PythonPath, [string]$EnvironmentPath)
    if ((Test-IsBareCommand $PythonPath) -or (Test-Path -LiteralPath $PythonPath)) {
        return
    }
    if ($SkipEnvironmentSetup) {
        throw "Anomalib Python executable is missing and -SkipEnvironmentSetup was set: $PythonPath"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $EnvironmentPath) | Out-Null
    $logRoot = Join-Path $script:WorkFull "logs\environment"
    $create = Invoke-Logged `
        -Name "conda-create-anomalib" `
        -FilePath $CondaExecutable `
        -Arguments @("create", "-y", "-p", $EnvironmentPath, "python=3.10") `
        -LogPath (Join-Path $logRoot "conda_create.log")
    if ($create.exitCode -ne 0) {
        throw "Failed to create Anomalib conda env. Inspect $($create.logPath)"
    }
}

function Ensure-PythonPackages {
    param([string]$PythonPath)
    if ($SkipEnvironmentSetup) {
        return
    }
    $probe = Invoke-Logged `
        -Name "probe-anomalib-env-before-install" `
        -FilePath $PythonPath `
        -Arguments @("-c", "import importlib.util, sys; mods=['anomalib','torch','torchvision','lightning','timm','PIL','numpy','cv2']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 3)") `
        -LogPath (Join-Path $script:WorkFull "logs\environment\probe_before_install.log")
    if ($probe.exitCode -eq 0) {
        return
    }
    $pipUpgrade = Invoke-Logged `
        -Name "pip-upgrade" `
        -FilePath $PythonPath `
        -Arguments @("-m", "pip", "install", "--retries", "10", "--timeout", "120", "--upgrade", "pip", "setuptools", "wheel") `
        -LogPath (Join-Path $script:WorkFull "logs\environment\pip_upgrade.log")
    if ($pipUpgrade.exitCode -ne 0) {
        throw "Failed to upgrade pip. Inspect $($pipUpgrade.logPath)"
    }
    $torchInstall = Invoke-Logged `
        -Name "install-cuda-pytorch" `
        -FilePath $PythonPath `
        -Arguments @("-m", "pip", "install", "--retries", "10", "--timeout", "120", "--index-url", "https://download.pytorch.org/whl/cu126", "torch", "torchvision") `
        -LogPath (Join-Path $script:WorkFull "logs\environment\pip_torch_cuda.log")
    if ($torchInstall.exitCode -ne 0) {
        throw "Failed to install CUDA PyTorch. Inspect $($torchInstall.logPath)"
    }
    $requirements = Join-Path $script:Root "python_trainers\requirements-anomaly.txt"
    $anomalibInstall = Invoke-Logged `
        -Name "install-anomalib-requirements" `
        -FilePath $PythonPath `
        -Arguments @("-m", "pip", "install", "--retries", "10", "--timeout", "120", "-r", $requirements) `
        -LogPath (Join-Path $script:WorkFull "logs\environment\pip_anomalib.log")
    if ($anomalibInstall.exitCode -ne 0) {
        throw "Failed to install Anomalib requirements. Inspect $($anomalibInstall.logPath)"
    }
}

function Test-AnomalibEnvironment {
    param([string]$PythonPath)
    $code = @"
import importlib.util
import json
import sys
mods = ['anomalib','torch','torchvision','lightning','timm','PIL','numpy','cv2']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
import torch
payload = {
    'python': sys.executable,
    'pythonVersion': sys.version,
    'missing': missing,
    'torchVersion': getattr(torch, '__version__', ''),
    'cudaAvailable': bool(torch.cuda.is_available()),
    'cudaVersion': getattr(torch.version, 'cuda', None),
    'deviceCount': int(torch.cuda.device_count()),
}
print(json.dumps(payload, ensure_ascii=False))
sys.exit(0 if not missing and torch.cuda.is_available() else 3)
"@
    $result = Invoke-Logged `
        -Name "probe-anomalib-env" `
        -FilePath $PythonPath `
        -Arguments @("-c", $code) `
        -LogPath (Join-Path $script:WorkFull "logs\environment\probe_final.log")
    $payload = $null
    try {
        $payload = $result.stdout.Trim() | ConvertFrom-Json
    } catch {
        $payload = $null
    }
    if ($result.exitCode -eq 0) {
        Add-Check "anomalib_environment" "passed" "Anomalib Python environment is importable and CUDA is available." @{ python = $PythonPath; profile = $payload }
    } else {
        Add-Check "anomalib_environment" "blocked" "Anomalib environment probe failed. Inspect $($result.logPath)" @{ python = $PythonPath; profile = $payload }
        throw "Anomalib environment is not ready. Inspect $($result.logPath)"
    }
}

function Ensure-Imagenette {
    param([string]$TargetDir)
    if (Test-Path -LiteralPath (Join-Path $TargetDir "train")) {
        Add-Check "imagenette" "passed" "Imagenette directory is available." @{ path = $TargetDir }
        return
    }
    if ($SkipDownloads) {
        Add-Check "imagenette" "blocked" "Imagenette is missing and -SkipDownloads was set." @{ path = $TargetDir }
        throw "Imagenette directory is missing: $TargetDir"
    }
    $downloadDir = Join-Path $script:Root ".deps\anomalib\downloads"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $archive = Join-Path $downloadDir "imagenette2-160.tgz"
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri $ImagenetteUrl -OutFile $archive -UseBasicParsing
    }
    $extractRoot = Join-Path $downloadDir "extract"
    if (Test-Path -LiteralPath $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    tar -xzf $archive -C $extractRoot
    $source = Join-Path $extractRoot "imagenette2-160"
    if (-not (Test-Path -LiteralPath (Join-Path $source "train"))) {
        throw "Imagenette archive did not extract to expected layout: $source"
    }
    if (Test-Path -LiteralPath $TargetDir) {
        Remove-Item -LiteralPath $TargetDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TargetDir) | Out-Null
    Move-Item -LiteralPath $source -Destination $TargetDir
    Add-Check "imagenette" "passed" "Imagenette was downloaded and extracted." @{ path = $TargetDir; source = $ImagenetteUrl }
}

function Find-MvtecCategorySource {
    param([string]$Category, [string[]]$Roots)
    foreach ($root in $Roots) {
        if ([string]::IsNullOrWhiteSpace($root) -or -not (Test-Path -LiteralPath $root)) {
            continue
        }
        $direct = Join-Path $root $Category
        if (Test-Path -LiteralPath (Join-Path $direct "train\good")) {
            return [System.IO.Path]::GetFullPath($direct)
        }
        $nested = Get-ChildItem -LiteralPath $root -Directory -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq $Category -and (Test-Path -LiteralPath (Join-Path $_.FullName "train\good")) } |
            Select-Object -First 1
        if ($nested) {
            return [System.IO.Path]::GetFullPath($nested.FullName)
        }
    }
    return ""
}

function Materialize-MvtecCategory {
    param([string]$Source, [string]$Target)
    if ((Test-Path -LiteralPath $Target) -and (Test-Path -LiteralPath (Join-Path $Target "train\good"))) {
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    try {
        New-Item -ItemType Junction -Path $Target -Target $Source | Out-Null
    } catch {
        robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -gt 7) {
            throw "Failed to materialize MVTec category $Source -> $Target"
        }
    }
}

function Ensure-MvtecCategories {
    param([string[]]$CategoryList)
    $downloadRoot = Split-Path -Parent $script:MvtecArchiveFull
    $extractRoot = Join-Path $downloadRoot "extracted"
    $roots = @($script:MvtecRootFull, $extractRoot)
    $missing = @()
    foreach ($category in $CategoryList) {
        $source = Find-MvtecCategorySource -Category $category -Roots $roots
        if ([string]::IsNullOrWhiteSpace($source)) {
            $missing += $category
        }
    }
    if ($missing.Count -gt 0) {
        if (-not (Test-Path -LiteralPath $script:MvtecArchiveFull)) {
            if ([string]::IsNullOrWhiteSpace($MvtecArchiveUrl) -or $SkipDownloads) {
                Add-Check "mvtec_dataset" "blocked" "MVTec categories are missing. Download the official MVTec AD archive from https://www.mvtec.com/research-teaching/datasets/mvtec-ad and place it at $script:MvtecArchiveFull, or pass -MvtecArchiveUrl." @{ missingCategories = $missing }
                throw "MVTec AD dataset is missing."
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $script:MvtecArchiveFull) | Out-Null
            Invoke-WebRequest -Uri $MvtecArchiveUrl -OutFile $script:MvtecArchiveFull -UseBasicParsing
        }
        if (Test-Path -LiteralPath $extractRoot) {
            Remove-Item -LiteralPath $extractRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
        tar -xJf $script:MvtecArchiveFull -C $extractRoot
    }

    foreach ($category in $CategoryList) {
        $target = Join-Path $script:MvtecRootFull $category
        $source = Find-MvtecCategorySource -Category $category -Roots @($script:MvtecRootFull, $extractRoot)
        if ([string]::IsNullOrWhiteSpace($source)) {
            Add-Check "mvtec_category_$category" "blocked" "MVTec category is missing after extraction." @{ category = $category }
            throw "MVTec category missing: $category"
        }
        Materialize-MvtecCategory -Source $source -Target $target
        Add-Check "mvtec_category_$category" "passed" "MVTec category is available." @{ category = $category; path = $target }
    }
}

$script:ImageExtensions = @(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

function Test-IsImageFile {
    param([System.IO.FileInfo]$File)
    return $script:ImageExtensions -contains $File.Extension.ToLowerInvariant()
}

function Get-RelativePath {
    param([string]$BasePath, [string]$Path)
    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $full = [System.IO.Path]::GetFullPath($Path)
    $baseUri = [System.Uri]::new($base)
    $pathUri = [System.Uri]::new($full)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

function Get-AnomalyRole {
    param([string]$RelativePath)
    $rel = $RelativePath.Replace('/', '\').ToLowerInvariant()
    $extension = [System.IO.Path]::GetExtension($rel)
    if ($rel.StartsWith("masks\") -or $rel.StartsWith("ground_truth\")) {
        return "mask"
    }
    if ($extension -in @(".json", ".yaml", ".yml", ".txt")) {
        return "config"
    }
    if ($rel.StartsWith("train\good\") -or $rel.StartsWith("val\good\") -or $rel.StartsWith("test\good\")) {
        return "normal_image"
    }
    if ($rel.StartsWith("val\anomaly\") -or $rel.StartsWith("test\anomaly\")) {
        return "anomaly_image"
    }
    if ($rel.StartsWith("test\") -or $rel.StartsWith("val\")) {
        return "anomaly_image"
    }
    return "asset"
}

function Get-SplitName {
    param([string]$RelativePath)
    $rel = $RelativePath.Replace('/', '\').ToLowerInvariant()
    if ($rel.StartsWith("train\")) { return "train" }
    if ($rel.StartsWith("val\")) { return "val" }
    if ($rel.StartsWith("test\")) { return "test" }
    if ($rel.StartsWith("ground_truth\")) { return "ground_truth" }
    if ($rel.StartsWith("masks\")) { return "masks" }
    return "other"
}

function Get-MaskStem {
    param([System.IO.FileInfo]$File)
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($File.Name)
    if ($stem.EndsWith("_mask", [System.StringComparison]::OrdinalIgnoreCase)) {
        return $stem.Substring(0, $stem.Length - 5)
    }
    return $stem
}

function New-StringSha256 {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hash = $sha.ComputeHash($bytes)
        return -join ($hash | ForEach-Object { $_.ToString("x2") })
    } finally {
        $sha.Dispose()
    }
}

function Invoke-CategoryPreflight {
    param([string]$Category)
    $datasetPath = Join-Path $script:MvtecRootFull $Category
    $preflightRoot = Join-Path $script:WorkFull "$Category\preflight"
    New-Item -ItemType Directory -Force -Path $preflightRoot | Out-Null

    $allFiles = @(Get-ChildItem -LiteralPath $datasetPath -Recurse -File)
    $imageFiles = @($allFiles | Where-Object { Test-IsImageFile $_ })
    $items = foreach ($file in $allFiles) {
        $relative = Get-RelativePath -BasePath $datasetPath -Path $file.FullName
        $role = Get-AnomalyRole -RelativePath $relative
        [ordered]@{
            path = $relative
            size = $file.Length
            role = $role
            split = Get-SplitName -RelativePath $relative
        }
    }
    $roleCounts = [ordered]@{}
    $splitCounts = [ordered]@{}
    foreach ($item in $items) {
        if (-not $roleCounts.Contains($item.role)) { $roleCounts[$item.role] = 0 }
        if (-not $splitCounts.Contains($item.split)) { $splitCounts[$item.split] = 0 }
        $roleCounts[$item.role] = [int]$roleCounts[$item.role] + 1
        $splitCounts[$item.split] = [int]$splitCounts[$item.split] + 1
    }

    $trainGood = @($imageFiles | Where-Object { (Get-RelativePath -BasePath $datasetPath -Path $_.FullName).ToLowerInvariant().StartsWith("train\good\") })
    $normalImages = @($items | Where-Object { $_.role -eq "normal_image" })
    $anomalyImages = @($items | Where-Object { $_.role -eq "anomaly_image" })
    $maskFiles = @($imageFiles | Where-Object {
        $rel = (Get-RelativePath -BasePath $datasetPath -Path $_.FullName).ToLowerInvariant()
        $rel.StartsWith("ground_truth\") -or $rel.StartsWith("masks\")
    })
    $anomalyStems = @{}
    foreach ($file in $imageFiles) {
        $rel = Get-RelativePath -BasePath $datasetPath -Path $file.FullName
        if ((Get-AnomalyRole -RelativePath $rel) -eq "anomaly_image") {
            $anomalyStems[[System.IO.Path]::GetFileNameWithoutExtension($file.Name)] = $true
        }
    }
    $maskStems = @{}
    foreach ($file in $maskFiles) {
        $maskStems[(Get-MaskStem $file)] = $true
    }
    $missingMasks = @($anomalyStems.Keys | Where-Object { -not $maskStems.Contains($_) } | Sort-Object)
    $orphanMasks = @($maskStems.Keys | Where-Object { -not $anomalyStems.Contains($_) } | Sort-Object)
    $emptyFiles = @($allFiles | Where-Object { $_.Length -le 0 } | ForEach-Object { Get-RelativePath -BasePath $datasetPath -Path $_.FullName })

    $duplicateGroups = @()
    $hashRows = @()
    foreach ($file in $imageFiles) {
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $hashRows += [ordered]@{
            hash = $hash
            path = Get-RelativePath -BasePath $datasetPath -Path $file.FullName
        }
    }
    foreach ($group in ($hashRows | Group-Object -Property hash | Where-Object { $_.Count -gt 1 })) {
        $duplicateGroups += [ordered]@{
            hash = $group.Name
            paths = @($group.Group | ForEach-Object { $_.path })
        }
    }

    $validationIssues = @()
    if ($trainGood.Count -le 0) {
        $validationIssues += [ordered]@{
            severity = "error"
            code = "train_good_missing"
            message = "anomaly_folder requires train/good images."
        }
    }
    $evaluationState = if ($anomalyImages.Count -gt 0) { "ready" } else { "limited" }
    if ($evaluationState -eq "limited") {
        $validationIssues += [ordered]@{
            severity = "warning"
            code = "good_only_limited_evaluation"
            message = "Only good samples were found; training is allowed but evaluation is limited."
        }
    }
    $validationOk = $trainGood.Count -gt 0
    $validationReportPath = Join-Path $preflightRoot "dataset_validation_report.json"
    $validation = [ordered]@{
        kind = "dataset_validation_report"
        source = "phase-anomaly-mvtec-quality-matrix"
        ok = $validationOk
        status = if ($validationOk) { "passed" } else { "failed" }
        format = "anomaly_folder"
        datasetPath = $datasetPath
        category = $Category
        checkedAt = (Get-Date).ToUniversalTime().ToString("o")
        counts = [ordered]@{
            normalImages = $normalImages.Count
            anomalyImages = $anomalyImages.Count
            masks = $maskFiles.Count
            trainGood = $trainGood.Count
        }
        evaluationState = $evaluationState
        issues = $validationIssues
        reportPath = $validationReportPath
    }
    Write-JsonFile -Path $validationReportPath -Value $validation

    $qualityIssues = @()
    foreach ($path in $emptyFiles) {
        $qualityIssues += [ordered]@{ severity = "error"; code = "empty_file"; path = $path; message = "File is empty." }
    }
    foreach ($stem in $missingMasks) {
        $qualityIssues += [ordered]@{ severity = "warning"; code = "mask_missing"; stem = $stem; message = "Anomaly image has no matching pixel mask." }
    }
    foreach ($stem in $orphanMasks) {
        $qualityIssues += [ordered]@{ severity = "warning"; code = "mask_orphan"; stem = $stem; message = "Mask has no matching anomaly image stem." }
    }
    foreach ($group in $duplicateGroups) {
        $qualityIssues += [ordered]@{ severity = "warning"; code = "duplicate_image"; hash = $group.hash; paths = $group.paths; message = "Duplicate image hash found." }
    }
    $qualityReportPath = Join-Path $preflightRoot "dataset_quality_report.json"
    $quality = [ordered]@{
        kind = "dataset_quality_report"
        source = "phase-anomaly-mvtec-quality-matrix"
        ok = @($qualityIssues | Where-Object { $_.severity -eq "error" }).Count -eq 0
        status = if (@($qualityIssues | Where-Object { $_.severity -eq "error" }).Count -eq 0) { "passed" } else { "failed" }
        format = "anomaly_folder"
        datasetPath = $datasetPath
        category = $Category
        checkedAt = (Get-Date).ToUniversalTime().ToString("o")
        checks = [ordered]@{
            emptyFiles = $emptyFiles.Count
            duplicateGroups = $duplicateGroups.Count
            missingMasks = $missingMasks.Count
            orphanMasks = $orphanMasks.Count
        }
        issues = $qualityIssues
        reportPath = $qualityReportPath
    }
    Write-JsonFile -Path $qualityReportPath -Value $quality

    $hashInput = ($hashRows | Sort-Object -Property path | ForEach-Object { "$($_.path)|$($_.hash)" }) -join "`n"
    $snapshotHash = New-StringSha256 -Text $hashInput
    $snapshotReportPath = Join-Path $preflightRoot "dataset_snapshot_manifest.json"
    $snapshot = [ordered]@{
        kind = "dataset_snapshot"
        source = "phase-anomaly-mvtec-quality-matrix"
        ok = $true
        format = "anomaly_folder"
        datasetPath = $datasetPath
        category = $Category
        createdAt = (Get-Date).ToUniversalTime().ToString("o")
        fileCount = $allFiles.Count
        imageCount = $imageFiles.Count
        totalBytes = @($allFiles | Measure-Object -Property Length -Sum)[0].Sum
        contentHash = $snapshotHash
        roleCounts = $roleCounts
        splitCounts = $splitCounts
        files = $items
        reportPath = $snapshotReportPath
    }
    Write-JsonFile -Path $snapshotReportPath -Value $snapshot

    return [ordered]@{
        category = $Category
        ok = ($validation.ok -and $quality.ok)
        validationReportPath = $validationReportPath
        qualityReportPath = $qualityReportPath
        snapshotManifestPath = $snapshotReportPath
        contentHash = $snapshotHash
        counts = $validation.counts
        evaluationState = $evaluationState
    }
}

function Get-SampleAnomalyImage {
    param([string]$DatasetPath)
    $extensions = @("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    $canonical = Join-Path $DatasetPath "test\anomaly"
    if (Test-Path -LiteralPath $canonical) {
        $image = Get-ChildItem -LiteralPath $canonical -Recurse -File -Include $extensions | Select-Object -First 1
        if ($image) { return $image.FullName }
    }
    $testRoot = Join-Path $DatasetPath "test"
    if (Test-Path -LiteralPath $testRoot) {
        foreach ($dir in Get-ChildItem -LiteralPath $testRoot -Directory | Where-Object { $_.Name -notin @("good", "anomaly") }) {
            $image = Get-ChildItem -LiteralPath $dir.FullName -Recurse -File -Include $extensions | Select-Object -First 1
            if ($image) { return $image.FullName }
        }
    }
    return ""
}

function Invoke-AnomalyAdapter {
    param(
        [string]$Mode,
        [string]$Backend,
        [string]$Category,
        [string]$DatasetPath,
        [string]$OutputPath,
        [string]$ModelPath,
        [string]$ImagePath,
        [int]$RowBatchSize
    )
    New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
    $parameters = [ordered]@{
        trainingBackend = $Backend
        modelPreset = if ($Backend -eq "anomalib_efficientad") { "anomalib_efficientad_s" } else { "anomalib_patchcore_wide_resnet50_2" }
        epochs = $Epochs
        batchSize = $RowBatchSize
        imageSize = $ImageSize
        workers = $Workers
        device = $Device
        thresholdStrategy = "quantile"
        quantile = 0.995
    }
    if ($Backend -eq "anomalib_efficientad") {
        $parameters["modelSize"] = "small"
        $parameters["imagenetDir"] = $script:ImagenetteFull
    }
    $request = [ordered]@{
        protocolVersion = 1
        mode = $Mode
        taskId = "anomaly-mvtec-$Category-$Backend-$Mode"
        taskType = "anomaly_detection"
        datasetFormat = "anomaly_folder"
        backend = $Backend
        datasetPath = $DatasetPath
        outputPath = $OutputPath
        parameters = $parameters
        modelPath = $ModelPath
        imagePath = $ImagePath
        options = [ordered]@{
            runtime = "anomalib_python"
            iterations = $BenchmarkIterations
            warmupIterations = $BenchmarkWarmup
            device = $Device
        }
    }
    $requestPath = Join-Path $OutputPath "request.json"
    Write-JsonFile -Path $requestPath -Value $request
    $adapter = Join-Path $script:Root "python_trainers\anomaly\anomalib_adapter.py"
    $logPath = Join-Path $OutputPath "anomalib_$Mode.log"
    $env = @{
        AITRAIN_ANOMALIB_IMAGENET_DIR = $script:ImagenetteFull
        PYTHONUTF8 = "1"
        PYTHONIOENCODING = "utf-8"
    }
    $result = Invoke-Logged `
        -Name "$Category-$Backend-$Mode" `
        -FilePath $script:PythonFull `
        -Arguments @("-u", $adapter, "--request", $requestPath, "--mode", $Mode) `
        -LogPath $logPath `
        -Environment $env
    return [ordered]@{
        mode = $Mode
        backend = $Backend
        category = $Category
        outputPath = $OutputPath
        requestPath = $requestPath
        logPath = $logPath
        exitCode = $result.exitCode
        elapsedSeconds = $result.elapsedSeconds
        batchSize = $RowBatchSize
    }
}

function Test-ArtifactExists {
    param([string]$Path)
    return -not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)
}

function Invoke-MatrixRow {
    param([string]$Category, [string]$Backend)
    $datasetPath = Join-Path $script:MvtecRootFull $Category
    $rowRoot = Join-Path $script:WorkFull "$Category\$Backend"
    $effectiveBatchSize = if ($Backend -eq "anomalib_efficientad") { 1 } else { $BatchSize }
    if (Test-Path -LiteralPath $rowRoot) {
        Remove-Item -LiteralPath $rowRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $rowRoot | Out-Null
    $sample = Get-SampleAnomalyImage -DatasetPath $datasetPath
    $row = [ordered]@{
        category = $Category
        backend = $Backend
        status = "failed"
        ok = $false
        datasetPath = $datasetPath
        sampleImagePath = $sample
        batchSize = $effectiveBatchSize
        stages = @()
        artifacts = [ordered]@{}
        metrics = [ordered]@{}
        errorCode = ""
        message = ""
    }
    if ([string]::IsNullOrWhiteSpace($sample)) {
        $row.status = "blocked"
        $row.errorCode = "sample_image_missing"
        $row.message = "No anomaly test image was found for category $Category."
        return $row
    }

    $train = Invoke-AnomalyAdapter -Mode "train" -Backend $Backend -Category $Category -DatasetPath $datasetPath -OutputPath (Join-Path $rowRoot "train") -ModelPath (Join-Path $rowRoot "train\anomaly_sidecar.json") -ImagePath $sample -RowBatchSize $effectiveBatchSize
    $row.stages += $train
    if ($train.exitCode -ne 0 -and $effectiveBatchSize -gt 1 -and (Test-LogLooksLikeOom $train.logPath)) {
        $row.batchSize = 1
        $retry = Invoke-AnomalyAdapter -Mode "train" -Backend $Backend -Category $Category -DatasetPath $datasetPath -OutputPath (Join-Path $rowRoot "train_retry_batch1") -ModelPath (Join-Path $rowRoot "train_retry_batch1\anomaly_sidecar.json") -ImagePath $sample -RowBatchSize 1
        $row.stages += $retry
        $train = $retry
    }
    if ($train.exitCode -ne 0) {
        $row.status = "failed"
        $row.errorCode = "train_failed"
        $row.message = "Training failed. Inspect $($train.logPath)"
        return $row
    }

    $sidecar = Join-Path $train.outputPath "anomaly_sidecar.json"
    $trainingReport = Join-Path $train.outputPath "anomalib_training_report.json"
    $row.artifacts.trainingReport = $trainingReport
    $row.artifacts.sidecar = $sidecar
    if (-not (Test-ArtifactExists $sidecar)) {
        $row.status = "failed"
        $row.errorCode = "sidecar_missing"
        $row.message = "Training completed but anomaly_sidecar.json is missing."
        return $row
    }

    $eval = Invoke-AnomalyAdapter -Mode "evaluate" -Backend $Backend -Category $Category -DatasetPath $datasetPath -OutputPath (Join-Path $rowRoot "evaluate") -ModelPath $sidecar -ImagePath $sample -RowBatchSize ([int]$row.batchSize)
    $row.stages += $eval
    $infer = Invoke-AnomalyAdapter -Mode "infer" -Backend $Backend -Category $Category -DatasetPath $datasetPath -OutputPath (Join-Path $rowRoot "infer") -ModelPath $sidecar -ImagePath $sample -RowBatchSize ([int]$row.batchSize)
    $row.stages += $infer
    $bench = Invoke-AnomalyAdapter -Mode "benchmark" -Backend $Backend -Category $Category -DatasetPath $datasetPath -OutputPath (Join-Path $rowRoot "benchmark") -ModelPath $sidecar -ImagePath $sample -RowBatchSize ([int]$row.batchSize)
    $row.stages += $bench

    $evaluationReport = Join-Path $eval.outputPath "evaluation_report.json"
    $predictions = Join-Path $infer.outputPath "inference_predictions.json"
    $benchmarkReport = Join-Path $bench.outputPath "benchmark_report.json"
    $row.artifacts.evaluationReport = $evaluationReport
    $row.artifacts.predictions = $predictions
    $row.artifacts.benchmarkReport = $benchmarkReport
    $row.artifacts.heatmap = Join-Path $infer.outputPath "anomaly_heatmap.png"
    $row.artifacts.overlay = Join-Path $infer.outputPath "anomaly_overlay.png"
    $row.artifacts.mask = Join-Path $infer.outputPath "anomaly_mask.png"

    $required = @($trainingReport, $sidecar, $evaluationReport, $predictions, $benchmarkReport, $row.artifacts.heatmap, $row.artifacts.overlay, $row.artifacts.mask)
    $missing = @($required | Where-Object { -not (Test-ArtifactExists $_) })
    $failedStages = @($row.stages | Where-Object { $_.exitCode -ne 0 })
    if ($failedStages.Count -gt 0 -or $missing.Count -gt 0) {
        $row.status = "failed"
        $row.errorCode = if ($failedStages.Count -gt 0) { "stage_failed" } else { "artifact_missing" }
        $row.message = if ($failedStages.Count -gt 0) { "One or more lifecycle stages failed." } else { "One or more required artifacts are missing." }
        $row.missingArtifacts = $missing
        return $row
    }

    $evalJson = Read-JsonObject $evaluationReport
    $benchJson = Read-JsonObject $benchmarkReport
    if ($evalJson -and $evalJson.metrics) {
        foreach ($name in @("imageAUROC", "imageF1", "pixelAUROC", "pixelF1", "threshold")) {
            if ($evalJson.metrics.PSObject.Properties.Name -contains $name) {
                $row.metrics[$name] = $evalJson.metrics.$name
            }
        }
    }
    if ($benchJson) {
        foreach ($name in @("averageMs", "p50Ms", "p95Ms", "p99Ms", "throughput")) {
            if ($benchJson.PSObject.Properties.Name -contains $name) {
                $row.metrics[$name] = $benchJson.$name
            }
        }
    }
    $row.ok = $true
    $row.status = "passed"
    $row.message = "Lifecycle completed."
    return $row
}

function Get-MapValue {
    param([object]$Map, [string]$Name)
    if ($null -eq $Map) {
        return ""
    }
    if ($Map -is [System.Collections.IDictionary]) {
        if ($Map.Contains($Name)) {
            return $Map[$Name]
        }
        return ""
    }
    if ($Map.PSObject.Properties.Name -contains $Name) {
        return $Map.$Name
    }
    return ""
}

function Write-MatrixReports {
    $summaryPath = Join-Path $script:WorkFull "anomaly_mvtec_quality_matrix_summary.json"
    $csvPath = Join-Path $script:WorkFull "anomaly_mvtec_quality_matrix_summary.csv"
    $mdPath = Join-Path $script:WorkFull "anomaly_mvtec_quality_matrix_summary.md"
    Write-JsonFile -Path $summaryPath -Value $script:Summary
    $rows = foreach ($row in $script:Summary.rows) {
        [pscustomobject]@{
            category = $row.category
            backend = $row.backend
            status = $row.status
            batchSize = $row.batchSize
            imageAUROC = (Get-MapValue -Map $row.metrics -Name "imageAUROC")
            imageF1 = (Get-MapValue -Map $row.metrics -Name "imageF1")
            pixelAUROC = (Get-MapValue -Map $row.metrics -Name "pixelAUROC")
            pixelF1 = (Get-MapValue -Map $row.metrics -Name "pixelF1")
            averageMs = (Get-MapValue -Map $row.metrics -Name "averageMs")
            p95Ms = (Get-MapValue -Map $row.metrics -Name "p95Ms")
            throughput = (Get-MapValue -Map $row.metrics -Name "throughput")
            message = $row.message
        }
    }
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csvPath
    $md = New-Object System.Collections.Generic.List[string]
    $md.Add("# Anomalib MVTec Quality Matrix")
    $md.Add("")
    $md.Add("- Status: $($script:Summary.status)")
    $md.Add("- Runtime: anomalib_python")
    $md.Add("- Python: $script:PythonFull")
    $md.Add("- Work dir: $script:WorkFull")
    $md.Add("")
    $md.Add("| Category | Backend | Status | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Avg ms | P95 ms | Throughput |")
    $md.Add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    foreach ($row in $script:Summary.rows) {
        $imageAUROC = Get-MapValue -Map $row.metrics -Name "imageAUROC"
        $imageF1 = Get-MapValue -Map $row.metrics -Name "imageF1"
        $pixelAUROC = Get-MapValue -Map $row.metrics -Name "pixelAUROC"
        $pixelF1 = Get-MapValue -Map $row.metrics -Name "pixelF1"
        $averageMs = Get-MapValue -Map $row.metrics -Name "averageMs"
        $p95Ms = Get-MapValue -Map $row.metrics -Name "p95Ms"
        $throughput = Get-MapValue -Map $row.metrics -Name "throughput"
        $md.Add("| $($row.category) | $($row.backend) | $($row.status) | $imageAUROC | $imageF1 | $pixelAUROC | $pixelF1 | $averageMs | $p95Ms | $throughput |")
    }
    $md.Add("")
    $md.Add("This is public MVTec workflow evidence only. It is not customer-domain production precision evidence.")
    $md -join "`n" | Set-Content -Encoding UTF8 -LiteralPath $mdPath
    $script:Summary.summaryPath = $summaryPath
    $script:Summary.csvPath = $csvPath
    $script:Summary.markdownPath = $mdPath
    Write-JsonFile -Path $summaryPath -Value $script:Summary
}

$script:WorkFull = Resolve-RepoPath $WorkDir
$script:MvtecRootFull = Resolve-RepoPath $MvtecRoot
$script:MvtecArchiveFull = Resolve-RepoPath $MvtecArchivePath
$script:ImagenetteFull = Resolve-RepoPath $ImagenetteDir
$envFull = Resolve-RepoPath $EnvPath
if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $envFull "python.exe"
}
$script:PythonFull = Resolve-Executable $PythonExecutable

Assert-UnderDeps -Path $script:WorkFull -Description "work directory"
Assert-UnderDeps -Path $envFull -Description "Anomalib env"
Assert-UnderDeps -Path $script:MvtecRootFull -Description "MVTec materialized root"
Assert-UnderDeps -Path $script:MvtecArchiveFull -Description "MVTec archive"
Assert-UnderDeps -Path $script:ImagenetteFull -Description "Imagenette directory"

if ((Test-Path -LiteralPath $script:WorkFull) -and $Force) {
    $leaf = Split-Path -Leaf $script:WorkFull
    if ($leaf -ne "anomaly-mvtec-quality-matrix") {
        throw "Refusing to remove unexpected work dir: $script:WorkFull"
    }
    Remove-Item -LiteralPath $script:WorkFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $script:WorkFull | Out-Null
$script:Summary.workDir = $script:WorkFull
$script:Summary.python = $script:PythonFull
$script:Summary.mvtecRoot = $script:MvtecRootFull
$script:Summary.imagenetteDir = $script:ImagenetteFull

try {
    Write-Host "matrix.stage=ensure_mvtec_categories"
    Ensure-MvtecCategories -CategoryList $Categories
    foreach ($category in $Categories) {
        Write-Host "matrix.stage=preflight category=$category"
        $preflight = Invoke-CategoryPreflight -Category $category
        $script:Summary.categoryPreflight += $preflight
        Write-MatrixReports
        if (-not $preflight.ok) {
            throw "Dataset preflight failed for MVTec category $category."
        }
    }
    Write-Host "matrix.stage=ensure_conda_environment"
    Ensure-CondaEnvironment -PythonPath $script:PythonFull -EnvironmentPath $envFull
    Write-Host "matrix.stage=ensure_python_packages"
    Ensure-PythonPackages -PythonPath $script:PythonFull
    Write-Host "matrix.stage=test_anomalib_environment"
    Test-AnomalibEnvironment -PythonPath $script:PythonFull
    if ($Backends -contains "anomalib_efficientad") {
        Write-Host "matrix.stage=ensure_imagenette"
        Ensure-Imagenette -TargetDir $script:ImagenetteFull
    }

    foreach ($category in $Categories) {
        foreach ($backend in $Backends) {
            Write-Host "matrix.stage=row category=$category backend=$backend"
            $row = Invoke-MatrixRow -Category $category -Backend $backend
            $script:Summary.rows += $row
            Write-MatrixReports
        }
    }
    $failed = @($script:Summary.rows | Where-Object { $_.status -ne "passed" })
    if ($failed.Count -eq 0 -and $script:Summary.rows.Count -gt 0) {
        $script:Summary.ok = $true
        $script:Summary.status = "passed"
    } elseif ($script:Summary.rows.Count -eq 0) {
        $script:Summary.status = "blocked"
        $script:Summary.message = "No Anomalib MVTec matrix rows were executed."
    } else {
        $script:Summary.status = "failed"
        $script:Summary.message = "One or more Anomalib MVTec matrix rows failed."
    }
} catch {
    $script:Summary.ok = $false
    if (-not $script:Summary.status -or $script:Summary.status -eq "passed") {
        $script:Summary.status = "blocked"
    }
    $script:Summary.message = if ([string]::IsNullOrWhiteSpace($_.Exception.Message)) { $_.ToString() } else { $_.Exception.Message }
} finally {
    Write-MatrixReports
}

Write-Host "Anomalib MVTec quality matrix summary: $($script:Summary.summaryPath)"
if ($script:Summary.ok) {
    exit 0
}
exit 1
