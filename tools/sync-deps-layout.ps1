param(
    [switch]$NoJunctions
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
. (Join-Path $PSScriptRoot "deps-layout.ps1")

$layout = Get-AITrainDepsLayout -Root $root
foreach ($dir in @($layout.DepsRoot, $layout.EnvRoot, $layout.ReposRoot, $layout.SdkRoot, $layout.ArchivesRoot, $layout.ToolsRoot, $layout.UiWalkthroughRoot)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function New-CompatibleDirectoryAlias {
    param(
        [string]$CanonicalPath,
        [string[]]$LegacyCandidates,
        [string]$Description
    )

    $target = Resolve-AITrainFirstExistingPath -Candidates $LegacyCandidates
    $record = [ordered]@{
        description = $Description
        canonicalPath = $CanonicalPath
        targetPath = $target
        status = "missing"
        aliasType = ""
    }

    if (Test-Path -LiteralPath $CanonicalPath) {
        $record.status = "canonical_exists"
        $record.aliasType = (Get-Item -LiteralPath $CanonicalPath -Force).Attributes.ToString()
        return $record
    }
    if ([string]::IsNullOrWhiteSpace($target)) {
        return $record
    }
    if ($NoJunctions) {
        $record.status = "target_exists_no_alias"
        return $record
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CanonicalPath) | Out-Null
    New-Item -ItemType Junction -Path $CanonicalPath -Target $target | Out-Null
    $record.status = "junction_created"
    $record.aliasType = "Junction"
    return $record
}

$aliases = @()
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.OcrGpuEnv `
    -LegacyCandidates @((Join-Path $root ".deps\rtx4090-validation\python-ocr-gpu")) `
    -Description "CUDA PaddleOCR Python environment"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.OcrCpuEnv `
    -LegacyCandidates @((Join-Path $root ".deps\python-3.13.13-ocr-amd64"), (Join-Path $root ".deps\rtx4090-validation\python-ocr")) `
    -Description "CPU/portable PaddleOCR Python environment"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.PythonEmbed `
    -LegacyCandidates @((Join-Path $root ".deps\python-3.13.13-embed-amd64")) `
    -Description "portable embedded Python runtime"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.YoloCudaEnv `
    -LegacyCandidates @((Join-Path $root ".deps\rtx4090-validation\python-yolo-cuda"), (Join-Path $root ".deps\rtx4090-validation\python-yolo-venv")) `
    -Description "CUDA Ultralytics YOLO Python environment"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.PaddleOcrRepo `
    -LegacyCandidates @((Join-Path $root ".deps\PaddleOCR")) `
    -Description "PaddleOCR source checkout"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.NcnnRoot `
    -LegacyCandidates @((Join-Path $root ".deps\ncnn")) `
    -Description "NCNN SDK/runtime"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.OnnxRuntimeRoot `
    -LegacyCandidates @((Join-Path $root ".deps\onnxruntime-win-x64-1.24.3"), (Join-Path $root ".deps\onnxruntime")) `
    -Description "ONNX Runtime SDK/runtime"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.TensorRtOssRoot `
    -LegacyCandidates @((Join-Path $root ".deps\tensorrt-oss")) `
    -Description "TensorRT headers/source support"
$aliases += New-CompatibleDirectoryAlias `
    -CanonicalPath $layout.TensorRtRuntimeRoot `
    -LegacyCandidates @(
        (Join-Path $root ".deps\tensorrt-pypi\extracted\tensorrt_cu13_libs-10.16.1.11-py3-none-win_amd64\tensorrt_libs"),
        (Join-Path $root ".deps\rtx4090-validation\python-yolo-venv\Lib\site-packages\tensorrt_libs"),
        (Join-Path $root "build-vscode\bin\runtimes\tensorrt")
    ) `
    -Description "TensorRT runtime DLLs"

$archives = @(
    @{ canonical = Join-Path $layout.ArchivesRoot "python-3.13.13-embed-amd64.zip"; legacy = Join-Path $root ".deps\python-3.13.13-embed-amd64.zip"; description = "portable Python zip" },
    @{ canonical = Join-Path $layout.ArchivesRoot "get-pip.py"; legacy = Join-Path $root ".deps\get-pip.py"; description = "get-pip bootstrap script" }
)
foreach ($archive in $archives) {
    if (!(Test-Path -LiteralPath $archive.canonical) -and (Test-Path -LiteralPath $archive.legacy) -and !$NoJunctions) {
        New-Item -ItemType HardLink -Path $archive.canonical -Target $archive.legacy -ErrorAction SilentlyContinue | Out-Null
        if (!(Test-Path -LiteralPath $archive.canonical)) {
            Copy-Item -LiteralPath $archive.legacy -Destination $archive.canonical -Force
        }
    }
    $aliases += [ordered]@{
        description = $archive.description
        canonicalPath = $archive.canonical
        targetPath = $archive.legacy
        status = if (Test-Path -LiteralPath $archive.canonical) { "canonical_exists" } elseif (Test-Path -LiteralPath $archive.legacy) { "legacy_only" } else { "missing" }
        aliasType = "file"
    }
}

$manifest = [ordered]@{
    generatedAt = [DateTime]::UtcNow.ToString("o")
    repoRoot = $root
    layout = $layout
    aliases = $aliases
    note = "Canonical reusable environments are under .deps\\envs, source checkouts under .deps\\repos, SDK/runtime dependencies under .deps\\sdks, and UI walkthrough evidence under .deps\\UI-Walkthrough. Legacy validation directories remain evidence locations and are used only as compatibility targets."
}
$manifestPath = Join-Path $layout.DepsRoot "deps-layout-manifest.json"
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "AITrain .deps layout synchronized: $manifestPath" -ForegroundColor Green
foreach ($alias in $aliases) {
    Write-Host ("  {0}: {1}" -f $alias.description, $alias.status)
}
