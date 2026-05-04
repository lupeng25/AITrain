param(
    [string]$BuildDir = "build-vscode",
    [switch]$SkipLocalRc,
    [switch]$SkipPackageBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$qt = "C:\Qt\Qt5.12.9\5.12.9\msvc2015_64"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )

    Write-Host "Release freeze: $Name" -ForegroundColor Cyan
    & $Body
}

function Invoke-CommandLine {
    param([string]$CommandLine)

    cmd /c $CommandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $CommandLine"
    }
}

function Resolve-RepoPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

Set-Location $root

if (-not (Test-Path $vcvars)) {
    throw "MSVC environment script not found: $vcvars"
}
if (-not (Test-Path $qt)) {
    throw "Qt kit not found: $qt"
}

if (-not $SkipLocalRc) {
    Invoke-Step "local RC closeout" {
        & (Join-Path $root "tools\local-rc-closeout.ps1") -BuildDir $BuildDir
        if ($LASTEXITCODE -ne 0) {
            throw "local-rc-closeout.ps1 failed with exit code $LASTEXITCODE"
        }
    }
}

if (-not $SkipPackageBuild) {
    Invoke-Step "configure package build" {
        $configure = "call `"$vcvars`" >nul && cmake -S . -B `"$BuildDir`" -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=`"$qt`" -DAITRAIN_BUILD_TESTS=ON"
        Invoke-CommandLine $configure
    }

    Invoke-Step "build package inputs" {
        $build = "call `"$vcvars`" >nul && cmake --build `"$BuildDir`""
        Invoke-CommandLine $build
    }
}

Invoke-Step "generate CPack ZIP" {
    $cpack = "call `"$vcvars`" >nul && cpack --config `"$BuildDir\CPackConfig.cmake`" -B `"$BuildDir`""
    Invoke-CommandLine $cpack
}

$buildPath = Resolve-RepoPath $BuildDir
$handoffDir = Join-Path $buildPath "release-freeze-handoff"
New-Item -ItemType Directory -Force -Path $handoffDir | Out-Null

$zipFiles = @(
    Get-ChildItem -LiteralPath $buildPath -Filter "AITrainStudio-*-win64.zip" -File -ErrorAction SilentlyContinue
)
if ($zipFiles.Count -eq 0) {
    throw "No AITrainStudio ZIP package found under $buildPath"
}

$commit = (& git rev-parse HEAD).Trim()
$statusShort = @(& git status --short)
$packageEntries = @()
foreach ($zip in $zipFiles | Sort-Object LastWriteTime -Descending) {
    $hash = Get-FileHash -LiteralPath $zip.FullName -Algorithm SHA256
    $packageEntries += [ordered]@{
        path = $zip.FullName
        name = $zip.Name
        bytes = $zip.Length
        sha256 = $hash.Hash
        lastWriteTimeUtc = $zip.LastWriteTimeUtc.ToString("o")
    }
}

$handoffDocs = @(
    "docs\external-acceptance-handoff.md",
    "docs\acceptance-templates\clean-windows-acceptance-result.md",
    "docs\acceptance-templates\tensorrt-acceptance-result.md"
)
foreach ($doc in $handoffDocs) {
    $full = Resolve-RepoPath $doc
    if (-not (Test-Path $full)) {
        throw "Missing handoff doc: $doc"
    }
}

$manifest = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    sourceCommit = $commit
    worktreeDirty = ($statusShort.Count -gt 0)
    worktreeStatus = $statusShort
    buildDir = $buildPath
    packages = $packageEntries
    handoffDocs = $handoffDocs
    externalCommands = [ordered]@{
        cleanWindowsPackage = ".\tools\acceptance-smoke.ps1 -Package"
        tensorrt = ".\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt"
    }
    requiredReturnArtifacts = @(
        "filled clean Windows result template",
        "filled TensorRT result template when RTX / SM 75+ hardware is available",
        "acceptance_summary.json",
        "full console output",
        "Worker self-check JSON",
        "Worker plugin smoke JSON for package mode",
        "package root layout summary",
        "GPU and driver evidence for TensorRT mode"
    )
    limitations = @(
        "This manifest does not mark external acceptance as passed.",
        "GTX 1060 / SM 61 TensorRT remains hardware-blocked.",
        ".deps, generated datasets, downloaded tools, model weights, ONNX smoke outputs, and TensorRT engines are not source-controlled artifacts."
    )
}

$manifestPath = Join-Path $handoffDir "release_handoff_manifest.json"
$summaryPath = Join-Path $handoffDir "release_handoff_summary.md"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$summaryLines = @(
    '# AITrain Studio Release Handoff Summary',
    '',
    "- Generated UTC: $($manifest.generatedAtUtc)",
    "- Source commit: $commit",
    "- Worktree dirty: $($manifest.worktreeDirty)",
    '',
    '## Packages',
    ''
)
foreach ($package in $packageEntries) {
    $summaryLines += ('- `{0}`' -f $package.name)
    $summaryLines += ('  - Bytes: {0}' -f $package.bytes)
    $summaryLines += ('  - SHA256: `{0}`' -f $package.sha256)
    $summaryLines += ('  - Path: `{0}`' -f $package.path)
}
$summaryLines += @(
    '',
    '## External Commands',
    '',
    '```powershell',
    '.\tools\acceptance-smoke.ps1 -Package',
    '.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt',
    '```',
    '',
    'Do not update TensorRT status as passed until RTX / SM 75+ evidence is returned.'
)
$summaryLines | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host "Release freeze handoff manifest: $manifestPath" -ForegroundColor Green
Write-Host "Release freeze handoff summary:  $summaryPath" -ForegroundColor Green
