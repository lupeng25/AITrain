param(
    [string]$BuildDir = "build-vscode",
    [string]$SourceDir = "",
    [string]$OutputDir = "",
    [string]$InnoCompiler = "",
    [string]$AppVersion = "",
    [string]$Compression = "lzma/normal",
    [switch]$SkipPackageSmoke,
    [switch]$ExcludeTensorRt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Find-InnoCompiler {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $full = Resolve-RepoPath $RequestedPath
        if (-not (Test-Path $full)) {
            throw "Inno Setup compiler not found: $full"
        }
        return $full
    }

    $fromPath = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    throw "Inno Setup 6 compiler was not found. Install Inno Setup 6 or pass -InnoCompiler <path-to-ISCC.exe>."
}

function Read-CMakeProjectVersion {
    $cmakePath = Join-Path $root "CMakeLists.txt"
    $cmakeText = Get-Content -LiteralPath $cmakePath -Raw
    $match = [regex]::Match($cmakeText, 'project\s*\(\s*AITrainStudio\s+VERSION\s+([0-9]+(?:\.[0-9]+){1,3})', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        throw "Unable to read AITrainStudio project version from CMakeLists.txt"
    }
    return $match.Groups[1].Value
}

Set-Location $root

if (-not $AppVersion) {
    $AppVersion = Read-CMakeProjectVersion
}

if (-not $SourceDir) {
    $SourceDir = Join-Path $BuildDir "package-smoke"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $BuildDir "inno"
}

$sourceFull = Resolve-RepoPath $SourceDir
$outputFull = Resolve-RepoPath $OutputDir
$issPath = Resolve-RepoPath "installer\AITrainStudio.iss"
$packageExcludes = "*.pdb,*.ilk,*.exp,*.lib,installer\*,tools\build-inno-installer.ps1"
if ($ExcludeTensorRt) {
    $packageExcludes = "$packageExcludes,runtimes\tensorrt\*"
}

if (-not $SkipPackageSmoke) {
    Write-Host "Inno installer: refresh package-smoke layout" -ForegroundColor Cyan
    & (Join-Path $root "tools\package-smoke.ps1") -BuildDir $BuildDir
    if ($LASTEXITCODE -ne 0) {
        throw "package-smoke.ps1 failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $sourceFull)) {
    throw "Installer source layout not found: $sourceFull"
}
if (-not (Test-Path (Join-Path $sourceFull "AITrainStudio.exe"))) {
    throw "Installer source layout is missing AITrainStudio.exe: $sourceFull"
}
if (-not (Test-Path (Join-Path $sourceFull "aitrain_worker.exe"))) {
    throw "Installer source layout is missing aitrain_worker.exe: $sourceFull"
}
if (-not (Test-Path $issPath)) {
    throw "Inno script not found: $issPath"
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null

$iscc = Find-InnoCompiler -RequestedPath $InnoCompiler

Write-Host "Inno installer: compile" -ForegroundColor Cyan
Write-Host "  SourceDir: $sourceFull"
Write-Host "  OutputDir: $outputFull"
Write-Host "  Version:   $AppVersion"
Write-Host "  Excludes:  $packageExcludes"

& $iscc "/DSourceDir=$sourceFull" "/DOutputDir=$outputFull" "/DAppVersion=$AppVersion" "/DPackageExcludes=$packageExcludes" "/DInstallerCompression=$Compression" "/DInstallerSolidCompression=no" $issPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compiler failed with exit code $LASTEXITCODE"
}

$installerPath = Join-Path $outputFull "AITrainStudio-$AppVersion-Setup.exe"
if (-not (Test-Path $installerPath)) {
    throw "Expected installer was not created: $installerPath"
}

$hash = Get-FileHash -LiteralPath $installerPath -Algorithm SHA256
Write-Host "Inno installer created: $installerPath" -ForegroundColor Green
Write-Host "SHA256: $($hash.Hash)" -ForegroundColor Green
