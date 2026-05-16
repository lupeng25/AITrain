Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AITrainVcVars {
    if ($env:AITRAIN_VCVARS64 -and (Test-Path $env:AITRAIN_VCVARS64)) {
        return [System.IO.Path]::GetFullPath($env:AITRAIN_VCVARS64)
    }

    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
        "D:\Microsoft Visual Studio\18\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $vcvars = Join-Path ([string]$installPath) "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                return [System.IO.Path]::GetFullPath($vcvars)
            }
        }
    }

    throw "MSVC environment script not found. Set AITRAIN_VCVARS64 to vcvars64.bat."
}

function Resolve-AITrainQtRoot {
    if ($env:AITRAIN_QT_ROOT -and (Test-Path $env:AITRAIN_QT_ROOT)) {
        return [System.IO.Path]::GetFullPath($env:AITRAIN_QT_ROOT)
    }

    $candidates = @(
        "C:\Qt\Qt5.12.9\5.12.9\msvc2017_64",
        "D:\Qt\Qt5.12.9\5.12.9\msvc2017_64",
        "C:\Qt\Qt5.12.9\5.12.9\msvc2015_64",
        "D:\Qt\Qt5.12.9\5.12.9\msvc2015_64"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    foreach ($base in @("C:\Qt", "D:\Qt", "E:\Qt")) {
        if (!(Test-Path $base)) {
            continue
        }

        $qtRoot = Get-ChildItem $base -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '5\.12\.9\\msvc20(15|17)_64$' } |
            Sort-Object @{ Expression = { if ($_.Name -eq "msvc2017_64") { 0 } else { 1 } } }, FullName |
            Select-Object -First 1
        if ($qtRoot) {
            return [System.IO.Path]::GetFullPath($qtRoot.FullName)
        }
    }

    throw "Qt kit not found. Set AITRAIN_QT_ROOT to a Qt 5.12 msvc*_64 kit."
}

function Write-AITrainToolchainSelection {
    param(
        [string]$VcVars,
        [string]$QtRoot
    )

    Write-Host ("AITrain toolchain: MSVC environment = {0}" -f $VcVars) -ForegroundColor DarkCyan
    Write-Host ("AITrain toolchain: Qt root = {0}" -f $QtRoot) -ForegroundColor DarkCyan
}

function Get-AITrainBuildCommandPrefix {
    param(
        [string]$VcVars,
        [string]$QtRoot
    )

    if ([string]::IsNullOrWhiteSpace($VcVars)) {
        $VcVars = Resolve-AITrainVcVars
    }
    if ([string]::IsNullOrWhiteSpace($QtRoot)) {
        $QtRoot = Resolve-AITrainQtRoot
    }

    $qtBin = Join-Path $QtRoot "bin"
    $qtPlugins = Join-Path $QtRoot "plugins"
    $qtPlatforms = Join-Path $qtPlugins "platforms"
    return "set `"PATH=$qtBin;%PATH%`" && set `"QT_PLUGIN_PATH=$qtPlugins`" && set `"QT_QPA_PLATFORM_PLUGIN_PATH=$qtPlatforms`" && call `"$VcVars`" >nul"
}

function Set-AITrainQtRuntimeEnvironment {
    $qtRoot = Resolve-AITrainQtRoot
    $qtBin = Join-Path $qtRoot "bin"
    $qtPlugins = Join-Path $qtRoot "plugins"
    $qtPlatforms = Join-Path $qtPlugins "platforms"

    if (Test-Path $qtBin) {
        $pathParts = @($qtBin) + (($env:PATH -split ';') | Where-Object { $_ -and $_ -ne $qtBin })
        $env:PATH = ($pathParts -join ';')
    }
    $env:QT_PLUGIN_PATH = $qtPlugins
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = $qtPlatforms
}
