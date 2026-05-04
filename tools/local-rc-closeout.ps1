param(
    [switch]$RunLocalBaseline,
    [switch]$RunCpuTrainingSmoke,
    [switch]$SkipHarness,
    [switch]$SkipPackageSmoke,
    [string]$BuildDir = "build-vscode"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
Set-Location $root

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host "Local RC closeout: $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-PowerShellScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments = @()
    )

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$ScriptPath failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step "git diff whitespace check" {
    git diff --check
}

if (-not $SkipHarness) {
    Invoke-Step "harness check" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\harness-check.ps1")
    }
}

if (-not $SkipPackageSmoke) {
    Invoke-Step "package smoke" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\package-smoke.ps1") -Arguments @("-BuildDir", $BuildDir, "-SkipBuild")
    }
}

if ($RunLocalBaseline) {
    Invoke-Step "local baseline acceptance" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\acceptance-smoke.ps1") -Arguments @("-LocalBaseline", "-Package", "-SkipBuild", "-BuildDir", $BuildDir)
    }
}

if ($RunCpuTrainingSmoke) {
    Invoke-Step "CPU training smoke" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\acceptance-smoke.ps1") -Arguments @("-CpuTrainingSmoke", "-SkipOfficialOcr", "-BuildDir", $BuildDir)
    }
}

Write-Host "Local RC closeout passed." -ForegroundColor Green
