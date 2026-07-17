param(
    [switch]$RunLocalBaseline,
    [switch]$RunCpuTrainingSmoke,
    [switch]$SkipHarness,
    [switch]$SkipPackageSmoke,
    [switch]$SkipGuiWalkthrough,
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

function Assert-CloseoutResidueClean {
    $ownedProcessNames = @(
        "AITrainStudio",
        "aitrain_worker",
        "aitrain_platform_tests",
        "aitrain_application_tests",
        "aitrain_delivery_acceptance_ui_tests"
    )
    $running = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $ownedProcessNames -contains $_.ProcessName })
    if ($running.Count -gt 0) {
        throw ("Closeout left owned processes running: {0}" -f (($running | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ", "))
    }

    $stagingRoots = @(
        (Join-Path $root $BuildDir),
        (Join-Path $root (Join-Path $BuildDir "package-smoke"))
    ) | Where-Object { Test-Path -LiteralPath $_ }
    $dirty = @()
    foreach ($scanRoot in $stagingRoots) {
        $dirty += Get-ChildItem -LiteralPath $scanRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @(".staging", ".staging-meta", ".runtime-staging") } |
            Where-Object {
                @(Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue).Count -gt 0
            }
    }
    if ($dirty.Count -gt 0) {
        throw ("Closeout left non-empty staging directories: {0}" -f (($dirty | ForEach-Object FullName) -join "; "))
    }
    Write-Host "  [ok] no owned processes or non-empty staging directories"
}

Invoke-Step "git diff whitespace check" {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & git diff --check 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }
    if ($exitCode -ne 0) {
        exit $exitCode
    }
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

if (-not $SkipGuiWalkthrough) {
    Invoke-Step "GUI 1280x820 walkthrough" {
        $binDir = Join-Path $root (Join-Path $BuildDir "bin")
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\ui-workbench-walkthrough.ps1") -Arguments @(
            "-AppPath", (Join-Path $binDir "AITrainStudio.exe"),
            "-WorkingDirectory", $binDir,
            "-OutDir", (Join-Path $root ".deps\UI-Walkthrough\rc"))
    }
}

if ($RunLocalBaseline) {
    Invoke-Step "local baseline acceptance" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\acceptance-smoke.ps1") -Arguments @("-LocalBaseline", "-Package", "-SkipBuild", "-BuildDir", $BuildDir)
    }
}

if ($RunCpuTrainingSmoke) {
    Invoke-Step "CPU training smoke" {
        Invoke-PowerShellScript -ScriptPath (Join-Path $root "tools\acceptance-smoke.ps1") -Arguments @("-CpuTrainingSmoke", "-BuildDir", $BuildDir)
    }
}

Invoke-Step "residual process and staging scan" {
    Assert-CloseoutResidueClean
}

Write-Host "Local RC closeout passed." -ForegroundColor Green
