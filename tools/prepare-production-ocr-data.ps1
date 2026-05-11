param(
    [string]$WorkDir = ".deps\production-ocr-data",
    [string]$Python = "",
    [int]$Seed = 1337,
    [int]$MaxRecTrain = 2400,
    [int]$MaxRecVal = 300,
    [int]$MaxRecTest = 300,
    [int]$MinimumSystemImages = 100,
    [switch]$SkipDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "tools\prepare_production_ocr_data.py"

function Resolve-Python {
    if ($Python) {
        if ([System.IO.Path]::IsPathRooted($Python)) {
            return [System.IO.Path]::GetFullPath($Python)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $root $Python))
    }
    $candidates = @(
        (Join-Path $root ".deps\python-3.13.13-ocr-amd64\python.exe"),
        (Join-Path $root ".deps\python-3.13.13-embed-amd64\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }
    throw "No Python executable found. Provide -Python or provision the OCR Python environment."
}

$pythonExe = Resolve-Python
$argsList = @(
    $scriptPath,
    "--work-dir", $WorkDir,
    "--seed", "$Seed",
    "--max-rec-train", "$MaxRecTrain",
    "--max-rec-val", "$MaxRecVal",
    "--max-rec-test", "$MaxRecTest",
    "--minimum-system-images", "$MinimumSystemImages"
)
if ($SkipDownload) {
    $argsList += "--skip-download"
}

Write-Host "Preparing production OCR public data with $pythonExe" -ForegroundColor Cyan
& $pythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Production OCR data preparation failed with exit code $LASTEXITCODE"
}
