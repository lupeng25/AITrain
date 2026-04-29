Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "AITrain Studio Harness Context" -ForegroundColor Cyan
Write-Host "Root: $root"
Write-Host ""

Write-Host "Key files:" -ForegroundColor Cyan
$files = @(
    "HARNESS.md",
    "README.md",
    "src/core/include/aitrain/core/PluginInterfaces.h",
    "src/core/include/aitrain/core/JsonProtocol.h",
    "src/core/include/aitrain/core/ProjectRepository.h",
    "src/app/src/MainWindow.cpp",
    "src/worker/src/WorkerSession.cpp",
    "tests/tst_core.cpp"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  [ok] $file"
    } else {
        Write-Host "  [missing] $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Source summary:" -ForegroundColor Cyan
Get-ChildItem src -Recurse -File -Include *.cpp,*.h,*.json,*.txt,*.cmake,CMakeLists.txt |
    Where-Object { $_.FullName -notmatch "\\build" } |
    Group-Object { Split-Path (Resolve-Path -Relative $_.FullName) -Parent } |
    Sort-Object Name |
    ForEach-Object {
        Write-Host ("  {0}: {1} files" -f $_.Name, $_.Count)
    }

Write-Host ""
Write-Host "Recommended check:" -ForegroundColor Cyan
Write-Host "  .\tools\harness-check.ps1"
