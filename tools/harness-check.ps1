Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$qt = "C:\Qt\Qt5.12.9\5.12.9\msvc2015_64"
$buildDir = "build-vscode"

if (-not (Test-Path $vcvars)) {
    throw "MSVC environment script not found: $vcvars"
}

if (-not (Test-Path $qt)) {
    throw "Qt kit not found: $qt"
}

Set-Location $root

Write-Host "Harness check: configure" -ForegroundColor Cyan
$configure = "call `"$vcvars`" >nul && cmake -S . -B $buildDir -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=$qt -DAITRAIN_BUILD_TESTS=ON"
cmd /c $configure
if ($LASTEXITCODE -ne 0) {
    throw "Configure failed with exit code $LASTEXITCODE"
}

Write-Host "Harness check: build" -ForegroundColor Cyan
$build = "call `"$vcvars`" >nul && cmake --build $buildDir"
cmd /c $build
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

Write-Host "Harness check: tests" -ForegroundColor Cyan
$test = "call `"$vcvars`" >nul && ctest --test-dir $buildDir --output-on-failure"
cmd /c $test
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE"
}

Write-Host "Harness check passed." -ForegroundColor Green

