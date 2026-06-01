Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "toolchain-env.ps1")

$vcvars = Resolve-AITrainVcVars
$qt = Resolve-AITrainQtRoot
$commandPrefix = Get-AITrainBuildCommandPrefix -VcVars $vcvars -QtRoot $qt
$buildDir = if ($env:AITRAIN_BUILD_DIR) { $env:AITRAIN_BUILD_DIR } else { "build-vscode" }
$buildPath = if ([System.IO.Path]::IsPathRooted($buildDir)) {
    [System.IO.Path]::GetFullPath($buildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $buildDir))
}

function Get-CachedCxxCompiler {
    param([string]$CachePath)

    if (-not (Test-Path -LiteralPath $CachePath)) {
        return $null
    }

    $match = Select-String -LiteralPath $CachePath -Pattern '^CMAKE_CXX_COMPILER:[^=]*=(.+)$' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $match) {
        return $null
    }
    return [string]$match.Matches[0].Groups[1].Value
}

function Clear-CMakeConfigureCache {
    param([string]$BuildPath)

    $items = @(
        "CMakeCache.txt",
        "CMakeFiles",
        "cmake_install.cmake",
        "Makefile",
        "CTestTestfile.cmake",
        "DartConfiguration.tcl"
    )

    foreach ($item in $items) {
        $path = Join-Path $BuildPath $item
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

if (-not (Test-Path $vcvars)) {
    throw "MSVC environment script not found: $vcvars"
}

if (-not (Test-Path $qt)) {
    throw "Qt kit not found: $qt"
}

Set-Location $root

Write-Host "Harness check: encoding" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "encoding-check.ps1") -IncludeUntracked

Write-AITrainToolchainSelection -VcVars $vcvars -QtRoot $qt

$cachedCompiler = Get-CachedCxxCompiler -CachePath (Join-Path $buildPath "CMakeCache.txt")
if ($cachedCompiler -and -not (Test-Path -LiteralPath $cachedCompiler)) {
    Write-Host "Harness check: cached compiler no longer exists; clearing CMake configure cache." -ForegroundColor Yellow
    Write-Host ("Harness check: stale compiler = {0}" -f $cachedCompiler) -ForegroundColor Yellow
    Clear-CMakeConfigureCache -BuildPath $buildPath
}

Write-Host "Harness check: configure" -ForegroundColor Cyan
$configure = "$commandPrefix && cmake -S . -B $buildDir -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=$qt -DAITRAIN_BUILD_TESTS=ON"
cmd /c $configure
if ($LASTEXITCODE -ne 0) {
    throw "Configure failed with exit code $LASTEXITCODE"
}

Write-Host "Harness check: build" -ForegroundColor Cyan
$build = "$commandPrefix && cmake --build $buildDir"
cmd /c $build
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

Write-Host "Harness check: tests" -ForegroundColor Cyan
$test = "$commandPrefix && ctest --test-dir $buildDir --output-on-failure --interactive-debug-mode 1"
cmd /c $test
if ($LASTEXITCODE -ne 0) {
    $firstTestExitCode = $LASTEXITCODE
    Write-Host "Harness check: tests failed with exit code $firstTestExitCode; retrying once after cleanup delay." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    cmd /c $test
    if ($LASTEXITCODE -ne 0) {
        $testLogs = Get-ChildItem -LiteralPath (Join-Path $root "$buildDir\tests") -Filter "*_ctest.txt" -ErrorAction SilentlyContinue
        foreach ($testLog in $testLogs) {
            Write-Host ("Harness check: {0} log tail" -f $testLog.Name) -ForegroundColor Yellow
            Get-Content -LiteralPath $testLog.FullName -Encoding UTF8 -Tail 120
        }
        throw "Tests failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Harness check passed." -ForegroundColor Green
