param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Configure", "Build", "Test")]
    [string]$Action
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "toolchain-env.ps1")

$vcvars = Resolve-AITrainVcVars
$qt = Resolve-AITrainQtRoot
$buildDir = if ($env:AITRAIN_BUILD_DIR) { $env:AITRAIN_BUILD_DIR } else { "build-vscode" }
$buildPath = if ([System.IO.Path]::IsPathRooted($buildDir)) {
    [System.IO.Path]::GetFullPath($buildDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $buildDir))
}

Set-Location $root
$prefix = Get-AITrainBuildCommandPrefix -VcVars $vcvars -QtRoot $qt
$quotedBuildPath = '"{0}"' -f $buildPath
$quotedQt = '"{0}"' -f $qt

switch ($Action) {
    "Configure" {
        $command = '{0} && cmake -S . -B {1} -G "NMake Makefiles" -DCMAKE_PREFIX_PATH={2} -DAITRAIN_BUILD_TESTS=ON' -f $prefix, $quotedBuildPath, $quotedQt
    }
    "Build" {
        $command = '{0} && cmake --build {1}' -f $prefix, $quotedBuildPath
    }
    "Test" {
        $command = '{0} && set "QT_QPA_PLATFORM=offscreen" && ctest --test-dir {1} --output-on-failure' -f $prefix, $quotedBuildPath
    }
}

cmd /c $command
exit $LASTEXITCODE
