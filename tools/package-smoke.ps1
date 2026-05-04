param(
    [string]$BuildDir = "build-vscode",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$qt = "C:\Qt\Qt5.12.9\5.12.9\msvc2015_64"

if (-not (Test-Path $vcvars)) {
    throw "MSVC environment script not found: $vcvars"
}

if (-not (Test-Path $qt)) {
    throw "Qt kit not found: $qt"
}

Set-Location $root

$buildPath = Join-Path $root $BuildDir
$prefix = Join-Path $buildPath "package-smoke"
$buildPathFull = [System.IO.Path]::GetFullPath($buildPath)
$prefixFull = [System.IO.Path]::GetFullPath($prefix)
$expectedPrefixParent = [System.IO.Path]::GetFullPath((Join-Path $buildPathFull "."))

if (-not $prefixFull.StartsWith($expectedPrefixParent, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use package smoke prefix outside build directory: $prefixFull"
}

if (-not $SkipBuild) {
    Write-Host "Package smoke: configure" -ForegroundColor Cyan
    $configure = "call `"$vcvars`" >nul && cmake -S . -B `"$BuildDir`" -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=`"$qt`" -DAITRAIN_BUILD_TESTS=ON"
    cmd /c $configure
    if ($LASTEXITCODE -ne 0) {
        throw "Configure failed with exit code $LASTEXITCODE"
    }

    Write-Host "Package smoke: build" -ForegroundColor Cyan
    $build = "call `"$vcvars`" >nul && cmake --build `"$BuildDir`""
    cmd /c $build
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
}

if (Test-Path $prefixFull) {
    $leaf = Split-Path -Leaf $prefixFull
    if ($leaf -ne "package-smoke") {
        throw "Refusing to remove unexpected package smoke directory: $prefixFull"
    }
    Remove-Item -LiteralPath $prefixFull -Recurse -Force
}

Write-Host "Package smoke: install" -ForegroundColor Cyan
$install = "call `"$vcvars`" >nul && cmake --install `"$BuildDir`" --prefix `"$prefixFull`""
cmd /c $install
if ($LASTEXITCODE -ne 0) {
    throw "Install failed with exit code $LASTEXITCODE"
}

function Assert-PathExists {
    param(
        [string]$RelativePath,
        [string]$Description
    )

    $path = Join-Path $prefixFull $RelativePath
    if (-not (Test-Path $path)) {
        throw "Missing $Description`: $RelativePath"
    }
    Write-Host "  [ok] $RelativePath"
}

Write-Host "Package smoke: verify layout" -ForegroundColor Cyan
Assert-PathExists "AITrainStudio.exe" "AITrain Studio executable"
Assert-PathExists "aitrain_worker.exe" "Worker executable"
Assert-PathExists "plugins\models" "model plugin directory"
Assert-PathExists "plugins\models\DatasetInteropPlugin.dll" "dataset interop plugin"
Assert-PathExists "plugins\models\YoloNativePlugin.dll" "YOLO native plugin"
Assert-PathExists "plugins\models\OcrRecNativePlugin.dll" "OCR Rec plugin"
Assert-PathExists "runtimes\onnxruntime" "ONNX Runtime folder"
Assert-PathExists "runtimes\tensorrt" "TensorRT folder"
Assert-PathExists "examples" "examples folder"
Assert-PathExists "examples\create-minimal-datasets.py" "minimal dataset generator"
Assert-PathExists "docs\harness\current-status.md" "harness docs"
Assert-PathExists "docs\training-backends.md" "training backend docs"
Assert-PathExists "docs\hardware-compatibility.md" "hardware compatibility docs"
Assert-PathExists "docs\acceptance-runbook.md" "acceptance runbook"
Assert-PathExists "docs\local-rc-closeout.md" "local RC closeout checklist"
Assert-PathExists "docs\product-roadmap-local-training-platform.md" "local training platform roadmap"
Assert-PathExists "python_trainers\mock_trainer.py" "Python trainer adapter mock"
Assert-PathExists "python_trainers\requirements-yolo.txt" "YOLO Python requirements"
Assert-PathExists "python_trainers\requirements-ocr.txt" "OCR Python requirements"
Assert-PathExists "python_trainers\ocr_rec\paddleocr_official_adapter.py" "Official PaddleOCR adapter"
Assert-PathExists "python_trainers\ocr_det\paddleocr_det_official_adapter.py" "Official PaddleOCR Det adapter"
Assert-PathExists "python_trainers\ocr_system\paddleocr_system_official_adapter.py" "Official PaddleOCR System adapter"
Assert-PathExists "tools\acceptance-smoke.ps1" "acceptance smoke script"
Assert-PathExists "tools\local-rc-closeout.ps1" "local RC closeout script"
Assert-PathExists "tools\materialize-ultralytics-dataset.py" "Ultralytics dataset materializer"
Assert-PathExists "tools\phase31-paddleocr-full-official-smoke.ps1" "Phase 31 PaddleOCR full smoke script"

$onnxRuntimeRootDll = Join-Path $prefixFull "onnxruntime.dll"
$onnxRuntimeFolderDll = Join-Path $prefixFull "runtimes\onnxruntime\onnxruntime.dll"
if ((Test-Path $onnxRuntimeRootDll) -or (Test-Path $onnxRuntimeFolderDll)) {
    Write-Host "  [ok] ONNX Runtime DLL"
} else {
    Write-Host "  [warn] ONNX Runtime DLL not packaged; ONNX Runtime SDK may be disabled for this build." -ForegroundColor Yellow
}

$qtCoreDll = Get-ChildItem -LiteralPath $prefixFull -Filter "Qt5Core*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($qtCoreDll) {
    Write-Host "  [ok] Qt runtime DLL"
} else {
    Write-Host "  [warn] Qt runtime DLL not found; check AITRAIN_INSTALL_QT_RUNTIME or windeployqt availability." -ForegroundColor Yellow
}

$workerExe = Join-Path $prefixFull "aitrain_worker.exe"
$pluginDir = Join-Path $prefixFull "plugins\models"

Write-Host "Package smoke: worker self-check" -ForegroundColor Cyan
$workerSelfCheckOutput = & $workerExe --self-check
if ($LASTEXITCODE -ne 0) {
    throw "Packaged worker self-check failed with exit code $LASTEXITCODE"
}
$workerSelfCheck = $workerSelfCheckOutput | Select-Object -Last 1 | ConvertFrom-Json
if (-not $workerSelfCheck.ok) {
    throw "Packaged worker self-check reported ok=false"
}
Write-Host ("  [ok] worker self-check status={0}" -f $workerSelfCheck.status)
$missingRuntimeChecks = @($workerSelfCheck.checks | Where-Object { $_.status -eq "missing" } | ForEach-Object { $_.name })
if ($missingRuntimeChecks.Count -gt 0) {
    Write-Host ("  [info] missing optional runtimes: {0}" -f ($missingRuntimeChecks -join ", "))
}

Write-Host "Package smoke: plugin load" -ForegroundColor Cyan
$pluginSmokeOutput = & $workerExe --plugin-smoke $pluginDir
if ($LASTEXITCODE -ne 0) {
    throw "Packaged plugin smoke failed with exit code $LASTEXITCODE"
}
$pluginSmoke = $pluginSmokeOutput | Select-Object -Last 1 | ConvertFrom-Json
if (-not $pluginSmoke.ok) {
    throw "Packaged plugin smoke reported ok=false"
}
if ($pluginSmoke.pluginCount -lt 3) {
    throw "Expected at least 3 packaged plugins, found $($pluginSmoke.pluginCount)"
}
Write-Host ("  [ok] packaged plugins={0}" -f $pluginSmoke.pluginCount)

Write-Host "Package smoke passed: $prefixFull" -ForegroundColor Green
