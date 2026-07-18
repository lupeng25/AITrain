param(
    [string]$BuildDir = "build-vscode",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "toolchain-env.ps1")

$vcvars = Resolve-AITrainVcVars
$qt = Resolve-AITrainQtRoot
$commandPrefix = Get-AITrainBuildCommandPrefix -VcVars $vcvars -QtRoot $qt

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
    $configure = "$commandPrefix && cmake -S . -B `"$BuildDir`" -G `"NMake Makefiles`" -DCMAKE_PREFIX_PATH=`"$qt`" -DAITRAIN_BUILD_TESTS=ON"
    cmd /c $configure
    if ($LASTEXITCODE -ne 0) {
        throw "Configure failed with exit code $LASTEXITCODE"
    }

    Write-Host "Package smoke: build" -ForegroundColor Cyan
    $build = "$commandPrefix && cmake --build `"$BuildDir`""
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
$install = "$commandPrefix && cmake --install `"$BuildDir`" --prefix `"$prefixFull`""
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
Assert-PathExists "runtimes\onnxruntime" "ONNX Runtime folder"
Assert-PathExists "runtimes\tensorrt" "TensorRT folder"
Assert-PathExists "examples" "examples folder"
Assert-PathExists "examples\create-minimal-datasets.py" "minimal dataset generator"
Assert-PathExists "docs\harness\current-status.md" "harness docs"
Assert-PathExists "docs\user-guide.md" "user guide"
Assert-PathExists "docs\deps-layout.md" ".deps layout docs"
Assert-PathExists "docs\training-backends.md" "training backend docs"
Assert-PathExists "docs\hardware-compatibility.md" "hardware compatibility docs"
Assert-PathExists "docs\acceptance-runbook.md" "acceptance runbook"
Assert-PathExists "docs\yolo-model-support-matrix.md" "YOLO model support matrix"
Assert-PathExists "docs\local-rc-closeout.md" "local RC closeout checklist"
Assert-PathExists "docs\external-acceptance-handoff.md" "external acceptance handoff"
Assert-PathExists "docs\release-freeze-handoff.md" "release freeze handoff"
Assert-PathExists "docs\production-ocr-acceptance.md" "production OCR acceptance runbook"
Assert-PathExists "docs\customer-ocr-validation.md" "customer OCR validation runbook"
Assert-PathExists "docs\acceptance-templates\clean-windows-acceptance-result.md" "clean Windows acceptance template"
Assert-PathExists "docs\acceptance-templates\tensorrt-acceptance-result.md" "TensorRT acceptance template"
Assert-PathExists "docs\acceptance-templates\production-ocr-acceptance-result.md" "production OCR acceptance template"
Assert-PathExists "docs\product-roadmap-local-training-platform.md" "local training platform roadmap"
Assert-PathExists "python_trainers\requirements-yolo.txt" "YOLO Python requirements"
Assert-PathExists "python_trainers\requirements-smp.txt" "SMP Python requirements"
Assert-PathExists "python_trainers\requirements-anomaly.txt" "Anomalib Python requirements"
Assert-PathExists "python_trainers\requirements-ocr.txt" "OCR Python requirements"
Assert-PathExists "python_trainers\yolo\ultralytics_evaluator.py" "Official Ultralytics evaluator"
Assert-PathExists "python_trainers\semantic_segmentation\smp_trainer.py" "SMP semantic segmentation trainer"
Assert-PathExists "python_trainers\semantic_segmentation\smp_evaluator.py" "SMP semantic segmentation evaluator"
Assert-PathExists "python_trainers\anomaly\anomalib_adapter.py" "Anomalib anomaly detection adapter"
Assert-PathExists "python_trainers\ocr_rec\paddleocr_official_adapter.py" "Official PaddleOCR adapter"
Assert-PathExists "python_trainers\ocr_det\paddleocr_det_official_adapter.py" "Official PaddleOCR Det adapter"
Assert-PathExists "python_trainers\ocr_system\paddleocr_system_official_adapter.py" "Official PaddleOCR System adapter"
if (Test-Path (Join-Path $prefixFull "python_trainers\mock_trainer.py")) {
    throw "Diagnostic Python mock trainer must not be packaged"
}
if (Test-Path (Join-Path $prefixFull "python_trainers\ocr_rec\paddleocr_trainer.py")) {
    throw "Removed small PaddleOCR CTC trainer must not be packaged"
}
Assert-PathExists "installer\AITrainStudio.iss" "Inno Setup installer script"
Assert-PathExists "installer\AITrainStudioDependencies.iss" "Inno Setup dependency installer script"
Assert-PathExists "installer\AITrainStudioPythonEnv.iss" "Inno Setup Python environment installer script"
Assert-PathExists "tools\acceptance-smoke.ps1" "acceptance smoke script"
Assert-PathExists "tools\build-inno-installer.ps1" "Inno Setup installer build script"
Assert-PathExists "tools\ui-workbench-walkthrough.ps1" "UI workbench walkthrough RC script"
Assert-PathExists "tools\phase45-yolo-model-matrix-smoke.ps1" "Phase 45 YOLO model matrix smoke script"
Assert-PathExists "tools\phase-smp-semantic-segmentation-smoke.ps1" "SMP semantic segmentation smoke script"
Assert-PathExists "tools\phase-anomaly-anomalib-smoke.ps1" "Anomalib anomaly detection smoke script"
Assert-PathExists "tools\phase-anomaly-mvtec-quality-matrix.ps1" "Anomalib MVTec quality matrix script"
Assert-PathExists "tools\full-model-lifecycle-progress-server.py" "full model lifecycle progress server"
Assert-PathExists "tools\local-rc-closeout.ps1" "local RC closeout script"
Assert-PathExists "tools\release-freeze-handoff.ps1" "release freeze handoff script"
Assert-PathExists "tools\materialize-ultralytics-dataset.py" "Ultralytics dataset materializer"
Assert-PathExists "tools\materialize-oxford-pets-semantic.py" "Oxford Pets SMP dataset materializer"
Assert-PathExists "tools\phase31-paddleocr-full-official-smoke.ps1" "Phase 31 PaddleOCR full smoke script"
Assert-PathExists "tools\phase-ppocrv6-model-matrix-smoke.ps1" "PP-OCRv6 model matrix smoke script"
Assert-PathExists "tools\phase47-paddleocr-det-onnx-smoke.ps1" "historical Phase 47 PaddleOCR Det ONNX compatibility script"
Assert-PathExists "tools\prepare-production-ocr-data.ps1" "production OCR public data preparation script"
Assert-PathExists "tools\prepare_production_ocr_data.py" "production OCR public data preparation helper"
Assert-PathExists "tools\run-production-ocr-rec-experiment.ps1" "production OCR Rec experiment script"
Assert-PathExists "tools\run-production-ocr-official-chain.ps1" "production OCR official chain script"
Assert-PathExists "tools\production-ocr-acceptance.ps1" "production OCR acceptance script"
Assert-PathExists "tools\customer-ocr-validation.ps1" "customer OCR validation script"
Assert-PathExists "tools\phase50-paddleocr-v5-gpu-official-chain.ps1" "Phase 50 PP-OCRv5 GPU official chain script"

$forbiddenLegacyPaths = @(
    "plugins",
    "pluginModels",
    "examples\plugin-package-template",
    "docs\plugin-marketplace.md",
    "docs\plugin-package-format.md"
)
foreach ($legacyPath in $forbiddenLegacyPaths) {
    if (Test-Path -LiteralPath (Join-Path $prefixFull $legacyPath)) {
        throw "Package contains removed legacy plugin path: $legacyPath"
    }
}
Write-Host "  [ok] no legacy dynamic-plugin package paths"

$pythonCacheDirs = @(Get-ChildItem -LiteralPath $prefixFull -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue)
$pythonCacheFiles = @(Get-ChildItem -LiteralPath $prefixFull -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq ".pyc" -or $_.Extension -eq ".pyo" })
if ($pythonCacheDirs.Count -gt 0 -or $pythonCacheFiles.Count -gt 0) {
    $cacheItems = @()
    $cacheItems += $pythonCacheDirs | Select-Object -First 10 | ForEach-Object { $_.FullName }
    $cacheItems += $pythonCacheFiles | Select-Object -First 10 | ForEach-Object { $_.FullName }
    throw ("Package contains Python cache artifacts: {0}" -f ($cacheItems -join "; "))
}
Write-Host "  [ok] no Python cache artifacts"

$privateKeyFiles = @(Get-ChildItem -LiteralPath $prefixFull -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -like "*aitrain-license-private-key*.json" -or
        $_.Name -like "*license-private-key*.json" -or
        $_.Name -like "*private-key*.json"
    })
if ($privateKeyFiles.Count -gt 0) {
    $privateKeyItems = $privateKeyFiles | Select-Object -First 10 | ForEach-Object { $_.FullName }
    throw ("Package contains license private-key material: {0}" -f ($privateKeyItems -join "; "))
}
Write-Host "  [ok] no license private-key artifacts"

$cmakeCachePath = Join-Path $buildPathFull "CMakeCache.txt"
$publicKeyLine = if (Test-Path -LiteralPath $cmakeCachePath) {
    Select-String -LiteralPath $cmakeCachePath -Pattern '^AITRAIN_LICENSE_PUBLIC_KEY:[^=]*=(.*)$' -ErrorAction SilentlyContinue | Select-Object -First 1
} else {
    $null
}
$configuredPublicKey = if ($publicKeyLine) { [string]$publicKeyLine.Matches[0].Groups[1].Value } else { "" }
if ([string]::IsNullOrWhiteSpace($configuredPublicKey)) {
    throw "Package build has no AITRAIN_LICENSE_PUBLIC_KEY configured. Set AITRAIN_LICENSE_PUBLIC_KEY or AITRAIN_LICENSE_PUBLIC_KEY_FILE before packaging."
}
Write-Host "  [ok] license public key configured"

$onnxRuntimeRootDll = Join-Path $prefixFull "onnxruntime.dll"
$onnxRuntimeFolderDll = Join-Path $prefixFull "runtimes\onnxruntime\onnxruntime.dll"
if ((Test-Path $onnxRuntimeRootDll) -or (Test-Path $onnxRuntimeFolderDll)) {
    Write-Host "  [ok] ONNX Runtime DLL"
} else {
    Write-Host "  [warn] ONNX Runtime DLL not packaged; ONNX Runtime SDK may be disabled for this build." -ForegroundColor Yellow
}

foreach ($qtModule in @("Core", "Gui", "Widgets")) {
    $qtDll = Get-ChildItem -LiteralPath $prefixFull -Filter "Qt5$qtModule*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $qtDll) {
        throw "Missing required Qt5$qtModule runtime DLL under package root"
    }
    Write-Host "  [ok] Qt5$qtModule runtime DLL"
}
$qtWindowsPlugin = @("platforms\qwindows.dll", "platforms\qwindowsd.dll") |
    ForEach-Object { Join-Path $prefixFull $_ } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $qtWindowsPlugin) {
    throw "Missing Qt Windows platform plugin: platforms\qwindows.dll or platforms\qwindowsd.dll"
}
Write-Host ("  [ok] Qt Windows platform plugin: {0}" -f (Split-Path -Leaf $qtWindowsPlugin))
Assert-PathExists "translations\aitrain_zh_CN.qm" "Chinese translation catalog"
Assert-PathExists "translations\aitrain_en_US.qm" "English translation catalog"

Write-Host "Package smoke: GUI package-root startup handshake" -ForegroundColor Cyan
$studioExe = Join-Path $prefixFull "AITrainStudio.exe"
$studioProcess = Start-Process -FilePath $studioExe `
    -ArgumentList "--package-startup-check" `
    -WorkingDirectory $prefixFull `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
if ($studioProcess.ExitCode -ne 0) {
    throw "Packaged GUI startup handshake failed with exit code $($studioProcess.ExitCode)"
}
Write-Host "  [ok] packaged GUI loaded Qt platform and translation runtime from package root"

$workerExe = Join-Path $prefixFull "aitrain_worker.exe"

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

Write-Host "Package smoke: built-in capability registry" -ForegroundColor Cyan
$capabilityOutput = & $workerExe --builtin-capabilities
if ($LASTEXITCODE -ne 0) {
    throw "Packaged built-in capability check failed with exit code $LASTEXITCODE"
}
$capabilityCheck = $capabilityOutput | Select-Object -Last 1 | ConvertFrom-Json
if (-not $capabilityCheck.ok) {
    throw "Built-in capability check reported ok=false"
}
if ($capabilityCheck.capabilityCount -lt 1) {
    throw "Expected at least one built-in capability"
}
Write-Host ("  [ok] built-in capabilities={0}" -f $capabilityCheck.capabilityCount)

Write-Host "Package smoke: workspace first-start and reopen" -ForegroundColor Cyan
$workspaceSelfCheckRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("aitrain-package-workspace-" + [guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Path $workspaceSelfCheckRoot -Force | Out-Null
    $workspaceSelfCheckOutput = & $workerExe --workspace-self-check --workspace $workspaceSelfCheckRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged worker workspace self-check failed with exit code $LASTEXITCODE"
    }
    $workspaceSelfCheck = $workspaceSelfCheckOutput | Select-Object -Last 1 | ConvertFrom-Json
    if ((-not $workspaceSelfCheck.ok) -or (-not $workspaceSelfCheck.firstOpen) -or (-not $workspaceSelfCheck.secondOpen) -or (-not $workspaceSelfCheck.layoutValid) -or (-not $workspaceSelfCheck.stagingClean) -or $workspaceSelfCheck.projectsTablePresent -or ($workspaceSelfCheck.storedSchemaVersion -ne 11)) {
        throw "Workspace first-start self-check reported an invalid result"
    }
    Write-Host ("  [ok] firstOpen={0}, secondOpen={1}, schema={2}" -f `
        $workspaceSelfCheck.firstOpen, $workspaceSelfCheck.secondOpen, $workspaceSelfCheck.storedSchemaVersion)
}
finally {
    if (Test-Path -LiteralPath $workspaceSelfCheckRoot) {
        Remove-Item -LiteralPath $workspaceSelfCheckRoot -Recurse -Force
    }
}

Write-Host "Package smoke passed: $prefixFull" -ForegroundColor Green
