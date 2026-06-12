param(
    [string]$BuildDir = "build-vscode",
    [string]$SourceDir = "",
    [string]$OutputDir = "",
    [string]$InnoCompiler = "",
    [string]$AppVersion = "",
    [string]$Compression = "lzma/normal",
    [ValidateSet("Split", "Product", "Dependencies", "PythonEnv", "Full")]
    [string]$PackageMode = "Split",
    [string]$PythonEnvSourceDir = "",
    [string]$PaddleOcrSourceDir = "",
    [switch]$SkipPythonEnvProbe,
    [switch]$SkipPackageSmoke,
    [switch]$ExcludeTensorRt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Find-InnoCompiler {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $full = Resolve-RepoPath $RequestedPath
        if (-not (Test-Path $full)) {
            throw "Inno Setup compiler not found: $full"
        }
        return $full
    }

    $fromPath = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    throw "Inno Setup 6 compiler was not found. Install Inno Setup 6 or pass -InnoCompiler <path-to-ISCC.exe>."
}

function Read-CMakeProjectVersion {
    $cmakePath = Join-Path $root "CMakeLists.txt"
    $cmakeText = Get-Content -LiteralPath $cmakePath -Encoding UTF8 -Raw
    $match = [regex]::Match($cmakeText, 'project\s*\(\s*AITrainStudio\s+VERSION\s+([0-9]+(?:\.[0-9]+){1,3})', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        throw "Unable to read AITrainStudio project version from CMakeLists.txt"
    }
    return $match.Groups[1].Value
}

function Test-PythonEnvRoot {
    param([string]$Path)

    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }
    return (Test-Path -LiteralPath (Join-Path $Path "python.exe") -PathType Leaf) -or
        (Test-Path -LiteralPath (Join-Path $Path "Scripts\python.exe") -PathType Leaf)
}

function Resolve-PythonEnvSource {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $full = Resolve-RepoPath $RequestedPath
        if (-not (Test-PythonEnvRoot $full)) {
            throw "Python environment source must contain python.exe or Scripts\python.exe: $full"
        }
        return $full
    }

    $candidates = @(
        (Join-Path $BuildDir "python_env"),
        ".deps\aitrain-python-env",
        ".deps\python-3.13.13-ocr-amd64",
        ".deps\python-3.13.13-embed-amd64",
        ".deps\rtx4090-validation\python-yolo-venv",
        ".deps\rtx4090-validation\python-ocr-gpu",
        ".deps\rtx4090-validation\python-3.12-paddle2onnx"
    )
    foreach ($candidate in $candidates) {
        $full = Resolve-RepoPath $candidate
        if (Test-PythonEnvRoot $full) {
            return $full
        }
    }

    throw "No Python environment source was found. Pass -PythonEnvSourceDir <prepared-python-env-root>."
}

function Resolve-PythonExecutableInEnv {
    param([string]$PythonEnvRoot)

    $scriptPython = Join-Path $PythonEnvRoot "Scripts\python.exe"
    if (Test-Path -LiteralPath $scriptPython -PathType Leaf) {
        return $scriptPython
    }
    $rootPython = Join-Path $PythonEnvRoot "python.exe"
    if (Test-Path -LiteralPath $rootPython -PathType Leaf) {
        return $rootPython
    }
    throw "Python environment source must contain python.exe or Scripts\python.exe: $PythonEnvRoot"
}

function Test-PythonModuleSet {
    param(
        [string]$PythonExecutable,
        [string[]]$RequiredModules
    )

    $moduleList = ($RequiredModules | ForEach-Object { "'$_'" }) -join ","
    $code = "import importlib.util, json, sys; mods=[$moduleList]; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(json.dumps({'missing': missing})); sys.exit(0 if not missing else 7)"
    $output = & $PythonExecutable -c $code 2>$null
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        throw "Python environment is missing required modules for AITrain official backends: $output"
    }
}

function Resolve-OptionalExistingPath {
    param([string]$Path)

    if (-not $Path) {
        return ""
    }
    $full = Resolve-RepoPath $Path
    if (-not (Test-Path -LiteralPath $full -PathType Container)) {
        throw "Path does not exist: $full"
    }
    return $full
}

Set-Location $root

if (-not $AppVersion) {
    $AppVersion = Read-CMakeProjectVersion
}

if (-not $SourceDir) {
    $SourceDir = Join-Path $BuildDir "package-smoke"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $BuildDir "inno"
}

$sourceFull = Resolve-RepoPath $SourceDir
$outputFull = Resolve-RepoPath $OutputDir
$issPath = Resolve-RepoPath "installer\AITrainStudio.iss"
$dependenciesIssPath = Resolve-RepoPath "installer\AITrainStudioDependencies.iss"
$pythonEnvIssPath = Resolve-RepoPath "installer\AITrainStudioPythonEnv.iss"

$baseExcludes = @(
    "*.pdb",
    "*.ilk",
    "*.exp",
    "*.lib",
    "installer\*",
    "tools\build-inno-installer.ps1"
)
$runtimeExcludes = @(
    "runtimes\*",
    "platforms\*",
    "imageformats\*",
    "iconengines\*",
    "sqldrivers\*",
    "styles\*",
    "bearer\*",
    "Qt5*.dll",
    "Qt6*.dll",
    "libEGL*.dll",
    "libGLES*.dll",
    "opengl32sw.dll",
    "d3dcompiler_47.dll",
    "vcruntime*.dll",
    "msvcp*.dll",
    "concrt*.dll",
    "vccorlib*.dll",
    "api-ms-win-*.dll",
    "ucrtbase*.dll",
    "onnxruntime*.dll",
    "ncnn.dll"
)
$fullPackageExcludes = @($baseExcludes)
if ($ExcludeTensorRt) {
    $fullPackageExcludes += "runtimes\tensorrt\*"
}
$productPackageExcludes = @($baseExcludes) + $runtimeExcludes
$dependencyPackageExcludes = @()
if ($ExcludeTensorRt) {
    $dependencyPackageExcludes += "runtimes\tensorrt\*"
}

function Join-InnoExcludeList {
    param([string[]]$Items)

    return (($Items | Where-Object { $_ }) -join ",")
}

if (-not $SkipPackageSmoke) {
    Write-Host "Inno installer: refresh package-smoke layout" -ForegroundColor Cyan
    & (Join-Path $root "tools\package-smoke.ps1") -BuildDir $BuildDir
    if ($LASTEXITCODE -ne 0) {
        throw "package-smoke.ps1 failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $sourceFull)) {
    throw "Installer source layout not found: $sourceFull"
}
if (-not (Test-Path (Join-Path $sourceFull "AITrainStudio.exe"))) {
    throw "Installer source layout is missing AITrainStudio.exe: $sourceFull"
}
if (-not (Test-Path (Join-Path $sourceFull "aitrain_worker.exe"))) {
    throw "Installer source layout is missing aitrain_worker.exe: $sourceFull"
}
if (-not (Test-Path $issPath)) {
    throw "Inno script not found: $issPath"
}
if (-not (Test-Path $dependenciesIssPath)) {
    throw "Inno dependency script not found: $dependenciesIssPath"
}
if (-not (Test-Path $pythonEnvIssPath)) {
    throw "Inno Python environment script not found: $pythonEnvIssPath"
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null

$iscc = Find-InnoCompiler -RequestedPath $InnoCompiler

function Invoke-InnoBuild {
    param(
        [string]$Label,
        [string]$ScriptPath,
        [string]$OutputBaseFilename,
        [string[]]$Excludes,
        [string]$PythonEnvSource,
        [string]$PaddleOcrSource
    )

    $packageExcludes = Join-InnoExcludeList -Items $Excludes

    Write-Host "Inno installer: compile $Label" -ForegroundColor Cyan
    Write-Host "  SourceDir: $sourceFull"
    Write-Host "  OutputDir: $outputFull"
    Write-Host "  Version:   $AppVersion"
    Write-Host "  BaseName:  $OutputBaseFilename"
    Write-Host "  Excludes:  $packageExcludes"
    if ($PythonEnvSource) {
        Write-Host "  PythonEnv: $PythonEnvSource"
    }
    if ($PaddleOcrSource) {
        Write-Host "  PaddleOCR: $PaddleOcrSource"
    }

    $innoArgs = @(
        "/DSourceDir=$sourceFull",
        "/DOutputDir=$outputFull",
        "/DAppVersion=$AppVersion",
        "/DOutputBaseFilename=$OutputBaseFilename",
        "/DPackageExcludes=$packageExcludes",
        "/DInstallerCompression=$Compression",
        "/DInstallerSolidCompression=no",
        $ScriptPath
    )
    if ($PythonEnvSource) {
        $innoArgs = @(
            "/DPythonEnvSourceDir=$PythonEnvSource",
            "/DPaddleOcrSourceDir=$PaddleOcrSource"
        ) + $innoArgs
    }
    & $iscc @innoArgs | Out-Host
    $innoExitCode = $LASTEXITCODE
    if ($innoExitCode -ne 0) {
        throw "Inno Setup compiler failed for $Label with exit code $innoExitCode"
    }

    $installerPath = Join-Path $outputFull "$OutputBaseFilename.exe"
    if (-not (Test-Path $installerPath)) {
        throw "Expected installer was not created: $installerPath"
    }

    $hash = Get-FileHash -LiteralPath $installerPath -Algorithm SHA256
    Write-Host "Inno installer created: $installerPath" -ForegroundColor Green
    Write-Host "SHA256: $($hash.Hash)" -ForegroundColor Green

    return [pscustomobject]@{
        Label = $Label
        Path = $installerPath
        Sha256 = $hash.Hash
    }
}

$pythonEnvFull = ""
$paddleOcrFull = ""
if ($PackageMode -eq "Split" -or $PackageMode -eq "PythonEnv") {
    $pythonEnvFull = Resolve-PythonEnvSource -RequestedPath $PythonEnvSourceDir
    $paddleOcrFull = if ($PaddleOcrSourceDir) {
        Resolve-OptionalExistingPath -Path $PaddleOcrSourceDir
    } else {
        $candidate = Resolve-RepoPath ".deps\PaddleOCR"
        if (Test-Path -LiteralPath (Join-Path $candidate "tools\train.py") -PathType Leaf) { $candidate } else { "" }
    }
    if (-not $SkipPythonEnvProbe) {
        $pythonExe = Resolve-PythonExecutableInEnv -PythonEnvRoot $pythonEnvFull
        Test-PythonModuleSet -PythonExecutable $pythonExe -RequiredModules @(
            "ultralytics",
            "torch",
            "onnx",
            "onnxruntime",
            "paddle",
            "paddleocr",
            "numpy",
            "PIL",
            "yaml"
        )
    }
    if (Test-Path -LiteralPath (Join-Path $pythonEnvFull "pyvenv.cfg") -PathType Leaf) {
        Write-Warning "PythonEnvSourceDir appears to be a venv. Use a relocatable staging environment for clean-machine customer packages."
    }
}

$builds = @()
switch ($PackageMode) {
    "Split" {
        $builds += @{
            Label = "product"
            ScriptPath = $issPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-Product-Setup"
            Excludes = $productPackageExcludes
        }
        $builds += @{
            Label = "dependencies"
            ScriptPath = $dependenciesIssPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-Dependencies-Setup"
            Excludes = $dependencyPackageExcludes
        }
        $builds += @{
            Label = "python-env"
            ScriptPath = $pythonEnvIssPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-PythonEnv-Setup"
            Excludes = @()
            PythonEnvSource = $pythonEnvFull
            PaddleOcrSource = $paddleOcrFull
        }
    }
    "Product" {
        $builds += @{
            Label = "product"
            ScriptPath = $issPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-Product-Setup"
            Excludes = $productPackageExcludes
        }
    }
    "Dependencies" {
        $builds += @{
            Label = "dependencies"
            ScriptPath = $dependenciesIssPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-Dependencies-Setup"
            Excludes = $dependencyPackageExcludes
        }
    }
    "PythonEnv" {
        $builds += @{
            Label = "python-env"
            ScriptPath = $pythonEnvIssPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-PythonEnv-Setup"
            Excludes = @()
            PythonEnvSource = $pythonEnvFull
            PaddleOcrSource = $paddleOcrFull
        }
    }
    "Full" {
        $builds += @{
            Label = "full"
            ScriptPath = $issPath
            OutputBaseFilename = "AITrainStudio-$AppVersion-Setup"
            Excludes = $fullPackageExcludes
        }
    }
}

$results = @()
foreach ($build in $builds) {
    $pythonSource = if ($build.ContainsKey("PythonEnvSource")) { $build["PythonEnvSource"] } else { "" }
    $paddleSource = if ($build.ContainsKey("PaddleOcrSource")) { $build["PaddleOcrSource"] } else { "" }
    $results += Invoke-InnoBuild `
        -Label $build["Label"] `
        -ScriptPath $build["ScriptPath"] `
        -OutputBaseFilename $build["OutputBaseFilename"] `
        -Excludes $build["Excludes"] `
        -PythonEnvSource $pythonSource `
        -PaddleOcrSource $paddleSource
}

Write-Host "Inno installer output summary:" -ForegroundColor Cyan
foreach ($result in $results) {
    Write-Host "  [$($result.Label)] $($result.Path)"
    Write-Host "       SHA256: $($result.Sha256)"
}
