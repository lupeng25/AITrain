param(
    [string]$WorkDir = ".deps\phase47-paddleocr-det-onnx-smoke",
    [string]$Phase31WorkDir = ".deps\phase31-paddleocr-full-official-smoke",
    [string]$PythonDir = ".deps\python-3.13.13-ocr-amd64",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$ConversionPythonDir = ".deps\python-3.12.10-paddle2onnx-nightly-amd64",
    [string]$ConversionPythonUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip",
    [string]$ConversionPythonZip = ".deps\python-3.12.10-embed-amd64.zip",
    [string]$PaddlePaddleRequirement = "paddlepaddle",
    [string]$PaddlePaddleIndexUrl = "https://www.paddlepaddle.org.cn/packages/nightly/cpu/",
    [string]$PaddleXRequirement = "paddlex==3.5.1",
    [string]$Paddle2OnnxRequirement = "paddle2onnx==2.1.0",
    [string]$BuildDir = "build-vscode",
    [string]$ImagePath = "",
    [switch]$UseStablePaddleForConversion,
    [switch]$SkipInstall,
    [switch]$SkipPhase31,
    [switch]$SkipCtest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow

function Write-Step {
    param([string]$Message)
    Write-Host "Phase47 PaddleOCR Det ONNX: $Message" -ForegroundColor Cyan
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Description
    )
    if (!(Test-Path -LiteralPath $Path)) {
        throw "Missing $Description`: $Path"
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $script:Root
    )

    Write-Step ("{0} {1}" -f $FilePath, ($Arguments -join " "))
    Push-Location $WorkingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
            $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
        } else {
            $output = & $FilePath @Arguments 2>&1
        }
        foreach ($line in $output) {
            Write-Host $line
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
        }
        return @($output)
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
}

function Invoke-Probe {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $script:Root
    )

    Push-Location $WorkingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
            $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $FilePath @Arguments 2>&1
        } else {
            $output = & $FilePath @Arguments 2>&1
        }
        return [ordered]@{
            exitCode = $LASTEXITCODE
            output = @($output | ForEach-Object { [string]$_ })
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force $parent | Out-Null
    $Value | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Test-PythonModule {
    param(
        [string]$Python,
        [string]$ModuleName
    )
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 3)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-EmbeddedPython {
    param(
        [string]$PythonDirPath,
        [string]$ZipPath,
        [string]$DownloadUrl
    )

    $python = Join-Path $PythonDirPath "python.exe"
    if (Test-Path -LiteralPath $python) {
        return $python
    }
    if ($SkipInstall) {
        throw "Missing conversion Python: $python. Re-run without -SkipInstall or provide -ConversionPythonDir."
    }

    New-Item -ItemType Directory -Force (Split-Path -Parent $ZipPath) | Out-Null
    if (!(Test-Path -LiteralPath $ZipPath)) {
        Write-Step "download Python 3.12 embeddable runtime"
        Invoke-Checked -FilePath "curl.exe" -Arguments @("-L", "--fail", "-o", $ZipPath, $DownloadUrl)
    }

    New-Item -ItemType Directory -Force $PythonDirPath | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $PythonDirPath -Force
    $pth = Get-ChildItem -LiteralPath $PythonDirPath -Filter "python*._pth" | Select-Object -First 1
    if ($null -ne $pth) {
        (Get-Content -LiteralPath $pth.FullName) -replace "#import site", "import site" |
            Set-Content -LiteralPath $pth.FullName -Encoding ASCII
    }
    return $python
}

function Ensure-Pip {
    param([string]$Python)

    $sitePackages = Join-Path (Split-Path -Parent $Python) "Lib\site-packages\pip"
    if (Test-Path -LiteralPath $sitePackages) {
        return
    }
    $getPip = Resolve-RepoPath ".deps\get-pip.py"
    Assert-PathExists $getPip "get-pip.py"
    Invoke-Checked -FilePath $Python -Arguments @($getPip)
}

function Ensure-ConversionModules {
    param([string]$Python)

    $missing = @()
    foreach ($module in @("paddle", "paddlex", "paddle2onnx", "onnx")) {
        if (!(Test-PythonModule -Python $Python -ModuleName $module)) {
            $missing += $module
        }
    }
    if ($missing.Count -gt 0) {
        if ($SkipInstall) {
            throw "Missing conversion modules: $($missing -join ', '). Re-run without -SkipInstall."
        }

        Write-Step "install PaddleX/Paddle2ONNX conversion modules"
        $installArgs = @("-m", "pip", "install", "--no-warn-script-location")
        if (!$UseStablePaddleForConversion -and $PaddlePaddleIndexUrl) {
            $installArgs += @("--pre", "--index-url", $PaddlePaddleIndexUrl, "--extra-index-url", "https://pypi.org/simple")
        }
        $installArgs += @($PaddlePaddleRequirement, $PaddleXRequirement, $Paddle2OnnxRequirement)
        Invoke-Checked -FilePath $Python -Arguments $installArgs
    }

    if (!$SkipInstall) {
        $nativeProbe = Invoke-Probe -FilePath $Python -Arguments @("-c", "import paddle2onnx")
        if ($nativeProbe.exitCode -ne 0) {
            Write-Step "install Paddle2ONNX through PaddleX plugin installer"
            Invoke-Checked -FilePath $Python -Arguments @("-m", "paddlex", "--install", "paddle2onnx", "-y")
        }
    }

    Write-Host "  [ok] PaddleX/Paddle2ONNX conversion modules available"
}

function Get-WorkerExe {
    $candidates = @(
        (Join-Path $script:Root "$BuildDir\bin\aitrain_worker.exe"),
        (Join-Path $script:Root "$BuildDir\aitrain_worker.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "aitrain_worker.exe was not found under $BuildDir. Run .\tools\harness-check.ps1 first."
}

function Ensure-Phase31DetArtifacts {
    param([string]$Phase31Root)

    $detRun = Join-Path $Phase31Root "runs\paddleocr_det_official"
    $inferenceDir = Join-Path $detRun "official_inference"
    $reportPath = Join-Path $detRun "paddleocr_official_det_report.json"
    $modelPath = Join-Path $inferenceDir "inference.pdmodel"
    $pirModelPath = Join-Path $inferenceDir "inference.json"
    $paramsPath = Join-Path $inferenceDir "inference.pdiparams"

    $ready = (Test-Path -LiteralPath $reportPath) -and
        ((Test-Path -LiteralPath $modelPath) -or (Test-Path -LiteralPath $pirModelPath)) -and
        (Test-Path -LiteralPath $paramsPath)

    if ($ready) {
        $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reportPath | ConvertFrom-Json
        if (($report.PSObject.Properties.Name -contains "ok") -and $report.ok) {
            Write-Host "  [ok] existing Phase 31 official Det inference model found"
            return
        }
    }

    if ($SkipPhase31) {
        throw "Phase 31 official Det inference model is incomplete. Missing or failed artifact under $detRun."
    }

    $phase31Script = Join-Path $script:Root "tools\phase31-paddleocr-full-official-smoke.ps1"
    Assert-PathExists $phase31Script "Phase 31 PaddleOCR full smoke script"
    $phase31Args = @(
        "-WorkDir", $Phase31WorkDir,
        "-PythonDir", $PythonDir,
        "-PaddleOcrRepo", $PaddleOcrRepo
    )
    if ($SkipInstall) {
        $phase31Args += "-SkipInstall"
    }
    Invoke-Checked -FilePath $phase31Script -Arguments $phase31Args

    Assert-PathExists $reportPath "Phase 31 Det report"
    if (!(Test-Path -LiteralPath $modelPath) -and !(Test-Path -LiteralPath $pirModelPath)) {
        throw "Missing Phase 31 Det inference model: expected $modelPath or $pirModelPath"
    }
    Assert-PathExists $paramsPath "Phase 31 Det inference params"
}

function Write-BlockedSummary {
    param(
        [string]$SummaryPath,
        [string]$Stage,
        [string]$Reason,
        [object]$Details
    )

    $finishedAt = [DateTime]::UtcNow
    $summary = [ordered]@{
        ok = $false
        phase = "47"
        status = "conversion-blocked"
        stage = $Stage
        reason = $Reason
        workDir = $work
        startedAt = $script:StartedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
        conversionPythonExecutable = $conversionPythonExe
        phase31WorkDir = $phase31Root
        sourceDetReportPath = $detReportPath
        sourceDetInferenceDir = $detInferenceDir
        requirements = [ordered]@{
            paddlepaddle = $PaddlePaddleRequirement
            paddlepaddleIndexUrl = if ($UseStablePaddleForConversion) { "" } else { $PaddlePaddleIndexUrl }
            paddlex = $PaddleXRequirement
            paddle2onnx = $Paddle2OnnxRequirement
        }
        details = $Details
        note = "Phase 47 requires PaddleX/Paddle2ONNX conversion before C++ ONNX Runtime smoke can run. This is a blocked conversion report, not passing ONNX wiring evidence."
    }
    Write-JsonFile -Path $SummaryPath -Value $summary
}

function Convert-DetInferenceToOnnx {
    param(
        [string]$Python,
        [string]$InferenceDir,
        [string]$OutputOnnx,
        [string]$SummaryPath
    )

    Assert-PathExists (Join-Path $InferenceDir "inference.pdiparams") "PaddleOCR Det inference.pdiparams"
    if (!(Test-Path -LiteralPath (Join-Path $InferenceDir "inference.json")) -and
        !(Test-Path -LiteralPath (Join-Path $InferenceDir "inference.pdmodel"))) {
        throw "Missing PaddleOCR Det inference model under $InferenceDir"
    }

    $onnxDir = Join-Path (Split-Path -Parent $OutputOnnx) "paddlex_onnx"
    if (Test-Path -LiteralPath $onnxDir) {
        Remove-Item -LiteralPath $onnxDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force $onnxDir | Out-Null

    $versionProbe = Invoke-Probe -FilePath $Python -Arguments @("-c", "import sys,paddle,paddlex,onnx; import importlib.metadata as m; print('python', sys.version); print('paddle', paddle.__version__); print('paddlex', getattr(paddlex, '__version__', 'unknown')); print('paddle2onnx', m.version('paddle2onnx')); print('onnx', onnx.__version__)")
    if ($versionProbe.exitCode -ne 0) {
        Write-BlockedSummary -SummaryPath $SummaryPath -Stage "conversion-environment" -Reason "Cannot import required PaddleX/Paddle2ONNX modules." -Details $versionProbe
        throw "Phase 47 conversion environment is blocked. See $SummaryPath"
    }
    $nativeProbe = Invoke-Probe -FilePath $Python -Arguments @("-c", "import paddle2onnx; print('paddle2onnx-native-ok')")

    $convertProbe = Invoke-Probe -FilePath $Python -Arguments @(
        "-m", "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir", $InferenceDir,
        "--onnx_model_dir", $onnxDir,
        "--opset_version", "7"
    )
    foreach ($line in $convertProbe.output) {
        Write-Host $line
    }
    if ($convertProbe.exitCode -ne 0) {
        Write-BlockedSummary -SummaryPath $SummaryPath -Stage "paddlex-paddle2onnx" -Reason "PaddleX official paddle2onnx conversion failed." -Details ([ordered]@{
            versions = $versionProbe
            paddle2onnxNativeImport = $nativeProbe
            conversion = $convertProbe
            pipIndexPaddle2Onnx = Invoke-Probe -FilePath $Python -Arguments @("-m", "pip", "index", "versions", "paddle2onnx")
        })
        throw "Phase 47 PaddleX conversion failed. See $SummaryPath"
    }

    $generated = Get-ChildItem -LiteralPath $onnxDir -Filter "*.onnx" -Recurse | Select-Object -First 1
    if ($null -eq $generated) {
        Write-BlockedSummary -SummaryPath $SummaryPath -Stage "paddlex-paddle2onnx" -Reason "PaddleX conversion completed without producing an ONNX file." -Details ([ordered]@{
            versions = $versionProbe
            paddle2onnxNativeImport = $nativeProbe
            conversion = $convertProbe
            outputDirectory = $onnxDir
        })
        throw "Phase 47 PaddleX conversion produced no ONNX file. See $SummaryPath"
    }

    New-Item -ItemType Directory -Force (Split-Path -Parent $OutputOnnx) | Out-Null
    Copy-Item -LiteralPath $generated.FullName -Destination $OutputOnnx -Force
    Assert-PathExists $OutputOnnx "converted PaddleOCR Det ONNX"
}

function Write-AITrainSidecar {
    param(
        [string]$OnnxPath,
        [string]$Phase31ReportPath,
        [string]$SidecarPath
    )

    $phase31Report = Get-Content -Raw -Encoding UTF8 -LiteralPath $Phase31ReportPath | ConvertFrom-Json
    $sidecar = [ordered]@{
        ok = $true
        backend = "paddleocr_det_official"
        modelFamily = "ocr_detection"
        phase = "47"
        format = "onnx"
        source = "PaddleOCR official Det inference model converted with PaddleX paddle2onnx"
        sourceReportPath = [System.IO.Path]::GetFullPath($Phase31ReportPath)
        sourceInferenceModelDir = [string]$phase31Report.inferenceModelDir
        onnxPath = [System.IO.Path]::GetFullPath($OnnxPath)
        paddleOcrResolvedRef = [string]$phase31Report.paddleOcrResolvedRef
        conversion = [ordered]@{
            pythonExecutable = $conversionPythonExe
            paddlepaddleRequirement = $PaddlePaddleRequirement
            paddlepaddleIndexUrl = if ($UseStablePaddleForConversion) { "" } else { $PaddlePaddleIndexUrl }
            paddlexRequirement = $PaddleXRequirement
            paddle2onnxRequirement = $Paddle2OnnxRequirement
            opsetVersion = 7
        }
        limitation = "Real exported Det ONNX wiring smoke; tiny CPU model validates integration artifacts, not OCR accuracy."
    }
    Write-JsonFile -Path $SidecarPath -Value $sidecar
}

function Invoke-CtestForWorkDir {
    param([string]$WorkRoot)

    if ($SkipCtest) {
        Write-Host "  [skip] CTest skipped by -SkipCtest"
        return "skipped"
    }

    $ctestFile = Join-Path $script:Root "$BuildDir\CTestTestfile.cmake"
    if (!(Test-Path -LiteralPath $ctestFile)) {
        Write-Host "  [warn] CTest build directory not found; skipping C++ regression check." -ForegroundColor Yellow
        return "skipped-build-dir-missing"
    }

    $previous = $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT
    $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $WorkRoot
    try {
        Invoke-Checked -FilePath "ctest" -Arguments @("--test-dir", (Join-Path $script:Root $BuildDir), "--output-on-failure", "--timeout", "360")
        return "passed"
    } finally {
        if ($null -eq $previous) {
            Remove-Item Env:\AITRAIN_ACCEPTANCE_SMOKE_ROOT -ErrorAction SilentlyContinue
        } else {
            $env:AITRAIN_ACCEPTANCE_SMOKE_ROOT = $previous
        }
    }
}

$work = Resolve-RepoPath $WorkDir
$phase31Root = Resolve-RepoPath $Phase31WorkDir
$pythonExe = Join-Path (Resolve-RepoPath $PythonDir) "python.exe"
$paddleRepo = Resolve-RepoPath $PaddleOcrRepo
$conversionPythonDirFull = Resolve-RepoPath $ConversionPythonDir
$conversionPythonZipFull = Resolve-RepoPath $ConversionPythonZip
$conversionPythonExe = Join-Path $conversionPythonDirFull "python.exe"
$detRun = Join-Path $phase31Root "runs\paddleocr_det_official"
$detInferenceDir = Join-Path $detRun "official_inference"
$detReportPath = Join-Path $detRun "paddleocr_official_det_report.json"
$defaultImage = Join-Path $phase31Root "paddleocr_det\images\a.png"
$sampleImage = if ($ImagePath) { Resolve-RepoPath $ImagePath } else { $defaultImage }
$onnxPath = Join-Path $work "paddleocr_det_official.onnx"
$sidecarPath = "$onnxPath.aitrain-export.json"
$workerOutput = Join-Path $work "cpp_onnx_smoke"
$summaryPath = Join-Path $work "paddleocr_det_onnx_smoke_summary.json"

New-Item -ItemType Directory -Force $work | Out-Null
Assert-PathExists $pythonExe "isolated OCR Python"
Assert-PathExists $paddleRepo "PaddleOCR repo"

try {
    Ensure-Phase31DetArtifacts -Phase31Root $phase31Root
    Assert-PathExists $sampleImage "OCR Det smoke image"

    $conversionPythonExe = Ensure-EmbeddedPython -PythonDirPath $conversionPythonDirFull -ZipPath $conversionPythonZipFull -DownloadUrl $ConversionPythonUrl
    Ensure-Pip -Python $conversionPythonExe
    Ensure-ConversionModules -Python $conversionPythonExe

    Convert-DetInferenceToOnnx -Python $conversionPythonExe -InferenceDir $detInferenceDir -OutputOnnx $onnxPath -SummaryPath $summaryPath
    Write-AITrainSidecar -OnnxPath $onnxPath -Phase31ReportPath $detReportPath -SidecarPath $sidecarPath

    $workerExe = Get-WorkerExe
    $workerJsonLines = Invoke-Checked -FilePath $workerExe -Arguments @(
        "--ocr-det-onnx-smoke", $onnxPath,
        "--image", $sampleImage,
        "--output", $workerOutput,
        "--binary-threshold", "0.01",
        "--box-threshold", "0.0",
        "--min-area", "1",
        "--max-detections", "100"
    )
    $workerResult = ($workerJsonLines | Where-Object { $_ -match '^\s*\{' } | Select-Object -Last 1) | ConvertFrom-Json
    if (-not $workerResult.ok) {
        throw "Worker OCR Det ONNX smoke reported ok=false."
    }
    Assert-PathExists ([string]$workerResult.predictionsPath) "C++ OCR Det predictions JSON"
    Assert-PathExists ([string]$workerResult.overlayPath) "C++ OCR Det overlay"

    $ctestStatus = Invoke-CtestForWorkDir -WorkRoot $work
    $finishedAt = [DateTime]::UtcNow
    $summary = [ordered]@{
        ok = $true
        phase = "47"
        status = "passed"
        workDir = $work
        startedAt = $script:StartedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
        pythonExecutable = $pythonExe
        conversionPythonExecutable = $conversionPythonExe
        paddleOcrRepoPath = $paddleRepo
        phase31WorkDir = $phase31Root
        artifacts = [ordered]@{
            onnxPath = [System.IO.Path]::GetFullPath($onnxPath)
            sidecarPath = [System.IO.Path]::GetFullPath($sidecarPath)
            sourceDetReportPath = [System.IO.Path]::GetFullPath($detReportPath)
            sourceDetInferenceDir = [System.IO.Path]::GetFullPath($detInferenceDir)
            predictionsPath = [System.IO.Path]::GetFullPath([string]$workerResult.predictionsPath)
            overlayPath = [System.IO.Path]::GetFullPath([string]$workerResult.overlayPath)
        }
        conversion = [ordered]@{
            tool = "paddlex --paddle2onnx"
            paddlepaddleRequirement = $PaddlePaddleRequirement
            paddlepaddleIndexUrl = if ($UseStablePaddleForConversion) { "" } else { $PaddlePaddleIndexUrl }
            paddlexRequirement = $PaddleXRequirement
            paddle2onnxRequirement = $Paddle2OnnxRequirement
            opsetVersion = 7
        }
        ctestStatus = $ctestStatus
        workerResult = $workerResult
        note = "Phase 47 validates real exported PaddleOCR Det ONNX wiring through C++ ONNX Runtime DB-style postprocess on a tiny CPU smoke model; it is not PP-OCRv5 accuracy parity."
    }
    Write-JsonFile -Path $summaryPath -Value $summary
    Write-Host ("  [ok] summary={0}" -f $summaryPath)
    Write-Host "Phase47 PaddleOCR Det ONNX smoke passed" -ForegroundColor Green
} catch {
    if (!(Test-Path -LiteralPath $summaryPath)) {
        Write-BlockedSummary -SummaryPath $summaryPath -Stage "phase47" -Reason ([string]$_.Exception.Message) -Details ([ordered]@{
            exception = [string]$_.Exception
        })
    }
    throw
}
