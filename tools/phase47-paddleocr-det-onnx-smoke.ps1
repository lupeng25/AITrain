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

$root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$work = if ([System.IO.Path]::IsPathRooted($WorkDir)) {
    [System.IO.Path]::GetFullPath($WorkDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $WorkDir))
}
$summaryPath = Join-Path $work "paddleocr_det_onnx_smoke_summary.json"
New-Item -ItemType Directory -Force -Path $work | Out-Null

$summary = [ordered]@{
    ok = $false
    phase = "47"
    status = "blocked"
    stage = "official-only"
    workDir = $work
    summaryPath = $summaryPath
    ignoredInputs = [ordered]@{
        phase31WorkDir = $Phase31WorkDir
        pythonDir = $PythonDir
        paddleOcrRepo = $PaddleOcrRepo
        conversionPythonDir = $ConversionPythonDir
        conversionPythonUrl = $ConversionPythonUrl
        conversionPythonZip = $ConversionPythonZip
        paddlePaddleRequirement = $PaddlePaddleRequirement
        paddlePaddleIndexUrl = $PaddlePaddleIndexUrl
        paddleXRequirement = $PaddleXRequirement
        paddle2OnnxRequirement = $Paddle2OnnxRequirement
        buildDir = $BuildDir
        imagePath = $ImagePath
        useStablePaddleForConversion = [bool]$UseStablePaddleForConversion
        skipInstall = [bool]$SkipInstall
        skipPhase31 = [bool]$SkipPhase31
        skipCtest = [bool]$SkipCtest
    }
    reason = "Phase 47 OCR Det ONNX smoke is historical wiring evidence only. AITrain OCR product routes are official-only through PaddleOCR Det/Rec/System reports."
    nextAction = "Run tools\\phase31-paddleocr-full-official-smoke.ps1, tools\\production-ocr-acceptance.ps1, or tools\\customer-ocr-validation.ps1 with official PaddleOCR reports."
}

$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host ("Phase47 PaddleOCR Det ONNX smoke is blocked by OCR official-only policy. Summary: {0}" -f $summaryPath) -ForegroundColor Yellow
exit 11
