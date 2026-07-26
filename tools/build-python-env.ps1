param(
    [Parameter(Mandatory)]
    [ValidateSet("yolo", "smp_semantic_segmentation", "anomaly_detection", "ocr")]
    [string]$Profile,
    [Parameter(Mandatory)]
    [string]$Destination,
    [string]$BootstrapPython = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$profileFiles = @{
    yolo = @("requirements-yolo.txt", "locks\yolo-windows.txt")
    smp_semantic_segmentation = @("requirements-smp.txt", "locks\smp_semantic_segmentation-windows.txt")
    anomaly_detection = @("requirements-anomaly.txt", "locks\anomaly_detection-windows.txt")
    ocr = @("requirements-ocr.txt", "locks\ocr-windows.txt")
}

$destinationFull = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destinationFull) {
    throw "Python 环境目标已存在，请指定空路径：$destinationFull"
}

$requirementsRoot = Join-Path $root "python_trainers"
$requirements = Join-Path $requirementsRoot $profileFiles[$Profile][0]
$constraints = Join-Path $requirementsRoot $profileFiles[$Profile][1]
& $BootstrapPython -m venv $destinationFull
if ($LASTEXITCODE -ne 0) {
    throw "创建 Python 环境失败。"
}
$python = Join-Path $destinationFull "Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "升级 pip 失败。"
}
& $python -m pip install -r $requirements -c $constraints
if ($LASTEXITCODE -ne 0) {
    throw "安装 $Profile Profile 依赖失败。"
}

Write-Host "Python Profile 已构建：$Profile -> $destinationFull" -ForegroundColor Green
