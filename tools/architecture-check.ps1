Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$scanRoots = @(
    "src",
    "tests",
    "python_trainers",
    "tools",
    ".vscode",
    "CMakeLists.txt"
)

$files = @()
foreach ($scanRoot in $scanRoots) {
    if (-not (Test-Path -LiteralPath $scanRoot)) {
        continue
    }
    $item = Get-Item -LiteralPath $scanRoot
    if ($item.PSIsContainer) {
        $files += Get-ChildItem -LiteralPath $scanRoot -Recurse -File |
            Where-Object { $_.Extension -in @('.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.py', '.ps1', '.cmake', '.md', '.json', '.txt', '.yml', '.yaml', '.vcxproj', '.sln', '.qrc') }
    } else {
        $files += $item
    }
}
$selfPath = [IO.Path]::GetFullPath($PSCommandPath)
$files = @($files | Where-Object { [IO.Path]::GetFullPath($_.FullName) -ne $selfPath })

function Assert-CleanPattern {
    param(
        [string]$Pattern,
        [string]$Message
    )

    $findings = @($files | Select-String -CaseSensitive -Pattern $Pattern)
    if ($findings.Count -gt 0) {
        $findings | Select-Object -First 20 | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        throw $Message
    }
}

Write-Host "Architecture check: removed legacy entry points" -ForegroundColor Cyan
foreach ($pattern in @(
        'aitrain_core',
        'aitrain_foundation',
        'ProjectRepository',
        'DetectionTrainingOptions',
        'DetectionTrainingMetrics',
        'DetectionTrainingResult',
        'resumeCheckpointPath',
        'tiny_linear_detector',
        'python_mock',
        'QPluginLoader',
        '(?<![A-Za-z0-9])startTrain(?![A-Za-z0-9])',
        'runCustomerOcrAcceptance',
        'semantic-onnx-smoke',
        'phase-p1-yolo-full-matrix-smoke')) {
    Assert-CleanPattern $pattern 'Removed legacy target, protocol, or script entry detected.'
}

Write-Host "Architecture check: dependency direction" -ForegroundColor Cyan
$coreCmake = Join-Path $root "src/core/CMakeLists.txt"
if (Test-Path -LiteralPath $coreCmake) {
    $content = Get-Content -LiteralPath $coreCmake -Encoding UTF8 -Raw
    $storageBlock = [regex]::Match($content, 'target_link_libraries\(aitrain_storage(?s:.*?)\)')
    $applicationBlock = [regex]::Match($content, 'target_link_libraries\(aitrain_application(?s:.*?)\)')
    $reverseDependency = $storageBlock.Success -and $storageBlock.Value -match 'aitrain_application'
    if ($reverseDependency) {
        throw 'Reverse application/storage dependency detected.'
    }
}

Write-Host "Architecture check passed." -ForegroundColor Green
