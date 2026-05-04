param(
    [string]$WorkDir = ".deps\production-ocr-acceptance",
    [string]$DetDataset = "",
    [string]$RecDataset = "",
    [string]$SystemImages = "",
    [string]$OfficialDetReport = "",
    [string]$OfficialRecReport = "",
    [string]$OfficialSystemReport = "",
    [string]$OcrDetOnnxSummary = "",
    [int]$MinimumDetImages = 100,
    [int]$MinimumRecSamples = 1000,
    [int]$MinimumSystemImages = 100,
    [double]$MinimumRecAccuracy = 0.90,
    [double]$MaximumRecCer = 0.10,
    [switch]$RequireDetOnnxEvidence
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow

function Resolve-RepoPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Count-Images {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return 0
    }
    return @(Get-ChildItem -LiteralPath $Path -Recurse -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff -ErrorAction SilentlyContinue).Count
}

function Count-RecSamples {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return 0
    }
    $labelCandidates = @("rec_gt_test.txt", "rec_gt_val.txt", "rec_gt.txt", "train.txt", "val.txt", "test.txt") |
        ForEach-Object { Join-Path $Path $_ } |
        Where-Object { Test-Path -LiteralPath $_ }
    if ($labelCandidates.Count -eq 0) {
        return 0
    }
    $count = 0
    foreach ($label in $labelCandidates) {
        $count += @(Get-Content -LiteralPath $label -Encoding UTF8 | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
    }
    return $count
}

function Read-Json {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or !(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
}

function Get-JsonNumber {
    param(
        [object]$Object,
        [string[]]$Names
    )
    if ($null -eq $Object) {
        return $null
    }
    foreach ($name in $Names) {
        if ($Object.PSObject.Properties.Name -contains $name) {
            $value = $Object.$name
            if ($null -ne $value -and "$value" -ne "") {
                return [double]$value
            }
        }
    }
    if ($Object.PSObject.Properties.Name -contains "metrics") {
        return Get-JsonNumber -Object $Object.metrics -Names $Names
    }
    return $null
}

$work = Resolve-RepoPath $WorkDir
$reportPath = Join-Path $work "production_ocr_acceptance_report.json"
$summaryPath = Join-Path $work "production_ocr_acceptance_summary.md"
New-Item -ItemType Directory -Force $work | Out-Null

$detDatasetPath = Resolve-RepoPath $DetDataset
$recDatasetPath = Resolve-RepoPath $RecDataset
$systemImagesPath = Resolve-RepoPath $SystemImages
$officialDetReportPath = Resolve-RepoPath $OfficialDetReport
$officialRecReportPath = Resolve-RepoPath $OfficialRecReport
$officialSystemReportPath = Resolve-RepoPath $OfficialSystemReport
$ocrDetOnnxSummaryPath = Resolve-RepoPath $OcrDetOnnxSummary

$checks = @()
function Add-Check {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Message,
        [object]$Evidence = $null
    )
    $script:checks += [ordered]@{
        name = $Name
        status = $Status
        message = $Message
        evidence = $Evidence
    }
}

if ($detDatasetPath) {
    $detImages = Count-Images -Path $detDatasetPath
    Add-Check "det_dataset_size" ($(if ($detImages -ge $MinimumDetImages) { "passed" } else { "blocked" })) `
        "PaddleOCR Det production acceptance requires at least $MinimumDetImages images." `
        ([ordered]@{ path = $detDatasetPath; imageCount = $detImages; minimum = $MinimumDetImages })
} else {
    Add-Check "det_dataset_size" "blocked" "Missing -DetDataset for production OCR acceptance."
}

if ($recDatasetPath) {
    $recSamples = Count-RecSamples -Path $recDatasetPath
    Add-Check "rec_dataset_size" ($(if ($recSamples -ge $MinimumRecSamples) { "passed" } else { "blocked" })) `
        "PaddleOCR Rec production acceptance requires at least $MinimumRecSamples labeled samples." `
        ([ordered]@{ path = $recDatasetPath; sampleCount = $recSamples; minimum = $MinimumRecSamples })
} else {
    Add-Check "rec_dataset_size" "blocked" "Missing -RecDataset for production OCR acceptance."
}

if ($systemImagesPath) {
    $systemImagesCount = Count-Images -Path $systemImagesPath
    Add-Check "system_image_size" ($(if ($systemImagesCount -ge $MinimumSystemImages) { "passed" } else { "blocked" })) `
        "PaddleOCR System production acceptance requires at least $MinimumSystemImages end-to-end images." `
        ([ordered]@{ path = $systemImagesPath; imageCount = $systemImagesCount; minimum = $MinimumSystemImages })
} else {
    Add-Check "system_image_size" "blocked" "Missing -SystemImages for production OCR acceptance."
}

$detReport = Read-Json -Path $officialDetReportPath
Add-Check "official_det_report" ($(if ($null -ne $detReport) { "passed" } else { "blocked" })) `
    "Official PaddleOCR Det train/export/evaluation report must be supplied." `
    ([ordered]@{ path = $officialDetReportPath })

$recReport = Read-Json -Path $officialRecReportPath
$recAccuracy = Get-JsonNumber -Object $recReport -Names @("accuracy", "acc")
$recCer = Get-JsonNumber -Object $recReport -Names @("cer", "CER")
$recStatus = "blocked"
if ($null -ne $recReport -and $null -ne $recAccuracy -and $null -ne $recCer -and $recAccuracy -ge $MinimumRecAccuracy -and $recCer -le $MaximumRecCer) {
    $recStatus = "passed"
}
Add-Check "official_rec_metrics" $recStatus `
    "Official PaddleOCR Rec report must meet accuracy >= $MinimumRecAccuracy and CER <= $MaximumRecCer." `
    ([ordered]@{ path = $officialRecReportPath; accuracy = $recAccuracy; cer = $recCer; minimumAccuracy = $MinimumRecAccuracy; maximumCer = $MaximumRecCer })

$systemReport = Read-Json -Path $officialSystemReportPath
Add-Check "official_system_report" ($(if ($null -ne $systemReport) { "passed" } else { "blocked" })) `
    "Official PaddleOCR System prediction/evaluation report must be supplied for end-to-end OCR acceptance." `
    ([ordered]@{ path = $officialSystemReportPath })

if ($RequireDetOnnxEvidence) {
    $detOnnxSummary = Read-Json -Path $ocrDetOnnxSummaryPath
    $detOnnxPassed = $null -ne $detOnnxSummary -and
        ($detOnnxSummary.PSObject.Properties.Name -contains "ok") -and
        $detOnnxSummary.ok -and
        ($detOnnxSummary.PSObject.Properties.Name -contains "status") -and
        $detOnnxSummary.status -eq "passed"
    Add-Check "ocr_det_onnx_cpp_smoke" ($(if ($detOnnxPassed) { "passed" } else { "blocked" })) `
        "C++ OCR Det ONNX smoke evidence is required when -RequireDetOnnxEvidence is set." `
        ([ordered]@{ path = $ocrDetOnnxSummaryPath; status = if ($null -ne $detOnnxSummary) { $detOnnxSummary.status } else { "" } })
} else {
    Add-Check "ocr_det_onnx_cpp_smoke" "warning" "C++ Det ONNX smoke was not required for this production OCR acceptance run."
}

$blocked = @($checks | Where-Object { $_.status -eq "blocked" })
$warnings = @($checks | Where-Object { $_.status -eq "warning" })
$finishedAt = [DateTime]::UtcNow
$report = [ordered]@{
    ok = ($blocked.Count -eq 0)
    status = if ($blocked.Count -eq 0) { "passed" } else { "blocked" }
    scope = "production_ocr_acceptance"
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    workDir = $work
    thresholds = [ordered]@{
        minimumDetImages = $MinimumDetImages
        minimumRecSamples = $MinimumRecSamples
        minimumSystemImages = $MinimumSystemImages
        minimumRecAccuracy = $MinimumRecAccuracy
        maximumRecCer = $MaximumRecCer
        requireDetOnnxEvidence = [bool]$RequireDetOnnxEvidence
    }
    checks = $checks
    blockedCount = $blocked.Count
    warningCount = $warnings.Count
    note = "This gate is for production OCR evidence. Tiny smoke artifacts must not be used to claim production OCR readiness."
}
Write-JsonFile -Path $reportPath -Value $report

$lines = @(
    "# Production OCR Acceptance Summary"
    ""
    ("- Status: {0}" -f $report.status)
    ("- Report: {0}" -f $reportPath)
    ("- Blocked checks: {0}" -f $blocked.Count)
    ("- Warning checks: {0}" -f $warnings.Count)
    ""
    "## Checks"
    ""
)
foreach ($check in $checks) {
    $lines += ("- {0}: {1} - {2}" -f $check.status, $check.name, $check.message)
}
$lines += @(
    "",
    "Tiny Phase 31/46/47 smoke artifacts validate wiring only. Production readiness requires representative data, official OCR reports, and explicit returned evidence."
)
$lines | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host "Production OCR acceptance report: $reportPath" -ForegroundColor Cyan
Write-Host "Production OCR acceptance summary: $summaryPath" -ForegroundColor Cyan
if ($blocked.Count -gt 0) {
    Write-Host "Production OCR acceptance blocked: $($blocked.Count) missing or failed evidence checks." -ForegroundColor Yellow
    exit 2
}
Write-Host "Production OCR acceptance passed." -ForegroundColor Green
