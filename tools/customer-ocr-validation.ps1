param(
    [Parameter(Mandatory = $true)]
    [string]$CustomerDataset,

    [string]$OutputDir = ".deps\customer-ocr-validation",

    [string]$DetReport = "",
    [string]$RecReport = "",
    [string]$SystemReport = "",

    [double]$MinRecAccuracy = 0.70,
    [double]$MaxRecCer = 0.30,

    [switch]$AllowPublicLikeData,
    [switch]$AllowBlocked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-FullPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Read-JsonObject {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-ObjectValue {
    param(
        [object]$Object,
        [string]$Path
    )
    $current = $Object
    foreach ($segment in $Path.Split(".")) {
        if ($null -eq $current) {
            return $null
        }
        $property = $current.PSObject.Properties | Where-Object { $_.Name -eq $segment } | Select-Object -First 1
        if ($null -eq $property) {
            return $null
        }
        $current = $property.Value
    }
    return $current
}

function Get-FirstNumber {
    param(
        [object]$Object,
        [string[]]$Paths
    )
    foreach ($path in $Paths) {
        $value = Get-ObjectValue -Object $Object -Path $path
        if ($null -ne $value -and "$value" -ne "") {
            return [double]$value
        }
    }
    return $null
}

function New-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Message
    )
    [pscustomobject]@{
        name = $Name
        passed = $Passed
        message = $Message
    }
}

$datasetPath = Resolve-FullPath $CustomerDataset
$outputPath = Resolve-FullPath $OutputDir
$detReportPath = Resolve-FullPath $DetReport
$recReportPath = Resolve-FullPath $RecReport
$systemReportPath = Resolve-FullPath $SystemReport

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$datasetExists = Test-Path -LiteralPath $datasetPath
$normalizedDataset = $datasetPath.Replace("\", "/").ToLowerInvariant()
$publicLikeData = $normalizedDataset.Contains("/.deps/production-ocr-data/") `
    -or $normalizedDataset.Contains("/examples/") `
    -or $normalizedDataset.Contains("/generated/")

$detReport = Read-JsonObject $detReportPath
$recReport = Read-JsonObject $recReportPath
$systemReport = Read-JsonObject $systemReportPath

$recAccuracy = Get-FirstNumber -Object $recReport -Paths @(
    "metrics.accuracy",
    "accuracy",
    "bestAccuracy",
    "best_accuracy",
    "eval.accuracy"
)
$recCer = Get-FirstNumber -Object $recReport -Paths @(
    "metrics.cer",
    "cer",
    "CER",
    "eval.cer"
)
if ($null -eq $recAccuracy -and $null -ne $recReport) {
    $metricsObject = Get-ObjectValue -Object $recReport -Path "metrics"
    if ($null -ne $metricsObject) {
        $recAccuracy = Get-ObjectValue -Object $metricsObject -Path "accuracy"
    }
}
if ($null -eq $recCer -and $null -ne $recReport) {
    $metricsObject = Get-ObjectValue -Object $recReport -Path "metrics"
    if ($null -ne $metricsObject) {
        $recCer = Get-ObjectValue -Object $metricsObject -Path "cer"
    }
}
if ($null -eq $recAccuracy -and $null -ne $recReport) {
    try { $recAccuracy = [double]$recReport.metrics.accuracy } catch { }
}
if ($null -eq $recCer -and $null -ne $recReport) {
    try { $recCer = [double]$recReport.metrics.cer } catch { }
}
if (($null -eq $recAccuracy -or $null -eq $recCer) -and (Test-Path -LiteralPath $recReportPath)) {
    $recReportRaw = Get-Content -LiteralPath $recReportPath -Raw -Encoding UTF8
    if ($null -eq $recAccuracy -and $recReportRaw -match '"accuracy"\s*:\s*([0-9]+(?:\.[0-9]+)?)') {
        $recAccuracy = [double]$Matches[1]
    }
    if ($null -eq $recCer -and $recReportRaw -match '"cer"\s*:\s*([0-9]+(?:\.[0-9]+)?)') {
        $recCer = [double]$Matches[1]
    }
    if ($null -eq $recCer -and $recReportRaw -match '"CER"\s*:\s*([0-9]+(?:\.[0-9]+)?)') {
        $recCer = [double]$Matches[1]
    }
}

$checks = @()
$checks += New-Check "customer_dataset_present" $datasetExists "Customer dataset directory must exist."
$checks += New-Check "customer_dataset_not_public_like" (-not $publicLikeData -or $AllowPublicLikeData.IsPresent) "Customer validation must use customer-domain data, not public/generated smoke data."
$checks += New-Check "det_report_present" ($null -ne $detReport) "Detection report must be attached."
$checks += New-Check "rec_report_present" ($null -ne $recReport) "Recognition report must be attached."
$checks += New-Check "system_report_present" ($null -ne $systemReport) "End-to-end system report must be attached."
$checks += New-Check "rec_accuracy_threshold" ($null -ne $recAccuracy -and $recAccuracy -ge $MinRecAccuracy) "Recognition accuracy must meet the configured customer-domain threshold."
$checks += New-Check "rec_cer_threshold" ($null -ne $recCer -and $recCer -le $MaxRecCer) "Recognition CER must meet the configured customer-domain threshold."

$failedChecks = @($checks | Where-Object { -not $_.passed })
$status = if ($failedChecks.Count -eq 0) { "passed" } else { "blocked" }

$manifest = [ordered]@{
    schemaVersion = 1
    kind = "customer_ocr_validation_manifest"
    createdAt = (Get-Date).ToUniversalTime().ToString("o")
    status = $status
    customerDataset = $datasetPath
    publicLikeDataDetected = $publicLikeData
    thresholds = [ordered]@{
        minRecAccuracy = $MinRecAccuracy
        maxRecCer = $MaxRecCer
    }
    evidence = [ordered]@{
        detReportPath = $detReportPath
        recReportPath = $recReportPath
        systemReportPath = $systemReportPath
    }
    metrics = [ordered]@{
        recAccuracy = $recAccuracy
        recCer = $recCer
    }
    checks = $checks
    failureCount = $failedChecks.Count
    failures = $failedChecks
    note = "Customer-domain OCR validation is separate from public Total-Text or generated smoke evidence."
}

$manifestPath = Join-Path $outputPath "customer_ocr_validation_manifest.json"
$summaryPath = Join-Path $outputPath "customer_ocr_validation_summary.md"
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$summary = @(
    "# Customer OCR Validation Summary",
    "",
    "- Status: $status",
    "- Customer dataset: $datasetPath",
    "- Rec accuracy: $recAccuracy",
    "- Rec CER: $recCer",
    "- Failure count: $($failedChecks.Count)",
    "",
    "## Evidence",
    "",
    "- Det report: $detReportPath",
    "- Rec report: $recReportPath",
    "- System report: $systemReportPath",
    "",
    "## Boundary",
    "",
    "This gate is for customer-domain evidence. Public or generated smoke data remains useful for wiring checks only."
)
$summary -join "`n" | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Host ("Customer OCR validation status={0}" -f $status)
Write-Host ("  manifest={0}" -f $manifestPath)
Write-Host ("  summary={0}" -f $summaryPath)

if ($status -ne "passed" -and -not $AllowBlocked.IsPresent) {
    throw "Customer OCR validation blocked. See $manifestPath"
}
