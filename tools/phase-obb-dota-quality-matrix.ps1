param(
    [string]$WorkDir = ".deps\obb-quality\dota",
    [string]$Python = "",
    [string]$WorkerExe = "build-vscode\bin\aitrain_worker.exe",
    [string]$Dataset = "DOTA8",
    [string[]]$ModelPresets = @("yolo11n-obb.pt"),
    [string]$Device = "0",
    [int]$Epochs = 30,
    [int]$ImageSize = 640,
    [int]$BatchSize = 2,
    [switch]$SkipDownload,
    [switch]$RequirePublicDataset,
    [switch]$SkipProductRuntime
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Write-Json {
    param([string]$Path, [object]$Value)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-Json {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Encoding UTF8 -Raw | ConvertFrom-Json
}

function Invoke-Logged {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$LogPath,
        [switch]$AllowFailure
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $File @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $text = (@($output | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
    Set-Content -LiteralPath $LogPath -Value $text -Encoding UTF8
    foreach ($line in @($output)) {
        Write-Host $line
    }
    if ($exitCode -ne 0 -and !$AllowFailure) {
        throw "Command failed with exit code $exitCode`: $File $($Arguments -join ' ')"
    }
    return [pscustomobject]@{
        exitCode = $exitCode
        text = $text
        logPath = $LogPath
    }
}

function Dataset-YamlName {
    param([string]$Name)
    $trimmed = $Name.Trim()
    if ($trimmed.EndsWith(".yaml", [StringComparison]::OrdinalIgnoreCase) -or $trimmed.EndsWith(".yml", [StringComparison]::OrdinalIgnoreCase)) {
        return $trimmed
    }
    switch ($trimmed.ToLowerInvariant()) {
        "dota8" { return "DOTA8.yaml" }
        "dota128" { return "DOTA128.yaml" }
        "dotav1" { return "DOTAv1.yaml" }
        default { return "$trimmed.yaml" }
    }
}

function Safe-Name {
    param([string]$Value)
    return ($Value -replace "[^A-Za-z0-9_.-]", "_")
}

$work = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force -Path $work | Out-Null
$scriptPath = Resolve-RepoPath "tools\phase-obb-ultralytics-smoke.ps1"
$datasetYaml = Dataset-YamlName -Name $Dataset
$startedAt = [DateTime]::UtcNow
$rows = @()

foreach ($model in $ModelPresets) {
    $safe = Safe-Name -Value $model
    $rowDir = Join-Path $work "runs\$safe"
    $args = @(
        "-WorkDir", $rowDir,
        "-DatasetYaml", $datasetYaml,
        "-Model", $model,
        "-Device", $Device,
        "-Epochs", [string]$Epochs,
        "-ImageSize", [string]$ImageSize,
        "-BatchSize", [string]$BatchSize
    )
    if ($Python) {
        $args += @("-Python", (Resolve-RepoPath $Python))
    }
    if ($WorkerExe) {
        $args += @("-WorkerExe", (Resolve-RepoPath $WorkerExe))
    }
    if ($SkipDownload) {
        $args += "-SkipDownload"
    }
    if ($RequirePublicDataset) {
        $args += "-RequirePublicDataset"
    }
    if ($SkipProductRuntime) {
        $args += "-SkipProductRuntime"
    }

    $logPath = Join-Path $work "logs\$safe.log"
    $result = Invoke-Logged -File $scriptPath -Arguments $args -LogPath $logPath -AllowFailure
    $summaryPath = Join-Path $rowDir "obb_ultralytics_smoke_summary.json"
    $summary = Read-Json $summaryPath
    if ($null -eq $summary) {
        $summary = [pscustomobject]@{
            ok = $false
            status = "failed"
            artifacts = [pscustomobject]@{}
            metrics = [pscustomobject]@{}
            productRuntime = [pscustomobject]@{}
        }
    }
    $runtime = if ($summary.PSObject.Properties["productRuntime"]) { $summary.productRuntime } else { $null }
    $metrics = if ($summary.PSObject.Properties["metrics"]) { $summary.metrics } else { $null }
    $artifacts = if ($summary.PSObject.Properties["artifacts"]) { $summary.artifacts } else { $null }
    $benchmark = if ($runtime -and $runtime.PSObject.Properties["benchmark"]) { $runtime.benchmark } else { $null }
    $latency = $null
    if ($benchmark) {
        if ($benchmark.PSObject.Properties["latencyMsP95"]) {
            $latency = $benchmark.latencyMsP95
        } elseif ($benchmark.PSObject.Properties["p95Ms"]) {
            $latency = $benchmark.p95Ms
        } elseif ($benchmark.PSObject.Properties["latency"] -and $benchmark.latency.PSObject.Properties["p95Ms"]) {
            $latency = $benchmark.latency.p95Ms
        }
    }
    $rows += [pscustomobject]@{
        preset = $model
        status = [string]$summary.status
        ok = [bool]$summary.ok
        mAP50 = if ($metrics -and $metrics.PSObject.Properties["mAP50"]) { $metrics.mAP50 } else { $null }
        mAP50_95 = if ($metrics -and $metrics.PSObject.Properties["mAP50_95"]) { $metrics.mAP50_95 } else { $null }
        p95Ms = $latency
        bestPt = if ($artifacts -and $artifacts.PSObject.Properties["bestPt"]) { [string]$artifacts.bestPt } else { "" }
        bestOnnx = if ($artifacts -and $artifacts.PSObject.Properties["bestOnnx"]) { [string]$artifacts.bestOnnx } else { "" }
        trainingReportPath = if ($artifacts -and $artifacts.PSObject.Properties["trainingReportPath"]) { [string]$artifacts.trainingReportPath } else { "" }
        evaluationReportPath = if ($artifacts -and $artifacts.PSObject.Properties["evaluationReportPath"]) { [string]$artifacts.evaluationReportPath } else { "" }
        workerSmokeSummaryPath = if ($artifacts -and $artifacts.PSObject.Properties["workerSmokeSummaryPath"]) { [string]$artifacts.workerSmokeSummaryPath } else { "" }
        logPath = $logPath
        exitCode = $result.exitCode
        summaryPath = $summaryPath
    }
}

$passedRows = @($rows | Where-Object { $_.ok })
$ranked = @($passedRows | Sort-Object `
    @{Expression = { if ($null -eq $_.mAP50_95) { -1.0 } else { [double]$_.mAP50_95 } }; Descending = $true}, `
    @{Expression = { if ($null -eq $_.mAP50) { -1.0 } else { [double]$_.mAP50 } }; Descending = $true}, `
    @{Expression = { if ($null -eq $_.p95Ms) { [double]::PositiveInfinity } else { [double]$_.p95Ms } }; Ascending = $true})
$best = if ($ranked.Count -gt 0) { $ranked[0] } else { $null }
$summary = [ordered]@{
    ok = ($rows.Count -gt 0 -and $passedRows.Count -eq $rows.Count)
    status = if ($rows.Count -gt 0 -and $passedRows.Count -eq $rows.Count) { "passed" } else { "failed" }
    phase = "obb_dota_quality_matrix"
    startedAt = $startedAt.ToString("o")
    finishedAt = ([DateTime]::UtcNow).ToString("o")
    workDir = $work
    dataset = [ordered]@{
        requested = $Dataset
        yaml = $datasetYaml
        publicBenchmarkScope = "DOTA public workflow/subset when materialization succeeds; generated fallback is workflow smoke only."
    }
    parameters = [ordered]@{
        modelPresets = $ModelPresets
        epochs = $Epochs
        imageSize = $ImageSize
        batchSize = $BatchSize
        device = $Device
    }
    bestPreset = if ($best) { $best.preset } else { "" }
    results = $rows
    note = "This OBB matrix is public DOTA/workflow evidence only and must not be represented as customer-domain industrial accuracy."
}

$summaryPath = Join-Path $work "obb_dota_quality_matrix_summary.json"
$csvPath = Join-Path $work "obb_dota_model_comparison.csv"
$mdPath = Join-Path $work "obb_dota_model_comparison.md"
Write-Json -Path $summaryPath -Value $summary
$rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

$md = @()
$md += "# OBB DOTA Quality Matrix"
$md += ""
$md += "- Status: $($summary.status)"
$md += "- Dataset request: $Dataset ($datasetYaml)"
$md += "- Epochs: $Epochs"
$md += "- Image size: $ImageSize"
$md += "- Device: $Device"
$md += "- Best preset: $($summary.bestPreset)"
$md += ""
$md += "| Preset | Status | mAP50-95 | mAP50 | p95 ms | ONNX |"
$md += "|---|---:|---:|---:|---:|---|"
foreach ($row in $ranked + @($rows | Where-Object { -not $_.ok })) {
    $md += "| $($row.preset) | $($row.status) | $($row.mAP50_95) | $($row.mAP50) | $($row.p95Ms) | $($row.bestOnnx) |"
}
$md += ""
$md += "Scope: public DOTA/subset workflow evidence when materialization succeeds; generated fallback rows are smoke evidence only. This does not represent customer-domain industrial precision."
$md -join [Environment]::NewLine | Set-Content -LiteralPath $mdPath -Encoding UTF8

$summary | ConvertTo-Json -Depth 100
if (-not $summary["ok"]) {
    exit 1
}
