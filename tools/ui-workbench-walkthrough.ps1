param(
    [string]$AppPath = ".\build-vscode\bin\AITrainStudio.exe",
    [string]$WorkingDirectory = ".\build-vscode\bin",
    [string]$OutDir = ".deps\ui-walkthrough-rc",
    [int]$Width = 1280,
    [int]$Height = 820,
    [string]$WalkthroughScript = (Join-Path $env:USERPROFILE ".codex\skills\qt-gui-walkthrough\scripts\qt_walkthrough.ps1")
)

$ErrorActionPreference = "Stop"

function New-UiLabel {
    param([int[]]$CodePoints)

    $chars = foreach ($point in $CodePoints) {
        [char]$point
    }
    return -join $chars
}

function Write-RcSummary {
    param(
        [string]$Status,
        [string]$Message,
        [string]$ErrorCode = "",
        [object]$Details = @{}
    )

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $summary = [ordered]@{
        kind = "ui_workbench_walkthrough_rc_summary"
        status = $Status
        errorCode = $ErrorCode
        message = $Message
        appPath = $AppPath
        workingDirectory = $WorkingDirectory
        outDir = $OutDir
        width = $Width
        height = $Height
        pages = $script:Pages
        details = $Details
        createdAt = (Get-Date).ToUniversalTime().ToString("o")
    }
    $summaryPath = Join-Path $OutDir "ui_walkthrough_rc_summary.json"
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $summaryPath
    Write-Host "UI walkthrough RC summary: $summaryPath"
}

$script:Pages = @(
    (New-UiLabel @(0x603B, 0x89C8)),
    (New-UiLabel @(0x9879, 0x76EE)),
    (New-UiLabel @(0x6570, 0x636E, 0x96C6)),
    (New-UiLabel @(0x6837, 0x672C, 0x590D, 0x6838)),
    (New-UiLabel @(0x8BAD, 0x7EC3, 0x5B9E, 0x9A8C)),
    (New-UiLabel @(0x4EFB, 0x52A1, 0x4E0E, 0x4EA7, 0x7269)),
    (New-UiLabel @(0x6A21, 0x578B, 0x5E93)),
    (New-UiLabel @(0x8BC4, 0x4F30, 0x62A5, 0x544A)),
    (New-UiLabel @(0x6A21, 0x578B, 0x5BFC, 0x51FA)),
    (New-UiLabel @(0x63A8, 0x7406, 0x9A8C, 0x8BC1)),
    (New-UiLabel @(0x4EA4, 0x4ED8, 0x9A8C, 0x6536)),
    (New-UiLabel @(0x63D2, 0x4EF6)),
    (New-UiLabel @(0x73AF, 0x5883)),
    (New-UiLabel @(0x8BBE, 0x7F6E))
)

$script:EnglishPages = @(
    "Overview",
    "Project",
    "Datasets",
    "Sample Review",
    "Training Runs",
    "Tasks and Artifacts",
    "Model Library",
    "Evaluation Reports",
    "Model Export",
    "Inference Check",
    "Delivery Acceptance",
    "Plugins",
    "Environment",
    "Settings"
)

$script:EnglishMixedPages = @(
    "Overview",
    "Project",
    "Datasets",
    (New-UiLabel @(0x6837, 0x672C, 0x590D, 0x6838)),
    "Training Runs",
    "Tasks and Artifacts",
    "Model Library",
    "Evaluation Reports",
    "Model Export",
    "Inference Check",
    (New-UiLabel @(0x4EA4, 0x4ED8, 0x9A8C, 0x6536)),
    "Plugins",
    "Environment",
    "Settings"
)

function Get-AppProcessName {
    if ([string]::IsNullOrWhiteSpace($AppPath)) {
        return "AITrainStudio"
    }
    return [System.IO.Path]::GetFileNameWithoutExtension($AppPath)
}

function Get-AppMainWindowTitle {
    $name = Get-AppProcessName
    $process = Get-Process -Name $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $process) {
        return ""
    }
    return $process.MainWindowTitle
}

function Stop-AppProcess {
    $name = Get-AppProcessName
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Test-LicenseWindowTitle {
    param([string]$Title)

    if ([string]::IsNullOrWhiteSpace($Title)) {
        return $false
    }
    $registrationText = New-UiLabel @(0x6CE8, 0x518C)
    return $Title.Contains("Registration") -or $Title.Contains($registrationText)
}

if (-not (Test-Path -LiteralPath $AppPath)) {
    Write-RcSummary -Status "blocked" -ErrorCode "app_missing" -Message "AITrainStudio.exe was not found. Build the app before running the UI walkthrough gate."
    exit 2
}

if (-not (Test-Path -LiteralPath $WorkingDirectory)) {
    Write-RcSummary -Status "blocked" -ErrorCode "working_directory_missing" -Message "The configured working directory does not exist."
    exit 2
}

if (-not (Test-Path -LiteralPath $WalkthroughScript)) {
    Write-RcSummary -Status "blocked" -ErrorCode "walkthrough_script_missing" -Message "qt-gui-walkthrough script is not installed in the Codex skill directory."
    exit 2
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$attempts = @(
    [ordered]@{ name = "zh_CN"; pages = $script:Pages },
    [ordered]@{ name = "en_US"; pages = $script:EnglishPages },
    [ordered]@{ name = "en_US_mixed_untranslated_uiText"; pages = $script:EnglishMixedPages }
)

$attemptResults = @()
$walkthroughSummaryCandidates = @(
    (Join-Path $OutDir "walkthrough-summary.json"),
    (Join-Path $OutDir "walkthrough_summary.json")
)

foreach ($attempt in $attempts) {
    $walkthroughExit = 0
    $walkthroughError = ""

    try {
        & $WalkthroughScript `
            -AppPath $AppPath `
            -WorkingDirectory $WorkingDirectory `
            -OutDir $OutDir `
            -PageNames $attempt.pages `
            -Width $Width `
            -Height $Height
        $walkthroughExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
        $walkthroughExit = 1
        $walkthroughError = $_.Exception.Message
    }

    $walkthroughSummaryPath = @($walkthroughSummaryCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
    $walkthroughSummaryPath = if ($walkthroughSummaryPath.Count -gt 0) { [string]$walkthroughSummaryPath[0] } else { "" }
    $windowTitle = Get-AppMainWindowTitle
    $attemptResults += [ordered]@{
        labelSet = $attempt.name
        exitCode = $walkthroughExit
        error = $walkthroughError
        activeWindowTitle = $windowTitle
        walkthroughSummaryPath = $walkthroughSummaryPath
    }

    if ($walkthroughExit -eq 0) {
        $details = [ordered]@{
            walkthroughScript = $WalkthroughScript
            walkthroughExitCode = $walkthroughExit
            walkthroughSummaryPath = $walkthroughSummaryPath
            selectedLabelSet = $attempt.name
            attempts = $attemptResults
        }
        Write-RcSummary -Status "passed" -Message "Qt GUI walkthrough completed for the RC 1280x820 page set." -Details $details
        exit 0
    }

    if (Test-LicenseWindowTitle $windowTitle) {
        $details = [ordered]@{
            walkthroughScript = $WalkthroughScript
            walkthroughExitCode = $walkthroughExit
            walkthroughError = $walkthroughError
            activeWindowTitle = $windowTitle
            initialScreenshot = (Join-Path $OutDir "00-initial.png")
            attempts = $attemptResults
        }
        Stop-AppProcess
        Write-RcSummary -Status "blocked" -ErrorCode "license_required" -Message "AITrain Studio opened the offline license registration dialog before the workbench. Configure a valid license token and build-time public key before running the GUI walkthrough gate." -Details $details
        exit 2
    }

    Stop-AppProcess
}

$lastAttempt = $attemptResults | Select-Object -Last 1
$details = [ordered]@{
    walkthroughScript = $WalkthroughScript
    walkthroughExitCode = $lastAttempt.exitCode
    walkthroughError = $lastAttempt.error
    activeWindowTitle = $lastAttempt.activeWindowTitle
    attempts = $attemptResults
}
Write-RcSummary -Status "failed" -ErrorCode "walkthrough_failed" -Message "Qt GUI walkthrough did not complete for any supported workbench label set." -Details $details
exit 1
