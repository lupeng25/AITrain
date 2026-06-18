param(
    [string]$WorkDir = ".deps\xanylabeling-user-flow",
    [string]$WorkerExe = "build-vscode\bin\aitrain_worker.exe",
    [string]$XAnyLabelingExe = "",
    [switch]$UseRealTool,
    [switch]$FakeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = $script:Utf8NoBom
try {
    [Console]::OutputEncoding = $script:Utf8NoBom
    [Console]::InputEncoding = $script:Utf8NoBom
} catch {
}

$script:Root = Split-Path -Parent $PSScriptRoot
Set-Location $script:Root
$script:Lanes = @()
$script:SummaryPath = $null

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $script:Root $Path))
}

function New-UnicodeText {
    param([int[]]$CodePoints)
    return (-join ($CodePoints | ForEach-Object { [string][char]$_ }))
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-Utf8NoBomText {
    param([string]$Path, [string]$Value)
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Encoding UTF8 -Raw | ConvertFrom-Json
}

function Read-LastJsonLine {
    param([string]$Text)
    $lines = @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    for ($index = $lines.Count - 1; $index -ge 0; --$index) {
        $line = $lines[$index].Trim()
        if ($line.StartsWith("{") -and $line.EndsWith("}")) {
            try {
                return ($line | ConvertFrom-Json)
            } catch {
            }
        }
    }
    return $null
}

function Invoke-LoggedCommand {
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
        $output = @(& $File @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $lines = @($output | ForEach-Object { [string]$_ })
    $text = $lines -join [Environment]::NewLine
    Set-Content -LiteralPath $LogPath -Value $text -Encoding UTF8
    foreach ($line in $lines) {
        Write-Host $line
    }
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "Command failed with exit code $exitCode`: $File $($Arguments -join ' ')"
    }
    return [pscustomobject]@{
        exitCode = $exitCode
        text = $text
        json = Read-LastJsonLine -Text $text
        logPath = $LogPath
        commandLine = "$File $($Arguments -join ' ')"
    }
}

function Add-Lane {
    param([object]$Lane)
    $script:Lanes += $Lane
}

function New-Lane {
    param(
        [string]$Id,
        [string]$Status,
        [string]$FailureReason = "",
        [object[]]$Commands = @(),
        [object[]]$ReportPaths = @(),
        [object[]]$Artifacts = @(),
        [object]$Details = $null
    )
    $lane = [ordered]@{
        id = $Id
        status = $Status
        commands = @($Commands)
        reportPaths = @($ReportPaths)
        artifacts = @($Artifacts)
    }
    if (-not [string]::IsNullOrWhiteSpace($FailureReason)) {
        $lane.failureReason = $FailureReason
    }
    if ($null -ne $Details) {
        $lane.details = $Details
    }
    return [pscustomobject]$lane
}

function Resolve-Worker {
    $resolved = Resolve-RepoPath $WorkerExe
    if (-not (Test-Path -LiteralPath $resolved)) {
        throw "aitrain_worker.exe was not found: $resolved. Run .\tools\harness-check.ps1 or pass -WorkerExe."
    }
    return $resolved
}

function Resolve-RealXAnyLabeling {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($XAnyLabelingExe)) {
        $candidates.Add((Resolve-RepoPath $XAnyLabelingExe))
    }
    if (-not [string]::IsNullOrWhiteSpace($env:AITRAIN_XANYLABELING_EXE)) {
        $candidates.Add($env:AITRAIN_XANYLABELING_EXE)
    }
    $candidates.Add((Resolve-RepoPath ".deps\tools\annotation-tools\X-AnyLabeling\X-AnyLabeling.exe"))
    $candidates.Add((Resolve-RepoPath ".deps\annotation-tools\X-AnyLabeling\X-AnyLabeling.exe"))
    $pathCommand = Get-Command "xanylabeling" -ErrorAction SilentlyContinue
    if ($pathCommand) {
        $candidates.Add($pathCommand.Source)
    }
    $pathExe = Get-Command "X-AnyLabeling.exe" -ErrorAction SilentlyContinue
    if ($pathExe) {
        $candidates.Add($pathExe.Source)
    }
    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    return ""
}

function New-SmokeImage {
    param(
        [string]$Path,
        [string]$Kind,
        [int]$Index = 0
    )
    Add-Type -AssemblyName System.Drawing -ErrorAction Stop
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $bitmap = New-Object System.Drawing.Bitmap 128, 96
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::FromArgb(244, 246, 248))
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(55, 120, 200))
    if ($Kind -eq "segmentation") {
        $points = @(
            (New-Object System.Drawing.Point 20, 18),
            (New-Object System.Drawing.Point 105, 20),
            (New-Object System.Drawing.Point 96, 76),
            (New-Object System.Drawing.Point 22, 82)
        )
        $graphics.FillPolygon($brush, $points)
    } elseif ($Kind -eq "obb") {
        $points = @(
            (New-Object System.Drawing.Point 35, 22),
            (New-Object System.Drawing.Point 103, 35),
            (New-Object System.Drawing.Point 94, 72),
            (New-Object System.Drawing.Point 28, 59)
        )
        $graphics.FillPolygon($brush, $points)
    } else {
        $graphics.FillRectangle($brush, 32 + ($Index * 4), 24, 58, 42)
    }
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(20, 20, 20)), 2
    $graphics.DrawRectangle($pen, 4, 4, 120, 88)
    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Jpeg)
    $pen.Dispose()
    $brush.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

function New-YoloDataset {
    param(
        [string]$Root,
        [string]$Format
    )
    $kind = switch ($Format) {
        "yolo_segmentation" { "segmentation" }
        "yolo_obb" { "obb" }
        default { "detection" }
    }
    $imageDir = Join-Path $Root "images\train"
    $labelDir = Join-Path $Root "labels\train"
    New-Item -ItemType Directory -Force -Path $imageDir, $labelDir | Out-Null
    for ($i = 0; $i -lt 2; ++$i) {
        $name = "sample_$i"
        New-SmokeImage -Path (Join-Path $imageDir "$name.jpg") -Kind $kind -Index $i
        $label = switch ($Format) {
            "yolo_segmentation" { "0 0.16 0.19 0.82 0.20 0.75 0.79 0.17 0.85" }
            "yolo_obb" { "0 0.27 0.23 0.80 0.36 0.73 0.75 0.22 0.62" }
            default { "0 0.50 0.48 0.45 0.42" }
        }
        Write-Utf8NoBomText -Path (Join-Path $labelDir "$name.txt") -Value ($label + "`n")
    }
    $yaml = @"
path: .
train: images/train
val: images/train
nc: 2
names: [widget, part]
"@
    Write-Utf8NoBomText -Path (Join-Path $Root "data.yaml") -Value ($yaml + "`n")
}

function New-UnicodeYoloDetectionDataset {
    param([string]$Root)
    $imageDir = Join-Path $Root "images\train"
    $labelDir = Join-Path $Root "labels\train"
    [System.IO.Directory]::CreateDirectory($imageDir) | Out-Null
    [System.IO.Directory]::CreateDirectory($labelDir) | Out-Null
    $className = New-UnicodeText -CodePoints 0x96F6, 0x4EF6
    $sampleWord = New-UnicodeText -CodePoints 0x6837, 0x672C
    $name = "$className $sampleWord 1"
    New-SmokeImage -Path (Join-Path $imageDir "$name.jpg") -Kind "detection"
    Write-Utf8NoBomText -Path (Join-Path $labelDir "$name.txt") -Value "0 0.50 0.48 0.45 0.42`n"
    $yaml = @"
path: .
train: images/train
val: images/train
nc: 1
names: [$className]
"@
    Write-Utf8NoBomText -Path (Join-Path $Root "data.yaml") -Value ($yaml + "`n")
}

function New-FakeXAnyLabelingCli {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $script = @'
@echo off
setlocal EnableExtensions EnableDelayedExpansion
if /I "%~1"=="version" (
  echo fake-xanylabeling 1.0
  exit /b 0
)
if /I "%~1"=="checks" (
  echo ok
  exit /b 0
)
set "TASK="
set "MODE="
set "IMAGES="
set "LABELS="
set "OUTPUT="
set "CLASSES="
if /I "%~1"=="convert" shift
:parse
if "%~1"=="" goto run
if /I "%~1"=="--task" (
  set "TASK=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--mode" (
  set "MODE=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--images" (
  set "IMAGES=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--labels" (
  set "LABELS=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--output" (
  set "OUTPUT=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--classes" (
  set "CLASSES=%~2"
  shift
  shift
  goto parse
)
shift
goto parse
:run
if "%OUTPUT%"=="" exit /b 2
if not exist "%OUTPUT%" mkdir "%OUTPUT%"
if /I "%TASK%"=="yolo2xlabel" goto yolo2xlabel
if /I "%TASK%"=="xlabel2yolo" goto xlabel2yolo
exit /b 2
:yolo2xlabel
if "%IMAGES%"=="" exit /b 3
pushd "%IMAGES%" || exit /b 3
for %%F in (*.jpg *.jpeg *.png *.bmp) do (
  set "BASE=%%~nxF"
  set "NAME=%%~nF"
  >"%OUTPUT%\!NAME!.json" echo {"version":"fake","imagePath":"!BASE!","imageHeight":96,"imageWidth":128,"shapes":[]}
)
popd
exit /b 0
:xlabel2yolo
if "%LABELS%"=="" exit /b 4
if not exist "%OUTPUT%\labels" mkdir "%OUTPUT%\labels"
if not exist "%OUTPUT%\images" mkdir "%OUTPUT%\images"
> "%OUTPUT%\data.yaml" echo path: .
>> "%OUTPUT%\data.yaml" echo train: images
>> "%OUTPUT%\data.yaml" echo val: images
>> "%OUTPUT%\data.yaml" echo nc: 1
>> "%OUTPUT%\data.yaml" echo names: [widget]
for %%F in ("%LABELS%\*.json") do (
  >"%OUTPUT%\labels\%%~nF.txt" echo 0 0.5 0.5 0.25 0.25
)
exit /b 0
'@
    Set-Content -LiteralPath $Path -Value $script -Encoding ASCII
    return [System.IO.Path]::GetFullPath($Path)
}

function Invoke-WorkerRequest {
    param(
        [string]$Worker,
        [string]$Option,
        [object]$Request,
        [string]$RequestPath,
        [string]$LogPath,
        [switch]$AllowFailure
    )
    Write-JsonFile -Path $RequestPath -Value $Request
    return Invoke-LoggedCommand -File $Worker -Arguments @($Option, $RequestPath) -LogPath $LogPath -AllowFailure:$AllowFailure
}

function Test-JsonArrayContains {
    param([object]$Array, [string]$Value)
    foreach ($item in @($Array)) {
        if ([string]$item -eq $Value) {
            return $true
        }
    }
    return $false
}

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw $Message
    }
}

function Test-AnnotationSessionArtifacts {
    param([object]$Result)
    Assert-Condition ($Result.ok -eq $true) "prepareAnnotationSession returned ok=false"
    Assert-Condition (Test-Path -LiteralPath $Result.manifestPath) "Session manifest was not written."
    Assert-Condition (Test-Path -LiteralPath $Result.launchRequestPath) "Launch request was not written."
    Assert-Condition (Test-Path -LiteralPath $Result.reviewSamplesPath) "Review samples were not written."
    if (-not [string]::IsNullOrWhiteSpace([string]$Result.classesPath)) {
        Assert-Condition (Test-Path -LiteralPath $Result.classesPath) "classes.txt was not written."
    }
    $launch = Read-JsonFile -Path $Result.launchRequestPath
    Assert-Condition (Test-JsonArrayContains -Array $launch.arguments -Value "--filename") "Launch request is missing --filename."
    Assert-Condition (Test-JsonArrayContains -Array $launch.arguments -Value "--output") "Launch request is missing --output."
    Assert-Condition (Test-JsonArrayContains -Array $launch.arguments -Value "--labels") "Launch request is missing --labels."
}

function Add-SimulatedXLabelOutput {
    param([string]$SessionOutputPath)
    New-SmokeImage -Path (Join-Path $SessionOutputPath "images\review_sample.jpg") -Kind "detection"
    $label = @{
        version = "simulated-user-edit"
        imagePath = "images/review_sample.jpg"
        imageHeight = 96
        imageWidth = 128
        shapes = @(@{
            label = "widget"
            shape_type = "rectangle"
            points = @(@(32, 24), @(90, 66))
        })
    }
    Write-JsonFile -Path (Join-Path $SessionOutputPath "review_sample.json") -Value $label
}

function Test-ConversionReport {
    param([object]$Result, [string]$ExpectedSource, [string]$ExpectedTarget)
    Assert-Condition ($Result.ok -eq $true) "Conversion $ExpectedSource -> $ExpectedTarget returned ok=false: $($Result.errorCode) $($Result.errorMessage)"
    Assert-Condition (Test-Path -LiteralPath $Result.outputPath) "Conversion output path does not exist: $($Result.outputPath)"
    Assert-Condition (Test-Path -LiteralPath $Result.reportPath) "Conversion report was not written: $($Result.reportPath)"
    $report = Read-JsonFile -Path $Result.reportPath
    Assert-Condition ($report.conversionEngine -eq "xanylabeling_cli") "Report conversionEngine is not xanylabeling_cli."
    Assert-Condition (-not [string]::IsNullOrWhiteSpace([string]$report.resolvedImagesPath)) "Report is missing resolvedImagesPath."
    Assert-Condition (-not [string]::IsNullOrWhiteSpace([string]$report.resolvedLabelsPath)) "Report is missing resolvedLabelsPath."
    Assert-Condition ($report.process.status -eq "ok") "X-AnyLabeling process status is not ok."
    return $report
}

function Test-XLabelImagePaths {
    param(
        [string]$XLabelRoot,
        [switch]$RequireShapes
    )
    $jsonFiles = @(Get-ChildItem -LiteralPath $XLabelRoot -Filter "*.json" -File | Where-Object { $_.Name -notlike "*report*.json" })
    Assert-Condition ($jsonFiles.Count -gt 0) "YOLO -> XLABEL produced no label JSON files."
    foreach ($file in $jsonFiles) {
        $label = Read-JsonFile -Path $file.FullName
        $imagePath = [string]$label.imagePath
        Assert-Condition (-not [string]::IsNullOrWhiteSpace($imagePath)) "XLABEL file is missing imagePath: $($file.FullName)"
        $resolved = [System.IO.Path]::GetFullPath((Join-Path $XLabelRoot $imagePath))
        Assert-Condition (Test-Path -LiteralPath $resolved) "XLABEL imagePath cannot be opened: $resolved"
        if ($RequireShapes) {
            $shapes = @($label.shapes)
            Assert-Condition ($shapes.Count -gt 0) "Real YOLO -> XLABEL produced no shapes: $($file.FullName)"
            $firstShape = $shapes[0]
            Assert-Condition (-not [string]::IsNullOrWhiteSpace([string]$firstShape.label)) "Real XLABEL shape is missing label: $($file.FullName)"
            Assert-Condition (@($firstShape.points).Count -ge 2) "Real XLABEL shape has too few points: $($file.FullName)"
        }
    }
}

function Test-YoloLabelOutputs {
    param(
        [string]$YoloRoot,
        [string]$Format
    )
    $labelsRoot = Join-Path $YoloRoot "labels"
    if (-not (Test-Path -LiteralPath $labelsRoot)) {
        $labelsRoot = $YoloRoot
    }
    $labelFiles = @(Get-ChildItem -LiteralPath $labelsRoot -Recurse -Filter "*.txt" -File | Where-Object { $_.Name -ne "classes.txt" })
    Assert-Condition ($labelFiles.Count -gt 0) "XLABEL -> YOLO produced no label .txt files in $labelsRoot."
    $minimumTokenCount = switch ($Format) {
        "yolo_segmentation" { 7 }
        "yolo_obb" { 9 }
        default { 5 }
    }
    foreach ($file in $labelFiles) {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
        Assert-Condition (-not ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)) "YOLO label has UTF-8 BOM: $($file.FullName)"
        $text = ([System.Text.Encoding]::UTF8.GetString($bytes)).Trim()
        Assert-Condition (-not [string]::IsNullOrWhiteSpace($text)) "YOLO label is empty: $($file.FullName)"
        $firstLine = @($text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })[0]
        $tokens = @($firstLine -split "\s+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        Assert-Condition ($tokens.Count -ge $minimumTokenCount) "YOLO label token count $($tokens.Count) is too small for ${Format}: $($file.FullName)"
    }
}

function Invoke-ConversionLane {
    param(
        [string]$LaneId,
        [string]$Worker,
        [string]$Executable,
        [string]$OutputRoot,
        [object]$Datasets
    )
    $commands = @()
    $reports = @()
    $artifacts = @()
    try {
        $requireRealSemantics = $LaneId.EndsWith("_real")
        $formats = @("yolo_detection", "yolo_segmentation", "yolo_obb")
        foreach ($format in $formats) {
            $formatRoot = Join-Path $OutputRoot $format
            $yoloToXLabel = Invoke-WorkerRequest `
                -Worker $Worker `
                -Option "--dataset-conversion-request" `
                -Request @{
                    sourcePath = $Datasets[$format]
                    sourceFormat = $format
                    targetFormat = "xanylabeling_xlabel"
                    outputPath = (Join-Path $formatRoot "xlabel")
                    options = @{
                        conversionEngine = "xanylabeling_cli"
                        xAnyLabelingExecutable = $Executable
                        disableXAnyLabelingAutoDiscovery = $true
                        timeoutMs = 60000
                    }
                } `
                -RequestPath (Join-Path $formatRoot "requests\yolo_to_xlabel.json") `
                -LogPath (Join-Path $formatRoot "logs\yolo_to_xlabel.log") `
                -AllowFailure
            $commands += $yoloToXLabel.commandLine
            $xlabelResult = $yoloToXLabel.json
            $xlabelReport = Test-ConversionReport -Result $xlabelResult -ExpectedSource $format -ExpectedTarget "xanylabeling_xlabel"
            $reports += $xlabelResult.reportPath
            $artifacts += $xlabelResult.outputPath
            Test-XLabelImagePaths -XLabelRoot $xlabelResult.outputPath -RequireShapes:$requireRealSemantics

            $xLabelToYolo = Invoke-WorkerRequest `
                -Worker $Worker `
                -Option "--dataset-conversion-request" `
                -Request @{
                    sourcePath = $xlabelResult.outputPath
                    sourceFormat = "xanylabeling_xlabel"
                    targetFormat = $format
                    outputPath = (Join-Path $formatRoot "roundtrip_yolo")
                    options = @{
                        conversionEngine = "xanylabeling_cli"
                        xAnyLabelingExecutable = $Executable
                        disableXAnyLabelingAutoDiscovery = $true
                        timeoutMs = 60000
                    }
                } `
                -RequestPath (Join-Path $formatRoot "requests\xlabel_to_yolo.json") `
                -LogPath (Join-Path $formatRoot "logs\xlabel_to_yolo.log") `
                -AllowFailure
            $commands += $xLabelToYolo.commandLine
            $yoloResult = $xLabelToYolo.json
            $yoloReport = Test-ConversionReport -Result $yoloResult -ExpectedSource "xanylabeling_xlabel" -ExpectedTarget $format
            $reports += $yoloResult.reportPath
            $artifacts += $yoloResult.outputPath
            if ($requireRealSemantics) {
                Test-YoloLabelOutputs -YoloRoot $yoloResult.outputPath -Format $format
            }
            $null = $xlabelReport
            $null = $yoloReport
        }
        Add-Lane (New-Lane -Id $LaneId -Status "passed" -Commands $commands -ReportPaths $reports -Artifacts $artifacts)
    } catch {
        Add-Lane (New-Lane -Id $LaneId -Status "failed" -FailureReason $_.Exception.Message -Commands $commands -ReportPaths $reports -Artifacts $artifacts)
    }
}

function Invoke-UnicodePathLane {
    param(
        [string]$Worker,
        [string]$Executable,
        [string]$OutputRoot
    )
    $commands = @()
    $reports = @()
    $artifacts = @()
    try {
        $zhInput = New-UnicodeText -CodePoints 0x8F93, 0x5165
        $zhDataset = New-UnicodeText -CodePoints 0x6570, 0x636E, 0x96C6
        $zhChinese = New-UnicodeText -CodePoints 0x4E2D, 0x6587
        $zhOutput = New-UnicodeText -CodePoints 0x8F93, 0x51FA
        $zhRoundTrip = New-UnicodeText -CodePoints 0x56DE, 0x8F6C
        $datasetPath = Join-Path $OutputRoot "$zhInput $zhDataset $zhChinese"
        New-UnicodeYoloDetectionDataset -Root $datasetPath

        $yoloToXLabel = Invoke-WorkerRequest `
            -Worker $Worker `
            -Option "--dataset-conversion-request" `
            -Request @{
                sourcePath = $datasetPath
                sourceFormat = "yolo_detection"
                targetFormat = "xanylabeling_xlabel"
                outputPath = (Join-Path $OutputRoot "$zhOutput XLABEL $zhChinese")
                options = @{
                    conversionEngine = "xanylabeling_cli"
                    xAnyLabelingExecutable = $Executable
                    disableXAnyLabelingAutoDiscovery = $true
                    timeoutMs = 60000
                }
            } `
            -RequestPath (Join-Path $OutputRoot "requests\yolo_to_xlabel_unicode.json") `
            -LogPath (Join-Path $OutputRoot "logs\yolo_to_xlabel_unicode.log") `
            -AllowFailure
        $commands += $yoloToXLabel.commandLine
        $xlabelResult = $yoloToXLabel.json
        $null = Test-ConversionReport -Result $xlabelResult -ExpectedSource "yolo_detection" -ExpectedTarget "xanylabeling_xlabel"
        $reports += $xlabelResult.reportPath
        $artifacts += $xlabelResult.outputPath
        Test-XLabelImagePaths -XLabelRoot $xlabelResult.outputPath -RequireShapes

        $xLabelToYolo = Invoke-WorkerRequest `
            -Worker $Worker `
            -Option "--dataset-conversion-request" `
            -Request @{
                sourcePath = $xlabelResult.outputPath
                sourceFormat = "xanylabeling_xlabel"
                targetFormat = "yolo_detection"
                outputPath = (Join-Path $OutputRoot "$zhRoundTrip YOLO $zhChinese")
                options = @{
                    conversionEngine = "xanylabeling_cli"
                    xAnyLabelingExecutable = $Executable
                    disableXAnyLabelingAutoDiscovery = $true
                    timeoutMs = 60000
                }
            } `
            -RequestPath (Join-Path $OutputRoot "requests\xlabel_to_yolo_unicode.json") `
            -LogPath (Join-Path $OutputRoot "logs\xlabel_to_yolo_unicode.log") `
            -AllowFailure
        $commands += $xLabelToYolo.commandLine
        $yoloResult = $xLabelToYolo.json
        $null = Test-ConversionReport -Result $yoloResult -ExpectedSource "xanylabeling_xlabel" -ExpectedTarget "yolo_detection"
        $reports += $yoloResult.reportPath
        $artifacts += $yoloResult.outputPath
        Test-YoloLabelOutputs -YoloRoot $yoloResult.outputPath -Format "yolo_detection"

        Add-Lane (New-Lane `
            -Id "unicode_space_path_conversion_real" `
            -Status "passed" `
            -Commands $commands `
            -ReportPaths $reports `
            -Artifacts $artifacts `
            -Details @{ datasetPath = $datasetPath })
    } catch {
        Add-Lane (New-Lane `
            -Id "unicode_space_path_conversion_real" `
            -Status "failed" `
            -FailureReason $_.Exception.Message `
            -Commands $commands `
            -ReportPaths $reports `
            -Artifacts $artifacts)
    }
}

function New-XLabelVariant {
    param(
        [string]$Root,
        [string]$ImagePathValue,
        [string]$ImageDiskPath
    )
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    Write-Utf8NoBomText -Path (Join-Path $Root "classes.txt") -Value "widget`n"
    New-SmokeImage -Path $ImageDiskPath -Kind "detection"
    $label = @{
        version = "simulated-user-export"
        imagePath = $ImagePathValue
        imageHeight = 96
        imageWidth = 128
        shapes = @(@{
            label = "widget"
            shape_type = "rectangle"
            points = @(@(32, 24), @(90, 66))
        })
    }
    Write-JsonFile -Path (Join-Path $Root "sample.json") -Value $label
}

function Invoke-VariantLane {
    param(
        [string]$Worker,
        [string]$Executable,
        [string]$OutputRoot
    )
    $commands = @()
    $reports = @()
    $artifacts = @()
    try {
        $variantRoot = Join-Path $OutputRoot "fixtures"
        $relativeRoot = Join-Path $variantRoot "relative_images"
        New-XLabelVariant -Root $relativeRoot -ImagePathValue "images/sample.jpg" -ImageDiskPath (Join-Path $relativeRoot "images\sample.jpg")
        $basenameRoot = Join-Path $variantRoot "basename_images_subdir"
        New-XLabelVariant -Root $basenameRoot -ImagePathValue "sample.jpg" -ImageDiskPath (Join-Path $basenameRoot "images\sample.jpg")
        $explicitRoot = Join-Path $variantRoot "explicit_images_path"
        $externalImages = Join-Path $variantRoot "external_images"
        New-XLabelVariant -Root $explicitRoot -ImagePathValue "sample.jpg" -ImageDiskPath (Join-Path $externalImages "sample.jpg")

        $variants = @(
            @{ id = "relative_images"; source = $relativeRoot; expectedImages = (Join-Path $relativeRoot "images"); options = @{} },
            @{ id = "basename_images_subdir"; source = $basenameRoot; expectedImages = (Join-Path $basenameRoot "images"); options = @{} },
            @{ id = "explicit_images_path"; source = $explicitRoot; expectedImages = $externalImages; options = @{ imagesPath = $externalImages } }
        )
        foreach ($variant in $variants) {
            $options = @{
                conversionEngine = "xanylabeling_cli"
                xAnyLabelingExecutable = $Executable
                disableXAnyLabelingAutoDiscovery = $true
                timeoutMs = 60000
            }
            foreach ($key in $variant.options.Keys) {
                $options[$key] = $variant.options[$key]
            }
            $variantOut = Join-Path $OutputRoot $variant.id
            $run = Invoke-WorkerRequest `
                -Worker $Worker `
                -Option "--dataset-conversion-request" `
                -Request @{
                    sourcePath = $variant.source
                    sourceFormat = "xanylabeling_xlabel"
                    targetFormat = "yolo_detection"
                    outputPath = $variantOut
                    options = $options
                } `
                -RequestPath (Join-Path $variantOut "request.json") `
                -LogPath (Join-Path $variantOut "worker.log") `
                -AllowFailure
            $commands += $run.commandLine
            $report = Test-ConversionReport -Result $run.json -ExpectedSource "xanylabeling_xlabel" -ExpectedTarget "yolo_detection"
            $resolved = [System.IO.Path]::GetFullPath([string]$report.resolvedImagesPath)
            $expected = [System.IO.Path]::GetFullPath([string]$variant.expectedImages)
            Assert-Condition ($resolved -eq $expected) "Variant $($variant.id) resolved images path '$resolved', expected '$expected'."
            $reports += $run.json.reportPath
            $artifacts += $run.json.outputPath
        }
        Add-Lane (New-Lane -Id "user_edited_xlabel_variants_fake" -Status "passed" -Commands $commands -ReportPaths $reports -Artifacts $artifacts)
    } catch {
        Add-Lane (New-Lane -Id "user_edited_xlabel_variants_fake" -Status "failed" -FailureReason $_.Exception.Message -Commands $commands -ReportPaths $reports -Artifacts $artifacts)
    }
}

function Invoke-AnnotationLane {
    param(
        [string]$LaneId,
        [string]$Worker,
        [string]$Executable,
        [string]$DatasetPath,
        [string]$OutputRoot,
        [bool]$TryLaunch
    )
    $commands = @()
    $reports = @()
    $artifacts = @()
    $guiLaunch = $null
    try {
        $prepare = Invoke-WorkerRequest `
            -Worker $Worker `
            -Option "--annotation-session-request" `
            -Request @{
                datasetPath = $DatasetPath
                outputPath = (Join-Path $OutputRoot "session")
                format = "yolo_detection"
                options = @{
                    xAnyLabelingExecutable = $Executable
                    disableXAnyLabelingAutoDiscovery = $true
                    mode = "quality_fix"
                }
            } `
            -RequestPath (Join-Path $OutputRoot "requests\prepare_session.json") `
            -LogPath (Join-Path $OutputRoot "logs\prepare_session.log") `
            -AllowFailure
        $commands += $prepare.commandLine
        Test-AnnotationSessionArtifacts -Result $prepare.json
        $reports += $prepare.json.manifestPath
        $artifacts += $prepare.json.launchRequestPath
        $artifacts += $prepare.json.reviewSamplesPath
        $artifacts += $prepare.json.classesPath

        if ($TryLaunch) {
            $launch = Read-JsonFile -Path $prepare.json.launchRequestPath
            try {
                $process = Start-Process -FilePath $launch.executable -ArgumentList @($launch.arguments) -WorkingDirectory $launch.workingDirectory -PassThru -WindowStyle Minimized
                Start-Sleep -Seconds 3
                $started = -not $process.HasExited
                if ($started) {
                    $null = $process.CloseMainWindow()
                    Start-Sleep -Seconds 1
                    if (-not $process.HasExited) {
                        Stop-Process -Id $process.Id -Force
                    }
                }
                $guiLaunch = [ordered]@{
                    status = $(if ($started -or $process.ExitCode -eq 0) { "started" } else { "failed" })
                    processId = $process.Id
                    terminatedBySmoke = $started
                }
            } catch {
                $guiLaunch = [ordered]@{
                    status = "failed"
                    error = $_.Exception.Message
                }
            }
        }

        Add-SimulatedXLabelOutput -SessionOutputPath $prepare.json.outputPath
        $sync = Invoke-WorkerRequest `
            -Worker $Worker `
            -Option "--annotation-sync-request" `
            -Request @{
                sessionManifestPath = $prepare.json.manifestPath
                datasetPath = $DatasetPath
                outputPath = (Join-Path $OutputRoot "sync")
                format = "yolo_detection"
                options = @{}
            } `
            -RequestPath (Join-Path $OutputRoot "requests\sync_session.json") `
            -LogPath (Join-Path $OutputRoot "logs\sync_session.log") `
            -AllowFailure
        $commands += $sync.commandLine
        Assert-Condition ($sync.json.ok -eq $true) "syncAnnotationSession returned ok=false: $($sync.json.error)"
        Assert-Condition ($sync.json.modifiedSessionOutputLabelCount -gt 0) "syncAnnotationSession did not scan simulated session output labels."
        $reports += $sync.json.reportPath

        $details = [ordered]@{}
        if ($null -ne $guiLaunch) {
            $details.guiLaunch = $guiLaunch
        }
        Add-Lane (New-Lane -Id $LaneId -Status "passed" -Commands $commands -ReportPaths $reports -Artifacts $artifacts -Details $details)
    } catch {
        $details = [ordered]@{}
        if ($null -ne $guiLaunch) {
            $details.guiLaunch = $guiLaunch
        }
        Add-Lane (New-Lane -Id $LaneId -Status "failed" -FailureReason $_.Exception.Message -Commands $commands -ReportPaths $reports -Artifacts $artifacts -Details $details)
    }
}

function Write-Summary {
    param(
        [string]$Path,
        [string]$Worker,
        [string]$RealExecutable,
        [string]$FakeExecutable
    )
    $failed = @($script:Lanes | Where-Object { $_.status -eq "failed" })
    $summary = [ordered]@{
        schemaVersion = 1
        kind = "xanylabeling_user_flow_summary"
        generatedAt = [DateTime]::UtcNow.ToString("o")
        workDir = [System.IO.Path]::GetFullPath((Split-Path -Parent $Path))
        workerExe = $Worker
        realTool = [ordered]@{
            executable = $RealExecutable
            requestedDetachedLaunch = [bool]$UseRealTool
            status = $(if ([string]::IsNullOrWhiteSpace($RealExecutable)) { "blocked" } else { "available" })
        }
        fakeTool = [ordered]@{
            executable = $FakeExecutable
            status = "available"
        }
        lanes = $script:Lanes
        status = $(if ($failed.Count -gt 0) { "failed" } else { "passed" })
        failedLaneCount = $failed.Count
    }
    Write-JsonFile -Path $Path -Value $summary
    return $summary
}

try {
    $worker = Resolve-Worker
    $work = Resolve-RepoPath $WorkDir
    if (Test-Path -LiteralPath $work) {
        Remove-Item -LiteralPath $work -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $script:SummaryPath = Join-Path $work "xanylabeling_user_flow_summary.json"

    $fixtures = Join-Path $work "fixtures"
    $datasets = [ordered]@{
        yolo_detection = (Join-Path $fixtures "yolo_detection")
        yolo_segmentation = (Join-Path $fixtures "yolo_segmentation")
        yolo_obb = (Join-Path $fixtures "yolo_obb")
    }
    foreach ($key in $datasets.Keys) {
        New-YoloDataset -Root $datasets[$key] -Format $key
    }

    $fakeExe = New-FakeXAnyLabelingCli -Path (Join-Path $work "tools\fake_xanylabeling.cmd")
    $realExe = Resolve-RealXAnyLabeling
    if ($FakeOnly) {
        $realExe = ""
    }

    if ($FakeOnly) {
        Add-Lane (New-Lane -Id "dependency_import_detection_real" -Status "blocked" -FailureReason "Skipped because -FakeOnly was requested.")
    } else {
        $dependencyCommands = @()
        $dependencyReports = @()
        try {
            $envRun = Invoke-WorkerRequest `
                -Worker $worker `
                -Option "--xany-environment-request" `
                -Request @{
                    outputPath = (Join-Path $work "environment")
                    options = @{}
                } `
                -RequestPath (Join-Path $work "environment\environment_request.json") `
                -LogPath (Join-Path $work "environment\environment_request.log") `
                -AllowFailure
            $dependencyCommands += $envRun.commandLine
            Assert-Condition ($null -ne $envRun.json) "Worker did not emit environment JSON."
            Assert-Condition (Test-Path -LiteralPath $envRun.json.reportPath) "environment_profiles_report.json was not written."
            Assert-Condition ($null -ne $envRun.json.profiles.xanylabeling) "X-AnyLabeling profile is missing."
            $dependencyReports += $envRun.json.reportPath
            $dependencyReports += $envRun.json.xanylabelingEnvironmentReportPath
            $status = [string]$envRun.json.profiles.xanylabeling.status
            $laneStatus = if ($status -eq "missing") { "blocked" } else { "passed" }
            Add-Lane (New-Lane `
                -Id "dependency_import_detection_real" `
                -Status $laneStatus `
                -Commands $dependencyCommands `
                -ReportPaths $dependencyReports `
                -Details @{
                    profileStatus = $status
                    executable = $envRun.json.xanylabeling.executable
                    candidates = $envRun.json.xanylabeling.candidates
                    licenseBoundary = $envRun.json.xanylabeling.licenseBoundary
                    redistributionReviewRequired = $envRun.json.xanylabeling.redistributionReviewRequired
                })
        } catch {
            Add-Lane (New-Lane -Id "dependency_import_detection_real" -Status "failed" -FailureReason $_.Exception.Message -Commands $dependencyCommands -ReportPaths $dependencyReports)
        }
    }

    $fakeEnvCommands = @()
    $fakeEnvReports = @()
    try {
        $fakeEnvRun = Invoke-WorkerRequest `
            -Worker $worker `
            -Option "--xany-environment-request" `
            -Request @{
                outputPath = (Join-Path $work "environment_fake")
                options = @{
                    xAnyLabelingExecutable = $fakeExe
                    disableXAnyLabelingAutoDiscovery = $true
                }
            } `
            -RequestPath (Join-Path $work "environment_fake\environment_request.json") `
            -LogPath (Join-Path $work "environment_fake\environment_request.log") `
            -AllowFailure
        $fakeEnvCommands += $fakeEnvRun.commandLine
        Assert-Condition ($fakeEnvRun.json.profiles.xanylabeling.status -eq "ok") "Fake X-AnyLabeling environment profile did not pass."
        $fakeEnvReports += $fakeEnvRun.json.reportPath
        $fakeEnvReports += $fakeEnvRun.json.xanylabelingEnvironmentReportPath
        Add-Lane (New-Lane -Id "dependency_import_detection_fake" -Status "passed" -Commands $fakeEnvCommands -ReportPaths $fakeEnvReports)
    } catch {
        Add-Lane (New-Lane -Id "dependency_import_detection_fake" -Status "failed" -FailureReason $_.Exception.Message -Commands $fakeEnvCommands -ReportPaths $fakeEnvReports)
    }

    Invoke-AnnotationLane `
        -LaneId "annotation_session_fake" `
        -Worker $worker `
        -Executable $fakeExe `
        -DatasetPath $datasets.yolo_detection `
        -OutputRoot (Join-Path $work "annotation_fake") `
        -TryLaunch $false

    if (-not [string]::IsNullOrWhiteSpace($realExe)) {
        Invoke-AnnotationLane `
            -LaneId "annotation_session_real" `
            -Worker $worker `
            -Executable $realExe `
            -DatasetPath $datasets.yolo_detection `
            -OutputRoot (Join-Path $work "annotation_real") `
            -TryLaunch ([bool]$UseRealTool)
    } else {
        Add-Lane (New-Lane -Id "annotation_session_real" -Status "blocked" -FailureReason "Real X-AnyLabeling executable was not found.")
    }

    Invoke-ConversionLane `
        -LaneId "import_conversion_flow_fake" `
        -Worker $worker `
        -Executable $fakeExe `
        -OutputRoot (Join-Path $work "conversion_fake") `
        -Datasets $datasets

    if (-not [string]::IsNullOrWhiteSpace($realExe)) {
        Invoke-ConversionLane `
            -LaneId "import_conversion_flow_real" `
            -Worker $worker `
            -Executable $realExe `
            -OutputRoot (Join-Path $work "conversion_real") `
            -Datasets $datasets

        Invoke-UnicodePathLane `
            -Worker $worker `
            -Executable $realExe `
            -OutputRoot (Join-Path $work "unicode_real")
    } else {
        Add-Lane (New-Lane -Id "import_conversion_flow_real" -Status "blocked" -FailureReason "Real X-AnyLabeling executable was not found.")
        Add-Lane (New-Lane -Id "unicode_space_path_conversion_real" -Status "blocked" -FailureReason "Real X-AnyLabeling executable was not found.")
    }

    Invoke-VariantLane `
        -Worker $worker `
        -Executable $fakeExe `
        -OutputRoot (Join-Path $work "xlabel_variants_fake")

    $summary = Write-Summary -Path $script:SummaryPath -Worker $worker -RealExecutable $realExe -FakeExecutable $fakeExe
    Write-Host "X-AnyLabeling user-flow smoke summary: $script:SummaryPath"
    if ($summary.status -ne "passed") {
        exit 1
    }
    exit 0
} catch {
    $workerForSummary = ""
    try {
        $workerForSummary = Resolve-Worker
    } catch {
    }
    $workForSummary = Resolve-RepoPath $WorkDir
    if ([string]::IsNullOrWhiteSpace($script:SummaryPath)) {
        $script:SummaryPath = Join-Path $workForSummary "xanylabeling_user_flow_summary.json"
    }
    Add-Lane (New-Lane -Id "script" -Status "failed" -FailureReason $_.Exception.Message)
    $null = Write-Summary -Path $script:SummaryPath -Worker $workerForSummary -RealExecutable "" -FakeExecutable ""
    Write-Error $_.Exception.Message
    exit 1
}
