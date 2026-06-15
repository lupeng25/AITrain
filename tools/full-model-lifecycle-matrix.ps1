param(
    [string]$WorkDir = ".deps\full-model-lifecycle",
    [int]$Epochs = 100,
    [string]$Device = "0",
    [int]$ImageSize = 640,
    [int]$Batch = 1,
    [string[]]$DeploymentTargets = @("onnx", "tensorrt", "ncnn"),
    [bool]$Resume = $true,
    [int]$MaxParallel = 1,
    [switch]$DryRun,
    [switch]$Preflight,
    [switch]$SkipDataPrep,
    [switch]$SkipOcrInstall,
    [switch]$SkipOcrEnvironmentCreate,
    [string]$PythonExe = "",
    [string]$OcrPythonExe = "",
    [string]$OcrPythonDir = "",
    [string]$PaddleOcrRepo = ".deps\PaddleOCR",
    [string]$PaddleOcrRef = "v3.7.0",
    [string]$PaddlePaddleRequirement = "paddlepaddle-gpu==3.3.1",
    [string]$BuildDir = "build-vscode",
    [string]$WorkerExe = "",
    [string]$NcnnRoot = $env:AITRAIN_NCNN_ROOT,
    [string]$Yolo26MatrixSummary = ".deps\phase-yolo26-model-matrix\yolo26_model_matrix_summary.json",
    [switch]$AllowOcrCpu,
    [switch]$RerunFailed,
    [string]$RunId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:Root = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
$script:StartedAt = [DateTime]::UtcNow
$script:EffectiveEpochs = if ($Preflight) { 1 } else { $Epochs }
$script:OcrUseGpu = -not $AllowOcrCpu
$script:RunId = if ([string]::IsNullOrWhiteSpace($RunId)) { "run-" + $script:StartedAt.ToString("yyyyMMddTHHmmssZ") } else { $RunId.Trim() }
$script:WorkRoot = ""
$script:ControllerFailurePath = ""
$script:YoloCudaAvailable = $false

. (Join-Path $PSScriptRoot "toolchain-env.ps1")
Set-AITrainQtRuntimeEnvironment

function Write-Step {
    param([string]$Message)
    Write-Host "Full lifecycle: $Message" -ForegroundColor Cyan
}

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

function New-Utf8NoBomEncoding {
    return [System.Text.UTF8Encoding]::new($false)
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value,
        [int]$Depth = 80
    )
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force $parent | Out-Null
    }
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-ControllerFailure {
    param([object]$ErrorRecord)
    if ([string]::IsNullOrWhiteSpace($script:WorkRoot)) {
        return
    }
    $path = if ([string]::IsNullOrWhiteSpace($script:ControllerFailurePath)) {
        Join-Path $script:WorkRoot "controller_failure.json"
    } else {
        $script:ControllerFailurePath
    }
    try {
        $failure = [ordered]@{
            ok = $false
            status = "controller_failed"
            runId = $script:RunId
            startedAt = $script:StartedAt.ToString("o")
            failedAt = [DateTime]::UtcNow.ToString("o")
            workDir = $script:WorkRoot
            message = [string]$ErrorRecord.Exception.Message
            exceptionType = $ErrorRecord.Exception.GetType().FullName
            scriptStackTrace = [string]$ErrorRecord.ScriptStackTrace
            invocation = if ($ErrorRecord.InvocationInfo) { [string]$ErrorRecord.InvocationInfo.PositionMessage } else { "" }
            note = "Controller failed before writing the normal summary. External process logs are written separately and only bounded tails are kept in JSON."
        }
        Write-JsonFile -Path $path -Value $failure
        Write-JsonFile -Path (Join-Path $script:WorkRoot "full_model_lifecycle_summary.json") -Value $failure
    } catch {
        Write-Warning ("Failed to write controller failure report: {0}" -f $_.Exception.Message)
    }
}

trap {
    Write-ControllerFailure -ErrorRecord $_
    break
}

function Read-JsonFile {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
}

function Get-ObjectString {
    param(
        [object]$Object,
        [string]$Name,
        [string]$Default = ""
    )
    if ($null -eq $Object) {
        return $Default
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return $Default
    }
    return [string]$property.Value
}

function Update-RowSummaryRunId {
    param(
        [string]$Path,
        [object]$Summary
    )
    if ($null -eq $Summary) {
        return $Summary
    }
    $current = Get-ObjectString -Object $Summary -Name "runId"
    if ($current -ne $script:RunId) {
        Add-Member -InputObject $Summary -NotePropertyName "runId" -NotePropertyValue $script:RunId -Force
        Write-JsonFile -Path $Path -Value $Summary
    }
    return $Summary
}

function Get-LastJsonObjectFromText {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }
    $lines = $Text -split "(`r`n|`n|`r)"
    for ($index = $lines.Count - 1; $index -ge 0; $index--) {
        $line = [string]$lines[$index]
        if ($line -notmatch '^\s*\{') {
            continue
        }
        try {
            return $line | ConvertFrom-Json
        } catch {
            continue
        }
    }
    return $null
}

function Get-FileTailText {
    param(
        [string]$Path,
        [int]$TailLines = 200,
        [int]$MaxChars = 65536
    )
    if ([string]::IsNullOrWhiteSpace($Path) -or !(Test-Path -LiteralPath $Path)) {
        return ""
    }
    try {
        $lines = Get-Content -LiteralPath $Path -Encoding UTF8 -Tail $TailLines -ErrorAction Stop
        $text = ([string[]]$lines -join [Environment]::NewLine).Trim()
    } catch {
        return "[tail read failed] $($_.Exception.Message)"
    }
    if ($text.Length -gt $MaxChars) {
        return $text.Substring($text.Length - $MaxChars)
    }
    return $text
}

function Copy-FileStream {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (!(Test-Path -LiteralPath $Source)) {
        return
    }
    $inputStream = [System.IO.File]::Open($Source, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        $outputStream = [System.IO.File]::Open($Destination, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $outputStream.Dispose()
        }
    } finally {
        $inputStream.Dispose()
    }
}

function Quote-ProcessArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    $escaped = $Value -replace '"', '\"'
    return '"' + $escaped + '"'
}

function Join-ProcessArguments {
    param([string[]]$Arguments)
    return (($Arguments | ForEach-Object { Quote-ProcessArgument ([string]$_) }) -join " ")
}

function Invoke-ProcessCapture {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$LogPath = "",
        [string]$WorkingDirectory = $script:Root,
        [switch]$AllowFailure,
        [int]$TailLines = 200,
        [int]$TailChars = 65536
    )

    Write-Step ("{0}: {1} {2}" -f $Name, $FilePath, ($Arguments -join " "))
    $processLogRoot = if ($LogPath) { Split-Path -Parent $LogPath } elseif ($script:WorkRoot) { Join-Path $script:WorkRoot "logs" } else { [System.IO.Path]::GetTempPath() }
    if ($processLogRoot) {
        New-Item -ItemType Directory -Force $processLogRoot | Out-Null
    }
    $safeName = Convert-ToSafeName $Name
    $finalLog = if ($LogPath) { $LogPath } else { Join-Path $processLogRoot "$safeName.log" }
    New-Item -ItemType Directory -Force (Split-Path -Parent $finalLog) | Out-Null
    $stdoutPath = $finalLog
    $stderrPath = "$finalLog.stderr"
    Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    $resolvedFile = $FilePath
    $resolvedArgs = @($Arguments)
    if ([System.IO.Path]::GetExtension($FilePath) -ieq ".ps1") {
        $resolvedFile = "powershell.exe"
        $resolvedArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $FilePath) + $Arguments
    }
    try {
        $process = Start-Process `
            -FilePath $resolvedFile `
            -ArgumentList (Join-ProcessArguments $resolvedArgs) `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -NoNewWindow `
            -Wait `
            -PassThru
        $exitCode = [int]$process.ExitCode
        Add-Content -LiteralPath $finalLog -Encoding UTF8 -Value ("`n# command: {0} {1}`n# exitCode: {2}`n" -f $resolvedFile, (Join-ProcessArguments $resolvedArgs), $exitCode)
        $stderrTail = Get-FileTailText -Path $stderrPath -TailLines 1 -MaxChars 1
        if ($stderrTail) {
            Add-Content -LiteralPath $finalLog -Encoding UTF8 -Value "`n# stderr`n"
            Copy-FileStream -Source $stderrPath -Destination $finalLog
        }
        $stdoutTail = Get-FileTailText -Path $stdoutPath -TailLines $TailLines -MaxChars $TailChars
        $stderrTextTail = Get-FileTailText -Path $stderrPath -TailLines $TailLines -MaxChars $TailChars
        $tail = (@($stdoutTail, $stderrTextTail) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join [Environment]::NewLine
        if ($exitCode -ne 0 -and !$AllowFailure) {
            throw "$Name failed with exit code $exitCode. See $finalLog"
        }
        return [pscustomobject]@{
            exitCode = $exitCode
            text = $tail
            json = Get-LastJsonObjectFromText -Text $tail
            logPath = $finalLog
        }
    } finally {
    }
}

function Resolve-PythonExe {
    if ($PythonExe) {
        $resolved = Resolve-RepoPath $PythonExe
        if (!(Test-Path -LiteralPath $resolved)) {
            throw "Python executable was not found: $resolved"
        }
        return $resolved
    }
    $candidates = @(
        (Join-Path $script:Root ".deps\python-3.13.13-embed-amd64\python.exe"),
        (Join-Path $script:Root ".deps\rtx4090-validation\python-yolo-venv\Scripts\python.exe"),
        (Join-Path $script:Root ".deps\python-3.13.13-ocr-amd64\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }
    throw "Python was not found. Install Python or pass -PythonExe."
}

function Resolve-WorkerExe {
    if ($WorkerExe) {
        $resolved = Resolve-RepoPath $WorkerExe
        if (!(Test-Path -LiteralPath $resolved)) {
            throw "Worker executable was not found: $resolved"
        }
        return $resolved
    }
    $candidates = @(
        (Join-Path $script:Root "$BuildDir\bin\aitrain_worker.exe"),
        (Join-Path $script:Root "aitrain_worker.exe"),
        (Join-Path $script:Root "$BuildDir\package-smoke\aitrain_worker.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "aitrain_worker.exe was not found. Build first or pass -WorkerExe."
}

function Resolve-OcrPythonExe {
    param([string]$WorkRoot)
    if ($OcrPythonExe) {
        $resolved = Resolve-RepoPath $OcrPythonExe
        if (!(Test-Path -LiteralPath $resolved)) {
            throw "OCR Python executable was not found: $resolved"
        }
        return $resolved
    }

    $ocrDir = if ($OcrPythonDir) { Resolve-RepoPath $OcrPythonDir } else { Join-Path $WorkRoot "env\ocr" }
    $candidates = @(
        (Join-Path $script:Root ".deps\rtx4090-validation\python-ocr-gpu\Scripts\python.exe"),
        (Join-Path $script:Root ".deps\rtx4090-validation\python-ocr\Scripts\python.exe"),
        (Join-Path $script:Root ".deps\python-3.13.13-ocr-amd64\Scripts\python.exe"),
        (Join-Path $script:Root ".deps\python-3.13.13-ocr-amd64\python.exe"),
        (Join-Path $ocrDir "Scripts\python.exe"),
        (Join-Path $ocrDir "python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    if ($DryRun -or $SkipOcrEnvironmentCreate) {
        return ""
    }

    $sourceZip = Join-Path $script:Root ".deps\python-3.13.13-embed-amd64.zip"
    if (!(Test-Path -LiteralPath $sourceZip)) {
        throw "OCR Python is missing and $sourceZip is unavailable. Run the Python setup first or pass -OcrPythonExe."
    }
    New-Item -ItemType Directory -Force $ocrDir | Out-Null
    Expand-Archive -Path $sourceZip -DestinationPath $ocrDir -Force
    $pth = Join-Path $ocrDir "python313._pth"
    if (Test-Path -LiteralPath $pth) {
        (Get-Content -LiteralPath $pth -Encoding ASCII) -replace "#import site", "import site" | Set-Content -LiteralPath $pth -Encoding ASCII
    }
    $createdPython = Join-Path $ocrDir "python.exe"
    if (!(Test-Path -LiteralPath $createdPython)) {
        throw "Created OCR Python directory but python.exe is missing: $createdPython"
    }
    $pipCheck = Invoke-ProcessCapture -Name "ocr-pip-check" -FilePath $createdPython -Arguments @("-m", "pip", "--version") -AllowFailure
    if ($pipCheck.exitCode -ne 0) {
        $getPip = Join-Path $script:Root ".deps\get-pip.py"
        if (!(Test-Path -LiteralPath $getPip)) {
            throw "OCR Python needs pip but $getPip is missing."
        }
        Invoke-ProcessCapture -Name "ocr-get-pip" -FilePath $createdPython -Arguments @($getPip) | Out-Null
    }
    return [System.IO.Path]::GetFullPath($createdPython)
}

function Ensure-OcrDependencies {
    param(
        [string]$Python,
        [string]$Repo,
        [string]$WorkRoot
    )
    if ([string]::IsNullOrWhiteSpace($Python) -or $DryRun -or $SkipOcrInstall) {
        return
    }
    $probe = Invoke-ProcessCapture `
        -Name "ocr-module-probe" `
        -FilePath $Python `
        -Arguments @("-c", "import paddle,paddleocr,yaml; print('paddle', paddle.__version__); print('paddleocr', getattr(paddleocr,'__version__','unknown'))") `
        -LogPath (Join-Path $WorkRoot "logs\ocr-module-probe.log") `
        -AllowFailure
    if ($probe.exitCode -eq 0) {
        return
    }
    $constraints = Join-Path $WorkRoot "ocr-constraints.txt"
    @(
        "albumentations==2.0.8",
        "lmdb==2.2.0",
        "numpy==2.4.4",
        "opencv-python==4.13.0.92",
        "pillow==12.2.0",
        "pydantic==2.13.3",
        "PyYAML==6.0.3",
        "RapidFuzz==3.14.5",
        "shapely==2.1.2",
        "tqdm==4.67.3"
    ) | Set-Content -LiteralPath $constraints -Encoding ASCII
    $args = @("-m", "pip", "install", "--no-warn-script-location", $PaddlePaddleRequirement, "-r", (Join-Path $Repo "requirements.txt"), "-c", $constraints)
    Invoke-ProcessCapture -Name "ocr-pip-install" -FilePath $Python -Arguments $args -LogPath (Join-Path $WorkRoot "logs\ocr-pip-install.log") | Out-Null
}

function Ensure-PaddleOcrRepo {
    param([string]$Repo)
    $repoFull = Resolve-RepoPath $Repo
    if ($DryRun) {
        return $repoFull
    }
    if (!(Test-Path -LiteralPath (Join-Path $repoFull "tools\train.py"))) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $repoFull) | Out-Null
        Invoke-ProcessCapture -Name "clone-paddleocr" -FilePath "git" -Arguments @("clone", "https://github.com/PaddlePaddle/PaddleOCR.git", $repoFull) | Out-Null
    }
    if ($PaddleOcrRef) {
        Invoke-ProcessCapture -Name "fetch-paddleocr-ref" -FilePath "git" -Arguments @("-C", $repoFull, "fetch", "--depth", "1", "origin", $PaddleOcrRef) | Out-Null
        Invoke-ProcessCapture -Name "checkout-paddleocr-ref" -FilePath "git" -Arguments @("-C", $repoFull, "checkout", "--detach", "FETCH_HEAD") | Out-Null
    }
    return $repoFull
}

function Test-PaddleGpuAvailable {
    param([string]$Python)
    if ([string]::IsNullOrWhiteSpace($Python)) {
        return $false
    }
    $probe = Invoke-ProcessCapture `
        -Name "paddle-gpu-probe" `
        -FilePath $Python `
        -Arguments @("-c", "import paddle; print(paddle.device.is_compiled_with_cuda())") `
        -AllowFailure
    return ($probe.exitCode -eq 0 -and ($probe.text -match "True"))
}

function Test-TorchCudaAvailable {
    param([string]$Python)
    if ([string]::IsNullOrWhiteSpace($Python)) {
        return $false
    }
    $probe = Invoke-ProcessCapture `
        -Name "torch-cuda-probe" `
        -FilePath $Python `
        -Arguments @("-c", "import torch; print(torch.cuda.is_available())") `
        -AllowFailure
    return ($probe.exitCode -eq 0 -and ($probe.text -match "True"))
}

function Test-DeviceRequiresCuda {
    param([string]$DeviceValue)
    $normalized = $DeviceValue.Trim().ToLowerInvariant()
    return ($normalized -ne "cpu" -and $normalized -ne "-1")
}

function ConvertTo-ProtocolLine {
    param(
        [string]$Type,
        [object]$Payload
    )
    return ([ordered]@{
        type = $Type
        payload = $Payload
    } | ConvertTo-Json -Depth 80 -Compress)
}

function Invoke-WorkerCommand {
    param(
        [string]$ResolvedWorkerExe,
        [string]$CommandType,
        [object]$Request,
        [string]$EventsPath
    )

    $eventsParent = Split-Path -Parent $EventsPath
    New-Item -ItemType Directory -Force $eventsParent | Out-Null
    if (Test-Path -LiteralPath $EventsPath) {
        Remove-Item -LiteralPath $EventsPath -Force
    }

    $pipeName = "aitrain_full_lifecycle_" + ([guid]::NewGuid().ToString("N"))
    $pipe = $null
    $worker = $null
    $reader = $null
    $writer = $null
    $terminalEvent = $null
    $startSent = $false
    $utf8 = New-Utf8NoBomEncoding

    try {
        $pipe = [System.IO.Pipes.NamedPipeServerStream]::new(
            $pipeName,
            [System.IO.Pipes.PipeDirection]::InOut,
            1,
            [System.IO.Pipes.PipeTransmissionMode]::Byte,
            [System.IO.Pipes.PipeOptions]::Asynchronous)

        $worker = Start-Process `
            -FilePath $ResolvedWorkerExe `
            -ArgumentList @("--server", $pipeName) `
            -WorkingDirectory (Split-Path -Parent $ResolvedWorkerExe) `
            -WindowStyle Hidden `
            -PassThru

        $connectWait = $pipe.BeginWaitForConnection($null, $null)
        if (-not $connectWait.AsyncWaitHandle.WaitOne(15000)) {
            throw "Worker did not connect to command pipe within 15 seconds."
        }
        $pipe.EndWaitForConnection($connectWait)
        if ($pipe.CanTimeout) {
            $pipe.ReadTimeout = 1000
            $pipe.WriteTimeout = 1000
        }

        $reader = [System.IO.StreamReader]::new($pipe, [System.Text.Encoding]::UTF8, $false, 4096, $true)
        $writer = [System.IO.StreamWriter]::new($pipe, $utf8, 4096, $true)
        $writer.AutoFlush = $true
        $requestLine = ConvertTo-ProtocolLine -Type $CommandType -Payload $Request

        while ($true) {
            $line = $null
            try {
                $line = $reader.ReadLine()
            } catch [System.IO.IOException] {
                if ($worker.HasExited) {
                    throw "Worker exited before emitting a terminal event. ExitCode=$($worker.ExitCode)"
                }
                continue
            }
            if ($null -eq $line) {
                if ($worker.HasExited) {
                    throw "Worker pipe closed before emitting a terminal event. ExitCode=$($worker.ExitCode)"
                }
                continue
            }
            [System.IO.File]::AppendAllText($EventsPath, $line + [Environment]::NewLine, $utf8)

            try {
                $event = $line | ConvertFrom-Json
            } catch {
                continue
            }
            $eventType = [string]$event.type
            if (-not $startSent -and $eventType -eq "ready") {
                $writer.WriteLine($requestLine)
                $startSent = $true
                continue
            }
            if ($eventType -in @("completed", "failed", "canceled")) {
                $terminalEvent = $event
                break
            }
        }

        if (-not $startSent) {
            throw "Worker connected but did not emit a ready event."
        }
        if ($null -eq $terminalEvent) {
            throw "Worker did not emit a terminal event."
        }
        if (-not $worker.WaitForExit(10000)) {
            Stop-Process -Id $worker.Id -Force
            throw "Worker emitted a terminal event but did not exit cleanly."
        }
        return $terminalEvent
    } finally {
        if ($writer) { $writer.Dispose() }
        if ($reader) { $reader.Dispose() }
        if ($pipe) { $pipe.Dispose() }
        if ($worker -and -not $worker.HasExited) {
            Stop-Process -Id $worker.Id -Force
        }
    }
}

function Convert-ToSafeName {
    param([string]$Value)
    return ($Value -replace '[^A-Za-z0-9_.-]+', '_' -replace '_+', '_').Trim('_')
}

function Get-FirstImage {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or !(Test-Path -LiteralPath $Path)) {
        return ""
    }
    $image = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png", ".bmp") } |
        Sort-Object FullName |
        Select-Object -First 1
    if ($image) {
        return $image.FullName
    }
    return ""
}

function Get-FirstRelativeImage {
    param(
        [string]$Root,
        [string]$PreferredSubdir = ""
    )
    $base = if ($PreferredSubdir) { Join-Path $Root $PreferredSubdir } else { $Root }
    $image = Get-FirstImage -Path $base
    if (!$image -and $PreferredSubdir) {
        $image = Get-FirstImage -Path $Root
    }
    if (!$image) {
        return ""
    }
    return [System.IO.Path]::GetRelativePath($Root, $image).Replace("/", "\")
}

function New-P1YoloCases {
    $cases = @()
    $scales = @("n", "s", "m", "l", "x")
    foreach ($scale in $scales) {
        $cases += [pscustomobject]@{ name = "yolov5$scale-detect-yaml"; group = "p1"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolov5$scale.yaml"; sourceType = "yaml"; end2end = "false" }
        $cases += [pscustomobject]@{ name = "yolov5$($scale)u-detect-pt"; group = "p1"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolov5$($scale)u.pt"; sourceType = "pt"; end2end = "false" }
    }
    foreach ($family in @("yolov8", "yolo11", "yolo12")) {
        foreach ($scale in $scales) {
            foreach ($sourceType in @("yaml", "pt")) {
                $cases += [pscustomobject]@{ name = "$family$scale-detect-$sourceType"; group = "p1"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "$family$scale.$sourceType"; sourceType = $sourceType; end2end = "false" }
                $cases += [pscustomobject]@{ name = "$family$scale-segment-$sourceType"; group = "p1"; task = "segmentation"; backend = "ultralytics_yolo_segment"; model = "$family$scale-seg.$sourceType"; sourceType = $sourceType; end2end = "false" }
            }
        }
    }
    foreach ($scale in $scales) {
        foreach ($variant in @("p2", "p6")) {
            $cases += [pscustomobject]@{ name = "yolov8$scale-$variant-detect-yaml"; group = "p1"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolov8$scale-$variant.yaml"; sourceType = "yaml"; end2end = "false" }
        }
    }
    return $cases
}

function New-Yolo26Cases {
    $cases = @()
    foreach ($scale in @("n", "s", "m", "l", "x")) {
        foreach ($sourceType in @("yaml", "pt")) {
            $cases += [pscustomobject]@{ name = "yolo26$scale-detect-$sourceType"; group = "yolo26"; task = "detection"; backend = "ultralytics_yolo_detect"; model = "yolo26$scale.$sourceType"; sourceType = $sourceType; end2end = "auto" }
            $cases += [pscustomobject]@{ name = "yolo26$scale-segment-$sourceType"; group = "yolo26"; task = "segmentation"; backend = "ultralytics_yolo_segment"; model = "yolo26$scale-seg.$sourceType"; sourceType = $sourceType; end2end = "auto" }
        }
    }
    return $cases
}

function New-OcrDetCases {
    return @(
        [pscustomobject]@{ name = "ppocrv4-mobile-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv4_mobile_det"; version = "PP-OCRv4" },
        [pscustomobject]@{ name = "ppocrv5-mobile-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv5_mobile_det"; version = "PP-OCRv5" },
        [pscustomobject]@{ name = "ppocrv5-server-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv5_server_det"; version = "PP-OCRv5" },
        [pscustomobject]@{ name = "ppocrv6-tiny-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv6_tiny_det"; version = "PP-OCRv6" },
        [pscustomobject]@{ name = "ppocrv6-small-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv6_small_det"; version = "PP-OCRv6" },
        [pscustomobject]@{ name = "ppocrv6-medium-det"; task = "ocr_detection"; backend = "paddleocr_det_official"; preset = "PP-OCRv6_medium_det"; version = "PP-OCRv6" }
    )
}

function New-OcrRecCases {
    return @(
        [pscustomobject]@{ name = "ppocrv4-mobile-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv4_mobile_rec"; version = "PP-OCRv4" },
        [pscustomobject]@{ name = "ppocrv5-mobile-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv5_mobile_rec"; version = "PP-OCRv5" },
        [pscustomobject]@{ name = "ppocrv5-server-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv5_server_rec"; version = "PP-OCRv5" },
        [pscustomobject]@{ name = "en-ppocrv5-mobile-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "en_PP-OCRv5_mobile_rec"; version = "PP-OCRv5" },
        [pscustomobject]@{ name = "ppocrv6-tiny-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv6_tiny_rec"; version = "PP-OCRv6" },
        [pscustomobject]@{ name = "ppocrv6-small-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv6_small_rec"; version = "PP-OCRv6" },
        [pscustomobject]@{ name = "ppocrv6-medium-rec"; task = "ocr_recognition"; backend = "paddleocr_rec_official"; preset = "PP-OCRv6_medium_rec"; version = "PP-OCRv6" }
    )
}

function New-OcrSystemCases {
    return @(
        [pscustomobject]@{ name = "ppocrv4-mobile-system"; detPreset = "PP-OCRv4_mobile_det"; recPreset = "PP-OCRv4_mobile_rec" },
        [pscustomobject]@{ name = "ppocrv5-mobile-system"; detPreset = "PP-OCRv5_mobile_det"; recPreset = "PP-OCRv5_mobile_rec" },
        [pscustomobject]@{ name = "ppocrv5-server-system"; detPreset = "PP-OCRv5_server_det"; recPreset = "PP-OCRv5_server_rec" },
        [pscustomobject]@{ name = "ppocrv5-mobile-en-system"; detPreset = "PP-OCRv5_mobile_det"; recPreset = "en_PP-OCRv5_mobile_rec" },
        [pscustomobject]@{ name = "ppocrv6-tiny-system"; detPreset = "PP-OCRv6_tiny_det"; recPreset = "PP-OCRv6_tiny_rec" },
        [pscustomobject]@{ name = "ppocrv6-small-system"; detPreset = "PP-OCRv6_small_det"; recPreset = "PP-OCRv6_small_rec" },
        [pscustomobject]@{ name = "ppocrv6-medium-system"; detPreset = "PP-OCRv6_medium_det"; recPreset = "PP-OCRv6_medium_rec" }
    )
}

function Select-PreflightYoloCases {
    param([object[]]$Cases)
    return @(
        ($Cases | Where-Object { $_.model -eq "yolov8n.yaml" } | Select-Object -First 1),
        ($Cases | Where-Object { $_.model -eq "yolov8n-seg.yaml" } | Select-Object -First 1)
    ) | Where-Object { $null -ne $_ }
}

function Select-PreflightOcrCases {
    return [pscustomobject]@{
        det = @(New-OcrDetCases | Where-Object { $_.preset -eq "PP-OCRv5_mobile_det" })
        rec = @(New-OcrRecCases | Where-Object { $_.preset -eq "PP-OCRv5_mobile_rec" })
        system = @(New-OcrSystemCases | Where-Object { $_.name -eq "ppocrv5-mobile-system" })
    }
}

function Initialize-Datasets {
    param(
        [string]$Python,
        [string]$OcrPython,
        [string]$WorkRoot
    )
    $dataRoot = Join-Path $WorkRoot "datasets"
    $downloads = Join-Path $dataRoot "downloads"
    $reports = Join-Path $dataRoot "reports"
    New-Item -ItemType Directory -Force $dataRoot, $downloads, $reports | Out-Null

    $manifest = [ordered]@{
        generatedAt = [DateTime]::UtcNow.ToString("o")
        policy = "Public datasets are required for this full lifecycle gate. Generated minimal datasets are diagnostic only and do not satisfy this run."
        yoloDetection = [ordered]@{ yaml = "coco128.yaml"; status = "not_run"; path = ""; reportPath = "" }
        yoloSegmentation = [ordered]@{ yaml = "coco128-seg.yaml"; status = "not_run"; path = ""; reportPath = "" }
        ocr = [ordered]@{ source = "Total-Text"; status = "not_run"; dataDir = ""; manifestPath = "" }
    }

    if ($DryRun) {
        $manifest.yoloDetection.status = "planned"
        $manifest.yoloDetection.path = Join-Path $dataRoot "coco128"
        $manifest.yoloSegmentation.status = "planned"
        $manifest.yoloSegmentation.path = Join-Path $dataRoot "coco128-seg"
        $manifest.ocr.status = "planned"
        $manifest.ocr.dataDir = Join-Path $dataRoot "production-ocr-data"
        return $manifest
    }

    if ($SkipDataPrep) {
        $manifest.yoloDetection.status = "skipped"
        $manifest.yoloDetection.path = Join-Path $dataRoot "coco128"
        $manifest.yoloSegmentation.status = "skipped"
        $manifest.yoloSegmentation.path = Join-Path $dataRoot "coco128-seg"
        $manifest.ocr.status = "skipped"
        $manifest.ocr.dataDir = Join-Path $dataRoot "production-ocr-data"
        return $manifest
    }

    $materializer = Join-Path $script:Root "tools\materialize-ultralytics-dataset.py"
    foreach ($item in @(
            @{ key = "yoloDetection"; yaml = "coco128.yaml"; destination = Join-Path $dataRoot "coco128" },
            @{ key = "yoloSegmentation"; yaml = "coco128-seg.yaml"; destination = Join-Path $dataRoot "coco128-seg" })) {
        $report = Join-Path $reports "$($item.yaml).json"
        $manifest[$item.key].reportPath = $report
        $manifest[$item.key].path = $item.destination
        try {
            $result = Invoke-ProcessCapture `
                -Name "materialize-$($item.yaml)" `
                -FilePath $Python `
                -Arguments @($materializer, "--yaml", $item.yaml, "--destination", $item.destination, "--downloads", $downloads, "--materialized-root", $dataRoot, "--report", $report) `
                -LogPath (Join-Path $reports "$($item.yaml).log") `
                -AllowFailure
            if ($result.exitCode -eq 0 -and (Test-Path -LiteralPath (Join-Path $item.destination "data.yaml"))) {
                $manifest[$item.key].status = "passed"
            } else {
                $manifest[$item.key].status = "blocked_dataset_download"
                $manifest[$item.key].error = $result.text
            }
        } catch {
            $manifest[$item.key].status = "blocked_dataset_download"
            $manifest[$item.key].error = $_.Exception.Message
        }
    }

    $ocrData = Join-Path $dataRoot "production-ocr-data"
    $ocrManifest = Join-Path $ocrData "manifests\production_ocr_data_manifest.json"
    $manifest.ocr.dataDir = $ocrData
    $manifest.ocr.manifestPath = $ocrManifest
    try {
        $ocrDataPython = if ($OcrPython) { $OcrPython } else { $Python }
        $result = Invoke-ProcessCapture `
            -Name "prepare-production-ocr-data" `
            -FilePath (Join-Path $script:Root "tools\prepare-production-ocr-data.ps1") `
            -Arguments @("-WorkDir", $ocrData, "-Python", $ocrDataPython) `
            -LogPath (Join-Path $reports "prepare-production-ocr-data.log") `
            -AllowFailure
        if ($result.exitCode -eq 0 -and (Test-Path -LiteralPath $ocrManifest)) {
            $manifest.ocr.status = "passed"
        } else {
            $manifest.ocr.status = "blocked_dataset_download"
            $manifest.ocr.error = $result.text
        }
    } catch {
        $manifest.ocr.status = "blocked_dataset_download"
        $manifest.ocr.error = $_.Exception.Message
    }

    return $manifest
}

function Write-BlockedRow {
    param(
        [string]$RowDir,
        [object]$Case,
        [string]$Status,
        [string]$Reason
    )
    $summary = [ordered]@{
        runId = $script:RunId
        name = [string]$Case.name
        group = Get-ObjectString -Object $Case -Name "group"
        task = Get-ObjectString -Object $Case -Name "task"
        backend = Get-ObjectString -Object $Case -Name "backend"
        model = if ($Case.PSObject.Properties.Name -contains "model") { [string]$Case.model } else { [string]$Case.preset }
        modelPreset = Get-ObjectString -Object $Case -Name "preset"
        sourceType = Get-ObjectString -Object $Case -Name "sourceType"
        status = $Status
        reason = $Reason
        startedAt = [DateTime]::UtcNow.ToString("o")
        finishedAt = [DateTime]::UtcNow.ToString("o")
    }
    Write-JsonFile -Path (Join-Path $RowDir "row_summary.json") -Value $summary
    return [pscustomobject]$summary
}

function Get-Yolo26TargetedMatrixAcceptedSummary {
    $summaryPath = Resolve-RepoPath $Yolo26MatrixSummary
    $summary = Read-JsonFile $summaryPath
    if (!$summary -or ![bool]$summary.ok -or [string]$summary.status -ne "passed" -or [string]$summary.mode -ne "full") {
        return $null
    }
    $requiredResults = @($summary.results | Where-Object { $_.required })
    if ($requiredResults.Count -lt 20) {
        return $null
    }
    $notPassed = @($requiredResults | Where-Object { [string]$_.status -ne "passed" })
    if ($notPassed.Count -gt 0) {
        return $null
    }
    foreach ($result in $requiredResults) {
        $deployments = $result.deployments
        if (!$deployments) {
            return $null
        }
        foreach ($target in @("onnx", "tensorrt")) {
            if ($deployments.PSObject.Properties.Name -notcontains $target) {
                return $null
            }
            $status = [string]$deployments.$target.status
            if ($status -notin @("passed", "blocked", "failed", "hardware-blocked")) {
                return $null
            }
        }
    }
    $summaryPython = Get-ObjectString -Object $summary -Name "yolo26Python"
    if ([string]::IsNullOrWhiteSpace($summaryPython)) {
        return $null
    }
    $resolvedPython = Resolve-RepoPath $summaryPython
    if (!(Test-Path -LiteralPath $resolvedPython)) {
        return $null
    }
    Add-Member -InputObject $summary -NotePropertyName "resolvedYolo26Python" -NotePropertyValue $resolvedPython -Force
    Add-Member -InputObject $summary -NotePropertyName "resolvedSummaryPath" -NotePropertyValue $summaryPath -Force
    return $summary
}

function Test-Yolo26TargetedMatrixAccepted {
    return $null -ne (Get-Yolo26TargetedMatrixAcceptedSummary)
}

function Resolve-YoloPythonForCase {
    param(
        [object]$Case,
        [string]$DefaultPython
    )
    $model = [string]$Case.model
    if ($model -match '^yolo26') {
        $summary = Get-Yolo26TargetedMatrixAcceptedSummary
        if ($summary) {
            return [string]$summary.resolvedYolo26Python
        }
    }
    return $DefaultPython
}

function Get-YoloMainlineBlocker {
    param([object]$Case)
    $model = [string]$Case.model
    if ($model -match '^yolo12[a-z0-9]*-seg\.pt$') {
        return [pscustomobject]@{
            status = "blocked"
            reason = "blocked_missing_official_weight: current official Ultralytics environment cannot resolve yolo12 segmentation .pt weights; keep YAML architecture evidence separate."
        }
    }
    if ($model -match '^yolo26') {
        if (Test-Yolo26TargetedMatrixAccepted) {
            return $null
        }
        $sourceType = Get-ObjectString -Object $Case -Name "sourceType"
        $code = if ($sourceType -eq "yaml") { "blocked_model_unavailable" } else { "blocked_ultralytics_incompatible" }
        return [pscustomobject]@{
            status = "blocked"
            reason = "${code}: YOLO26 is deferred from the main lifecycle resume until the isolated targeted YOLO26 matrix passes. Expected summary: $(Resolve-RepoPath $Yolo26MatrixSummary)"
        }
    }
    return $null
}

function Restore-OcrAdapterRowFromReport {
    param(
        [object]$Case,
        [string]$RowDir,
        [string]$OutputPath,
        [bool]$IsDet
    )
    $reportName = if ($IsDet) { "paddleocr_official_det_report.json" } else { "paddleocr_official_rec_report.json" }
    $reportPath = Join-Path $OutputPath $reportName
    $report = Read-JsonFile $reportPath
    if (!$report -or ($report.PSObject.Properties.Name -contains "ok" -and -not [bool]$report.ok)) {
        return $null
    }
    $trainOk = ($report.PSObject.Properties.Name -notcontains "trainExitCode") -or ([int]$report.trainExitCode -eq 0)
    $exportOk = ($report.PSObject.Properties.Name -notcontains "exportExitCode") -or ([int]$report.exportExitCode -eq 0)
    $predictOk = $IsDet -or ($report.PSObject.Properties.Name -notcontains "predictExitCode") -or ([int]$report.predictExitCode -eq 0)
    if (!$trainOk -or !$exportOk -or !$predictOk) {
        return $null
    }
    $finished = [DateTime]::UtcNow
    $summary = [ordered]@{
        runId = $script:RunId
        name = [string]$Case.name
        task = [string]$Case.task
        backend = [string]$Case.backend
        modelPreset = [string]$Case.preset
        status = "passed"
        startedAt = if ($report.PSObject.Properties.Name -contains "startedAt") { [string]$report.startedAt } else { $finished.ToString("o") }
        finishedAt = $finished.ToString("o")
        elapsedSeconds = 0
        epochs = $script:EffectiveEpochs
        outputPath = $OutputPath
        reportPath = $reportPath
        logPath = Join-Path $RowDir "adapter.log"
        exitCode = 0
        checkpointPath = if ($report.PSObject.Properties.Name -contains "checkpointPath") { [string]$report.checkpointPath } else { "" }
        inferenceModelDir = if ($report.PSObject.Properties.Name -contains "inferenceModelDir") { [string]$report.inferenceModelDir } else { (Join-Path $OutputPath "official_inference") }
        dictionaryFile = if ($report.PSObject.Properties.Name -contains "dictPath") { [string]$report.dictPath } else { (Join-Path $OutputPath "official_data\dict.txt") }
        predictionPath = if (!$IsDet) { Join-Path $OutputPath "official_prediction.json" } else { "" }
        restoredFromOfficialReport = $true
        failure = ""
    }
    Write-JsonFile -Path (Join-Path $RowDir "row_summary.json") -Value $summary
    return [pscustomobject]$summary
}

function Invoke-YoloLifecycleRow {
    param(
        [object]$Case,
        [string]$DatasetPath,
        [string]$SampleImage,
        [string]$Python,
        [string]$Worker,
        [string]$WorkRoot,
        [string[]]$Targets
    )
    $rowDir = Join-Path $WorkRoot ("yolo\" + (Convert-ToSafeName $Case.name))
    $rowSummaryPath = Join-Path $rowDir "row_summary.json"
    $existing = Read-JsonFile $rowSummaryPath
    if ($Resume -and !$RerunFailed -and $existing -and [string]$existing.status -in @("passed", "passed_with_findings")) {
        Write-Step "resume skip YOLO $($Case.name)"
        return Update-RowSummaryRunId -Path $rowSummaryPath -Summary $existing
    }
    New-Item -ItemType Directory -Force $rowDir | Out-Null
    $mainlineBlocker = Get-YoloMainlineBlocker -Case $Case
    if ($mainlineBlocker) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status $mainlineBlocker.status -Reason $mainlineBlocker.reason
    }
    $casePython = Resolve-YoloPythonForCase -Case $Case -DefaultPython $Python
    $caseCudaAvailable = if ([string]$Case.model -match '^yolo26') {
        [bool](Test-TorchCudaAvailable -Python $casePython)
    } else {
        [bool]$script:YoloCudaAvailable
    }
    if ((Test-DeviceRequiresCuda -DeviceValue $Device) -and -not $caseCudaAvailable) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "blocked_cuda_torch_unavailable: requested GPU YOLO training but torch.cuda.is_available() is false for the selected Python executable"
    }
    if ([string]::IsNullOrWhiteSpace($DatasetPath) -or !(Test-Path -LiteralPath $DatasetPath)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "public YOLO dataset is unavailable"
    }
    if ([string]::IsNullOrWhiteSpace($SampleImage) -or !(Test-Path -LiteralPath $SampleImage)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "sample image is unavailable for inference/deployment validation"
    }
    if ($DryRun) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "planned" -Reason "dry run"
    }

    $started = [DateTime]::UtcNow
    $trainingOutput = Join-Path $rowDir "training"
    $trainRequestPath = Join-Path $rowDir "train_request.json"
    $trainEventsPath = Join-Path $rowDir "train_worker_events.jsonl"
    $trainRequest = [ordered]@{
        protocolVersion = 1
        taskId = "full-yolo-$($Case.name)"
        taskType = [string]$Case.task
        datasetPath = $DatasetPath
        outputPath = $trainingOutput
        backend = [string]$Case.backend
        parameters = [ordered]@{
            trainingBackend = [string]$Case.backend
            model = [string]$Case.model
            modelPreset = [string]$Case.model
            epochs = $script:EffectiveEpochs
            batchSize = $Batch
            imageSize = $ImageSize
            device = $Device
            pythonExecutable = $casePython
            workers = 0
            runName = [string]$Case.name
            compactEvents = $true
            exportOnnx = $true
            ultralyticsExportArgs = [ordered]@{
                format = "onnx"
                dynamic = $false
                half = $false
                int8 = $false
                imgsz = $ImageSize
                batch = 1
                device = $Device
            }
        }
    }
    Write-JsonFile -Path $trainRequestPath -Value $trainRequest
    $trainEvent = Invoke-WorkerCommand -ResolvedWorkerExe $Worker -CommandType "startTrain" -Request $trainRequest -EventsPath $trainEventsPath
    if ([string]$trainEvent.type -ne "completed") {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "training failed: $($trainEvent.payload.message)"
    }

    $trainingReportPath = Join-Path $trainingOutput "ultralytics_training_report.json"
    $trainingReport = Read-JsonFile $trainingReportPath
    if (!$trainingReport -or !$trainingReport.ok) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "training report missing or not ok"
    }
    $onnxPath = [string]$trainingReport.onnxPath
    $checkpointPath = [string]$trainingReport.checkpointPath
    if (!(Test-Path -LiteralPath $onnxPath)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "ONNX artifact is missing"
    }

    $deployments = [ordered]@{}
    $inferenceRequestPath = Join-Path $rowDir "onnx_inference_request.json"
    $inferenceOutput = Join-Path $rowDir "inference\onnx"
    $inferenceEventsPath = Join-Path $rowDir "onnx_inference_worker_events.jsonl"
    $inferenceRequest = [ordered]@{
        taskId = "full-infer-$($Case.name)"
        checkpointPath = $onnxPath
        imagePath = $SampleImage
        outputPath = $inferenceOutput
        confidenceThreshold = 0.01
        iouThreshold = 0.45
        maxDetections = 300
    }
    Write-JsonFile -Path $inferenceRequestPath -Value $inferenceRequest
    $inferEvent = Invoke-WorkerCommand -ResolvedWorkerExe $Worker -CommandType "infer" -Request $inferenceRequest -EventsPath $inferenceEventsPath
    if ([string]$inferEvent.type -ne "completed") {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "ONNX inference failed: $($inferEvent.payload.message)"
    }
    $predictionsPath = Join-Path $inferenceOutput "inference_predictions.json"
    $overlayPath = Join-Path $inferenceOutput "inference_overlay.png"
    if (!(Test-Path -LiteralPath $predictionsPath) -or !(Test-Path -LiteralPath $overlayPath)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "ONNX inference artifacts are missing"
    }

    if ($Targets -contains "onnx") {
        $onnxDeployOutput = Join-Path $rowDir "deployment\onnx"
        $onnxDeployRequestPath = Join-Path $rowDir "deployment_onnx_request.json"
        $onnxDeployEventsPath = Join-Path $rowDir "deployment_onnx_worker_events.jsonl"
        $onnxRequest = [ordered]@{
            taskId = "full-deploy-onnx-$($Case.name)"
            modelPath = $onnxPath
            outputPath = $onnxDeployOutput
            format = "onnx"
            sampleImagePath = $SampleImage
            options = [ordered]@{
                sampleImagePath = $SampleImage
                confidenceThreshold = 0.01
                iouThreshold = 0.45
                maxDetections = 300
            }
        }
        Write-JsonFile -Path $onnxDeployRequestPath -Value $onnxRequest
        $onnxDeployEvent = Invoke-WorkerCommand -ResolvedWorkerExe $Worker -CommandType "validateDeploymentArtifact" -Request $onnxRequest -EventsPath $onnxDeployEventsPath
        $deployments.onnx = [ordered]@{
            status = if ([string]$onnxDeployEvent.type -eq "completed") { [string]$onnxDeployEvent.payload.status } else { "failed" }
            reportPath = if ($onnxDeployEvent.PSObject.Properties.Name -contains "payload") { [string]$onnxDeployEvent.payload.reportPath } else { "" }
            outputPath = $onnxDeployOutput
            event = $onnxDeployEvent
        }
        if ($deployments.onnx.status -ne "passed") {
            return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "failed" -Reason "ONNX deployment validation did not pass"
        }
    }

    if ($Targets -contains "tensorrt") {
        $trtOutput = Join-Path $rowDir "deployment\tensorrt"
        New-Item -ItemType Directory -Force $trtOutput | Out-Null
        $trt = Invoke-ProcessCapture -Name "tensorrt-smoke-$($Case.name)" -FilePath $Worker -Arguments @("--tensorrt-smoke", $onnxPath) -LogPath (Join-Path $trtOutput "tensorrt_smoke.log") -AllowFailure
        $trtStatus = if ($trt.exitCode -eq 0 -and $trt.json -and $trt.json.ok) { "passed" } elseif ($trt.text -match "hardware|unavailable|not found|missing|requires|blocked") { "blocked" } else { "failed" }
        $deployments.tensorrt = [ordered]@{
            status = $trtStatus
            exitCode = $trt.exitCode
            result = $trt.json
            logPath = Join-Path $trtOutput "tensorrt_smoke.log"
        }
        Write-JsonFile -Path (Join-Path $trtOutput "tensorrt_smoke_summary.json") -Value $deployments.tensorrt
    }

    if (($Targets -contains "ncnn") -and [string]$Case.model -notmatch "^yolo26") {
        $ncnnOutput = Join-Path $rowDir "deployment\ncnn"
        New-Item -ItemType Directory -Force $ncnnOutput | Out-Null
        $ncnnOnnxPath = $onnxPath
        $ncnnArgs = @("-OnnxPath", $ncnnOnnxPath, "-SampleImagePath", $SampleImage, "-OutputDir", $ncnnOutput, "-TaskType", [string]$Case.task, "-WorkerExe", $Worker)
        if ($NcnnRoot) {
            $ncnnArgs = @("-NcnnRoot", (Resolve-RepoPath $NcnnRoot)) + $ncnnArgs
        }
        $ncnn = Invoke-ProcessCapture -Name "ncnn-smoke-$($Case.name)" -FilePath (Join-Path $script:Root "tools\phase-ncnn-runtime-smoke.ps1") -Arguments $ncnnArgs -LogPath (Join-Path $ncnnOutput "ncnn_smoke.log") -AllowFailure
        $ncnnSummaryPath = Join-Path $ncnnOutput "ncnn_runtime_smoke_summary.json"
        $ncnnJson = Read-JsonFile $ncnnSummaryPath
        $deployments.ncnn = [ordered]@{
            status = if ($ncnnJson -and $ncnnJson.status) { [string]$ncnnJson.status } elseif ($ncnn.exitCode -eq 0) { "passed" } else { "failed" }
            exitCode = $ncnn.exitCode
            summaryPath = $ncnnSummaryPath
            result = $ncnnJson
            logPath = Join-Path $ncnnOutput "ncnn_smoke.log"
            onnxPath = $ncnnOnnxPath
        }
    }

    $optionalFindings = @()
    foreach ($target in @("tensorrt", "ncnn")) {
        if ($deployments.Contains($target) -and [string]$deployments[$target].status -ne "passed") {
            $optionalFindings += "$target=$($deployments[$target].status)"
        }
    }
    $status = if ($optionalFindings.Count -gt 0) { "passed_with_findings" } else { "passed" }
    $finished = [DateTime]::UtcNow
    $summary = [ordered]@{
        runId = $script:RunId
        name = [string]$Case.name
        group = [string]$Case.group
        task = [string]$Case.task
        backend = [string]$Case.backend
        model = [string]$Case.model
        sourceType = [string]$Case.sourceType
        status = $status
        findings = $optionalFindings
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        epochs = $script:EffectiveEpochs
        artifacts = [ordered]@{
            trainingReport = $trainingReportPath
            checkpointPath = $checkpointPath
            onnxPath = $onnxPath
            predictionsPath = $predictionsPath
            overlayPath = $overlayPath
        }
        deployments = $deployments
    }
    Write-JsonFile -Path $rowSummaryPath -Value $summary
    return [pscustomobject]$summary
}

function Invoke-OcrAdapterRow {
    param(
        [object]$Case,
        [string]$DatasetPath,
        [string]$Python,
        [string]$Repo,
        [string]$WorkRoot
    )
    $rowDir = Join-Path $WorkRoot ("ocr\" + (Convert-ToSafeName $Case.name))
    $rowSummaryPath = Join-Path $rowDir "row_summary.json"
    New-Item -ItemType Directory -Force $rowDir | Out-Null
    $outputPath = Join-Path $rowDir "official"
    $isDet = [string]$Case.backend -eq "paddleocr_det_official"
    $existing = Read-JsonFile $rowSummaryPath
    if ($Resume -and !$RerunFailed -and $existing -and [string]$existing.status -eq "passed") {
        Write-Step "resume skip OCR $($Case.name)"
        return Update-RowSummaryRunId -Path $rowSummaryPath -Summary $existing
    }
    if ($Resume -and !$RerunFailed) {
        $restored = Restore-OcrAdapterRowFromReport -Case $Case -RowDir $rowDir -OutputPath $outputPath -IsDet $isDet
        if ($restored) {
            Write-Step "resume restored OCR $($Case.name) from official report"
            return $restored
        }
    }
    if ([string]::IsNullOrWhiteSpace($DatasetPath) -or !(Test-Path -LiteralPath $DatasetPath)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "public OCR dataset is unavailable"
    }
    if ([string]::IsNullOrWhiteSpace($Python) -or !(Test-Path -LiteralPath $Python)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "OCR Python is unavailable"
    }
    if ($script:OcrUseGpu -and !(Test-PaddleGpuAvailable -Python $Python)) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "blocked" -Reason "PaddlePaddle GPU is unavailable; pass -AllowOcrCpu only for an explicit CPU run"
    }
    if ($DryRun) {
        return Write-BlockedRow -RowDir $rowDir -Case $Case -Status "planned" -Reason "dry run"
    }

    $started = [DateTime]::UtcNow
    $requestPath = Join-Path $rowDir "request.json"
    $logPath = Join-Path $rowDir "adapter.log"
    $adapter = if ($isDet) { "python_trainers\ocr_det\paddleocr_det_official_adapter.py" } else { "python_trainers\ocr_rec\paddleocr_official_adapter.py" }
    $parameters = [ordered]@{
        trainingBackend = [string]$Case.backend
        paddleOcrRepoPath = $Repo
        paddleOcrRef = $PaddleOcrRef
        runOfficial = $true
        prepareOnly = $false
        modelPreset = [string]$Case.preset
        epochs = $script:EffectiveEpochs
        batchSize = if ($isDet) { 1 } else { 8 }
        useGpu = [bool]$script:OcrUseGpu
        validationRatio = 0.2
        officialLogVerbosity = "summary"
        officialLogEventIntervalSeconds = 30
        officialLogTailLines = 200
        officialPrintBatchStep = 20
        officialSaveEpochStep = 10
        checkpointRetention = "latest_best_inference"
    }
    if ($isDet) {
        $parameters.imageSize = $ImageSize
        $parameters.trainLabelFile = "det_gt_train.txt"
        $parameters.valLabelFile = "det_gt_val.txt"
        $parameters.calMetricDuringTrain = $false
    } else {
        $parameters.imageWidth = 320
        $parameters.imageHeight = 48
        $parameters.recImageShape = "3,48,320"
        $parameters.maxTextLength = 25
        $parameters.trainLabelFile = "rec_gt_train.txt"
        $parameters.valLabelFile = "rec_gt_val.txt"
        $parameters.runInferenceAfterExport = $true
        $parameters.inferenceImage = Get-FirstRelativeImage -Root $DatasetPath -PreferredSubdir "images\test"
    }
    $request = [ordered]@{
        protocolVersion = 1
        taskId = "full-$($Case.name)"
        taskType = [string]$Case.task
        datasetPath = $DatasetPath
        outputPath = $outputPath
        backend = [string]$Case.backend
        parameters = $parameters
    }
    Write-JsonFile -Path $requestPath -Value $request
    $run = Invoke-ProcessCapture -Name "ocr-$($Case.name)" -FilePath $Python -Arguments @((Join-Path $script:Root $adapter), "--request", $requestPath) -LogPath $logPath -AllowFailure
    $reportName = if ($isDet) { "paddleocr_official_det_report.json" } else { "paddleocr_official_rec_report.json" }
    $reportPath = Join-Path $outputPath $reportName
    $report = Read-JsonFile $reportPath
    $ok = ($run.exitCode -eq 0 -and $report -and (($report.PSObject.Properties.Name -notcontains "ok") -or $report.ok))
    $finished = [DateTime]::UtcNow
    $summary = [ordered]@{
        runId = $script:RunId
        name = [string]$Case.name
        task = [string]$Case.task
        backend = [string]$Case.backend
        modelPreset = [string]$Case.preset
        status = if ($ok) { "passed" } else { "failed" }
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        epochs = $script:EffectiveEpochs
        outputPath = $outputPath
        reportPath = $reportPath
        logPath = $logPath
        exitCode = $run.exitCode
        checkpointPath = if ($report -and $report.PSObject.Properties.Name -contains "checkpointPath") { [string]$report.checkpointPath } else { "" }
        inferenceModelDir = if ($report -and $report.PSObject.Properties.Name -contains "inferenceModelDir") { [string]$report.inferenceModelDir } else { (Join-Path $outputPath "official_inference") }
        dictionaryFile = if ($report -and $report.PSObject.Properties.Name -contains "dictPath") { [string]$report.dictPath } else { (Join-Path $outputPath "official_data\dict.txt") }
        predictionPath = if (!$isDet) { Join-Path $outputPath "official_prediction.json" } else { "" }
        failure = if ($ok) { "" } else { $run.text }
    }
    Write-JsonFile -Path $rowSummaryPath -Value $summary
    return [pscustomobject]$summary
}

function Invoke-OcrSystemRow {
    param(
        [object]$Case,
        [object]$DetByPreset,
        [object]$RecByPreset,
        [string]$SystemImages,
        [string]$Python,
        [string]$Repo,
        [string]$WorkRoot
    )
    $rowDir = Join-Path $WorkRoot ("ocr-system\" + (Convert-ToSafeName $Case.name))
    $rowSummaryPath = Join-Path $rowDir "row_summary.json"
    $existing = Read-JsonFile $rowSummaryPath
    if ($Resume -and !$RerunFailed -and $existing -and [string]$existing.status -eq "passed") {
        Write-Step "resume skip OCR System $($Case.name)"
        return Update-RowSummaryRunId -Path $rowSummaryPath -Summary $existing
    }
    New-Item -ItemType Directory -Force $rowDir | Out-Null
    $det = $DetByPreset[[string]$Case.detPreset]
    $rec = $RecByPreset[[string]$Case.recPreset]
    if (!$det -or [string]$det.status -ne "passed") {
        return Write-BlockedRow -RowDir $rowDir -Case ([pscustomobject]@{ name = $Case.name; task = "ocr"; preset = "$($Case.detPreset)+$($Case.recPreset)" }) -Status "blocked" -Reason "required Det preset did not pass"
    }
    if (!$rec -or [string]$rec.status -ne "passed") {
        return Write-BlockedRow -RowDir $rowDir -Case ([pscustomobject]@{ name = $Case.name; task = "ocr"; preset = "$($Case.detPreset)+$($Case.recPreset)" }) -Status "blocked" -Reason "required Rec preset did not pass"
    }
    if ([string]::IsNullOrWhiteSpace($SystemImages) -or !(Test-Path -LiteralPath $SystemImages)) {
        return Write-BlockedRow -RowDir $rowDir -Case ([pscustomobject]@{ name = $Case.name; task = "ocr"; preset = "$($Case.detPreset)+$($Case.recPreset)" }) -Status "blocked" -Reason "system images are unavailable"
    }
    if ($DryRun) {
        return Write-BlockedRow -RowDir $rowDir -Case ([pscustomobject]@{ name = $Case.name; task = "ocr"; preset = "$($Case.detPreset)+$($Case.recPreset)" }) -Status "planned" -Reason "dry run"
    }
    $started = [DateTime]::UtcNow
    $outputPath = Join-Path $rowDir "official"
    $requestPath = Join-Path $rowDir "request.json"
    $logPath = Join-Path $rowDir "adapter.log"
    $request = [ordered]@{
        protocolVersion = 1
        taskId = "full-$($Case.name)"
        taskType = "ocr"
        datasetPath = $SystemImages
        outputPath = $outputPath
        backend = "paddleocr_system_official"
        parameters = [ordered]@{
            trainingBackend = "paddleocr_system_official"
            paddleOcrRepoPath = $Repo
            paddleOcrRef = $PaddleOcrRef
            detModelPreset = [string]$Case.detPreset
            recModelPreset = [string]$Case.recPreset
            recReportPath = [string]$rec.reportPath
            detModelDir = [string]$det.inferenceModelDir
            recModelDir = [string]$rec.inferenceModelDir
            dictionaryFile = [string]$rec.dictionaryFile
            inferenceImage = $SystemImages
            dropScore = 0.0
            useGpu = [bool]$script:OcrUseGpu
            officialLogVerbosity = "summary"
            officialLogEventIntervalSeconds = 30
            officialLogTailLines = 200
        }
    }
    Write-JsonFile -Path $requestPath -Value $request
    $run = Invoke-ProcessCapture -Name "ocr-system-$($Case.name)" -FilePath $Python -Arguments @((Join-Path $script:Root "python_trainers\ocr_system\paddleocr_system_official_adapter.py"), "--request", $requestPath) -LogPath $logPath -AllowFailure
    $reportPath = Join-Path $outputPath "paddleocr_official_system_report.json"
    $predictionPath = Join-Path $outputPath "official_system_prediction.json"
    $report = Read-JsonFile $reportPath
    $prediction = Read-JsonFile $predictionPath
    $ok = ($run.exitCode -eq 0 -and $report -and $prediction -and (($prediction.PSObject.Properties.Name -notcontains "ok") -or $prediction.ok))
    $finished = [DateTime]::UtcNow
    $summary = [ordered]@{
        runId = $script:RunId
        name = [string]$Case.name
        task = "ocr"
        backend = "paddleocr_system_official"
        detPreset = [string]$Case.detPreset
        recPreset = [string]$Case.recPreset
        status = if ($ok) { "passed" } else { "failed" }
        startedAt = $started.ToString("o")
        finishedAt = $finished.ToString("o")
        elapsedSeconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        outputPath = $outputPath
        reportPath = $reportPath
        predictionPath = $predictionPath
        systemResultsPath = Join-Path $outputPath "system_results.txt"
        logPath = $logPath
        exitCode = $run.exitCode
        failure = if ($ok) { "" } else { $run.text }
    }
    Write-JsonFile -Path $rowSummaryPath -Value $summary
    return [pscustomobject]$summary
}

function Write-MarkdownSummary {
    param(
        [string]$Path,
        [object]$Summary
    )
    $lines = @()
    $lines += "# Full Model Lifecycle Matrix"
    $lines += ""
    $lines += "- Status: $($Summary.status)"
    $lines += "- RunId: $($Summary.runId)"
    $lines += "- WorkDir: $($Summary.workDir)"
    $lines += "- Epochs: $($Summary.parameters.epochs)"
    $lines += "- Device: $($Summary.parameters.device)"
    $lines += "- YOLO rows: $($Summary.counts.yoloPassed)/$($Summary.counts.yoloTotal) passed or passed with findings"
    $lines += "- OCR rows: $($Summary.counts.ocrPassed)/$($Summary.counts.ocrTotal) passed"
    $lines += "- OCR System rows: $($Summary.counts.ocrSystemPassed)/$($Summary.counts.ocrSystemTotal) passed"
    $lines += ""
    $lines += "This is public-data engineering lifecycle evidence only. It is not customer-domain OCR production evidence and not an accuracy benchmark."
    $lines += ""
    $lines += "## Findings"
    if ($Summary.findings.Count -eq 0) {
        $lines += "- None"
    } else {
        foreach ($finding in $Summary.findings) {
            $lines += "- $finding"
        }
    }
    $lines | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-OverallStatus {
    param(
        [object[]]$YoloResults,
        [object[]]$OcrResults,
        [object[]]$SystemResults
    )
    $all = @($YoloResults + $OcrResults + $SystemResults)
    if (@($all | Where-Object { [string]$_.status -eq "failed" }).Count -gt 0) {
        return "failed"
    }
    if (@($all | Where-Object { [string]$_.status -eq "blocked" }).Count -gt 0) {
        return "blocked"
    }
    if (@($all | Where-Object { [string]$_.status -eq "passed_with_findings" }).Count -gt 0) {
        return "passed_with_findings"
    }
    if (@($all | Where-Object { [string]$_.status -eq "planned" }).Count -gt 0) {
        return "planned"
    }
    return "passed"
}

if ($MaxParallel -ne 1) {
    throw "full-model-lifecycle-matrix.ps1 currently supports sequential execution only. Use -MaxParallel 1."
}

$work = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force $work | Out-Null
$script:WorkRoot = $work
$script:ControllerFailurePath = Join-Path $work "controller_failure.json"
$mode = if ($DryRun) { "dry_run" } elseif ($Preflight) { "preflight" } else { "full" }
Write-JsonFile -Path (Join-Path $work "current_run.json") -Value ([ordered]@{
    runId = $script:RunId
    status = "running"
    mode = $mode
    pid = $PID
    startedAt = $script:StartedAt.ToString("o")
    workDir = $work
})
$targets = @($DeploymentTargets | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
foreach ($target in $targets) {
    if ($target -notin @("onnx", "tensorrt", "ncnn")) {
        throw "Unsupported deployment target: $target"
    }
}

$python = Resolve-PythonExe
$worker = Resolve-WorkerExe
$script:YoloCudaAvailable = if (Test-DeviceRequiresCuda -DeviceValue $Device) { [bool](Test-TorchCudaAvailable -Python $python) } else { $true }
$repoFull = Ensure-PaddleOcrRepo -Repo $PaddleOcrRepo
$ocrPython = Resolve-OcrPythonExe -WorkRoot $work
if (!$DryRun -and $repoFull -and (Test-Path -LiteralPath (Join-Path $repoFull "tools\train.py"))) {
    Ensure-OcrDependencies -Python $ocrPython -Repo $repoFull -WorkRoot $work
}

$envReportPath = Join-Path $work "environment_self_check.json"
$selfCheck = Invoke-ProcessCapture -Name "worker-self-check" -FilePath $worker -Arguments @("--self-check") -LogPath (Join-Path $work "logs\worker-self-check.log") -AllowFailure
$yolo26AcceptedSummary = Get-Yolo26TargetedMatrixAcceptedSummary
$environment = [ordered]@{
    runId = $script:RunId
    generatedAt = [DateTime]::UtcNow.ToString("o")
    workerExe = $worker
    yoloPython = $python
    yoloCudaAvailable = [bool]$script:YoloCudaAvailable
    yolo26MatrixSummary = Resolve-RepoPath $Yolo26MatrixSummary
    yolo26TargetedPython = if ($yolo26AcceptedSummary) { [string]$yolo26AcceptedSummary.resolvedYolo26Python } else { "" }
    ocrPython = $ocrPython
    paddleOcrRepo = $repoFull
    paddleOcrRef = if ((Test-Path -LiteralPath (Join-Path $repoFull ".git"))) { ((& git -C $repoFull rev-parse HEAD 2>$null) | Select-Object -First 1) } else { $PaddleOcrRef }
    ocrUseGpu = [bool]$script:OcrUseGpu
    paddleGpuAvailable = if ($ocrPython) { [bool](Test-PaddleGpuAvailable -Python $ocrPython) } else { $false }
    workerSelfCheck = $selfCheck.json
    ncnnRoot = if ($NcnnRoot) { Resolve-RepoPath $NcnnRoot } else { "" }
    deploymentTargets = $targets
}
Write-JsonFile -Path $envReportPath -Value $environment

$datasets = Initialize-Datasets -Python $python -OcrPython $ocrPython -WorkRoot $work
$datasetsPath = Join-Path $work "datasets_manifest.json"
Write-JsonFile -Path $datasetsPath -Value $datasets

$yoloCases = @(New-P1YoloCases) + @(New-Yolo26Cases)
$ocrDetCases = @(New-OcrDetCases)
$ocrRecCases = @(New-OcrRecCases)
$ocrSystemCases = @(New-OcrSystemCases)
if ($Preflight) {
    $yoloCases = @(Select-PreflightYoloCases -Cases $yoloCases)
    $preflightOcr = Select-PreflightOcrCases
    $ocrDetCases = @($preflightOcr.det)
    $ocrRecCases = @($preflightOcr.rec)
    $ocrSystemCases = @($preflightOcr.system)
}

$matrixPlan = [ordered]@{
    runId = $script:RunId
    generatedAt = [DateTime]::UtcNow.ToString("o")
    mode = $mode
    yolo = $yoloCases
    ocrDet = $ocrDetCases
    ocrRec = $ocrRecCases
    ocrSystem = $ocrSystemCases
}
Write-JsonFile -Path (Join-Path $work "matrix_plan.json") -Value $matrixPlan

$yoloResults = @()
$ocrResults = @()
$ocrSystemResults = @()

$detectDataset = [string]$datasets.yoloDetection.path
$segmentDataset = [string]$datasets.yoloSegmentation.path
$detectSample = Get-FirstImage -Path $detectDataset
$segmentSample = Get-FirstImage -Path $segmentDataset

foreach ($case in $yoloCases) {
    if ($DryRun) {
        $rowDir = Join-Path $work ("yolo\" + (Convert-ToSafeName $case.name))
        $yoloResults += Write-BlockedRow -RowDir $rowDir -Case $case -Status "planned" -Reason "dry run"
        continue
    }
    $datasetPath = if ([string]$case.task -eq "segmentation") { $segmentDataset } else { $detectDataset }
    $samplePath = if ([string]$case.task -eq "segmentation") { $segmentSample } else { $detectSample }
    if (([string]$case.task -eq "segmentation" -and [string]$datasets.yoloSegmentation.status -ne "passed") -or
        ([string]$case.task -eq "detection" -and [string]$datasets.yoloDetection.status -ne "passed")) {
        $rowDir = Join-Path $work ("yolo\" + (Convert-ToSafeName $case.name))
        $yoloResults += Write-BlockedRow -RowDir $rowDir -Case $case -Status "blocked" -Reason "required public YOLO dataset was not materialized"
        continue
    }
    $yoloResults += Invoke-YoloLifecycleRow -Case $case -DatasetPath $datasetPath -SampleImage $samplePath -Python $python -Worker $worker -WorkRoot $work -Targets $targets
}

$ocrDataDir = [string]$datasets.ocr.dataDir
$detDataset = Join-Path $ocrDataDir "det_dataset"
$recDataset = Join-Path $ocrDataDir "rec_dataset"
$systemImages = Join-Path $ocrDataDir "system_images"
if ($DryRun) {
    foreach ($case in @($ocrDetCases + $ocrRecCases)) {
        $rowDir = Join-Path $work ("ocr\" + (Convert-ToSafeName $case.name))
        $ocrResults += Write-BlockedRow -RowDir $rowDir -Case $case -Status "planned" -Reason "dry run"
    }
} elseif ([string]$datasets.ocr.status -ne "passed") {
    foreach ($case in @($ocrDetCases + $ocrRecCases)) {
        $rowDir = Join-Path $work ("ocr\" + (Convert-ToSafeName $case.name))
        $ocrResults += Write-BlockedRow -RowDir $rowDir -Case $case -Status "blocked" -Reason "required public OCR dataset was not materialized"
    }
} else {
    foreach ($case in $ocrDetCases) {
        $ocrResults += Invoke-OcrAdapterRow -Case $case -DatasetPath $detDataset -Python $ocrPython -Repo $repoFull -WorkRoot $work
    }
    foreach ($case in $ocrRecCases) {
        $ocrResults += Invoke-OcrAdapterRow -Case $case -DatasetPath $recDataset -Python $ocrPython -Repo $repoFull -WorkRoot $work
    }
}

$detByPreset = @{}
$recByPreset = @{}
foreach ($result in $ocrResults) {
    if ($result.PSObject.Properties.Name -notcontains "backend") {
        continue
    }
    if ([string]$result.backend -eq "paddleocr_det_official") {
        $detByPreset[[string]$result.modelPreset] = $result
    } elseif ([string]$result.backend -eq "paddleocr_rec_official") {
        $recByPreset[[string]$result.modelPreset] = $result
    }
}
foreach ($case in $ocrSystemCases) {
    if ($DryRun) {
        $rowDir = Join-Path $work ("ocr-system\" + (Convert-ToSafeName $case.name))
        $ocrSystemResults += Write-BlockedRow -RowDir $rowDir -Case ([pscustomobject]@{ name = $case.name; task = "ocr"; preset = "$($case.detPreset)+$($case.recPreset)" }) -Status "planned" -Reason "dry run"
        continue
    }
    $ocrSystemResults += Invoke-OcrSystemRow -Case $case -DetByPreset $detByPreset -RecByPreset $recByPreset -SystemImages $systemImages -Python $ocrPython -Repo $repoFull -WorkRoot $work
}

$findings = @()
foreach ($result in $yoloResults) {
    if ([string]$result.status -eq "passed_with_findings") {
        $findings += "YOLO $($result.name): $($result.findings -join ', ')"
    } elseif ([string]$result.status -notin @("passed", "passed_with_findings")) {
        $findings += "YOLO $($result.name): $($result.status) $($result.reason)"
    }
}
foreach ($result in @($ocrResults + $ocrSystemResults)) {
    if ([string]$result.status -ne "passed") {
        $reason = if ($result.PSObject.Properties.Name -contains "reason") { [string]$result.reason } else { [string]$result.failure }
        $findings += "OCR $($result.name): $($result.status) $reason"
    }
}

$status = Get-OverallStatus -YoloResults $yoloResults -OcrResults $ocrResults -SystemResults $ocrSystemResults
$finishedAt = [DateTime]::UtcNow
$summary = [ordered]@{
    ok = ($status -in @("passed", "passed_with_findings"))
    status = $status
    runId = $script:RunId
    mode = $mode
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    elapsedSeconds = [Math]::Round(($finishedAt - $script:StartedAt).TotalSeconds, 3)
    workDir = $work
    parameters = [ordered]@{
        epochs = $script:EffectiveEpochs
        requestedEpochs = $Epochs
        device = $Device
        imageSize = $ImageSize
        batch = $Batch
        deploymentTargets = $targets
        resume = [bool]$Resume
        maxParallel = $MaxParallel
        ocrUseGpu = [bool]$script:OcrUseGpu
    }
    reports = [ordered]@{
        environment = $envReportPath
        datasets = $datasetsPath
        matrixPlan = Join-Path $work "matrix_plan.json"
    }
    counts = [ordered]@{
        yoloTotal = $yoloCases.Count
        yoloPassed = @($yoloResults | Where-Object { [string]$_.status -in @("passed", "passed_with_findings") }).Count
        ocrTotal = @($ocrDetCases + $ocrRecCases).Count
        ocrPassed = @($ocrResults | Where-Object { [string]$_.status -eq "passed" }).Count
        ocrSystemTotal = $ocrSystemCases.Count
        ocrSystemPassed = @($ocrSystemResults | Where-Object { [string]$_.status -eq "passed" }).Count
    }
    yoloResults = $yoloResults
    ocrResults = $ocrResults
    ocrSystemResults = $ocrSystemResults
    findings = $findings
    note = "Public-data engineering lifecycle evidence only. Do not claim customer-domain OCR production readiness or benchmark accuracy from this run."
}
$summaryPath = Join-Path $work "full_model_lifecycle_summary.json"
$summaryMarkdownPath = Join-Path $work "full_model_lifecycle_summary.md"
Write-JsonFile -Path $summaryPath -Value $summary
Write-MarkdownSummary -Path $summaryMarkdownPath -Summary $summary
Write-JsonFile -Path (Join-Path $work "current_run.json") -Value ([ordered]@{
    runId = $script:RunId
    status = $status
    mode = $mode
    pid = $PID
    startedAt = $script:StartedAt.ToString("o")
    finishedAt = $finishedAt.ToString("o")
    workDir = $work
    summaryPath = $summaryPath
})

Write-Host "Full lifecycle summary: $summaryPath" -ForegroundColor Cyan
Write-Host "Full lifecycle markdown: $summaryMarkdownPath" -ForegroundColor Cyan

if ($status -eq "failed") {
    Write-Host "Full lifecycle matrix failed. See $summaryPath" -ForegroundColor Red
    exit 1
}
if ($status -eq "blocked") {
    Write-Host "Full lifecycle matrix is blocked. See $summaryPath" -ForegroundColor Yellow
    exit 2
}
Write-Host "Full lifecycle matrix completed with status=$status" -ForegroundColor Green
