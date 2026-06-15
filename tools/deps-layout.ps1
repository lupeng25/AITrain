Set-StrictMode -Version Latest

function Get-AITrainRepoRoot {
    param([string]$Root = "")

    if (-not [string]::IsNullOrWhiteSpace($Root)) {
        return [System.IO.Path]::GetFullPath($Root)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $PSScriptRoot) "."))
}

function Resolve-AITrainRepoPath {
    param(
        [string]$Path,
        [string]$Root = ""
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-AITrainRepoRoot -Root $Root) $Path))
}

function Get-AITrainDepsLayout {
    param([string]$Root = "")

    $repoRoot = Get-AITrainRepoRoot -Root $Root
    $depsRoot = Join-Path $repoRoot ".deps"
    $envRoot = Join-Path $depsRoot "envs"
    $reposRoot = Join-Path $depsRoot "repos"
    $sdkRoot = Join-Path $depsRoot "sdks"
    $archivesRoot = Join-Path $depsRoot "archives"
    $toolsRoot = Join-Path $depsRoot "tools"
    $uiWalkthroughRoot = Join-Path $depsRoot "UI-Walkthrough"

    return [pscustomobject][ordered]@{
        RepoRoot = $repoRoot
        DepsRoot = $depsRoot
        EnvRoot = $envRoot
        ReposRoot = $reposRoot
        SdkRoot = $sdkRoot
        ArchivesRoot = $archivesRoot
        ToolsRoot = $toolsRoot
        UiWalkthroughRoot = $uiWalkthroughRoot
        PythonEmbed = Join-Path $envRoot "python-embed-3.13.13"
        OcrCpuEnv = Join-Path $envRoot "ocr-cpu"
        OcrGpuEnv = Join-Path $envRoot "ocr-gpu"
        YoloCudaEnv = Join-Path $envRoot "yolo-cuda"
        Yolo26Env = Join-Path $envRoot "yolo26"
        Paddle2OnnxEnv = Join-Path $envRoot "paddle2onnx"
        PaddleOcrRepo = Join-Path $reposRoot "PaddleOCR"
        OnnxRuntimeRoot = Join-Path $sdkRoot "onnxruntime"
        NcnnRoot = Join-Path $sdkRoot "ncnn"
        TensorRtOssRoot = Join-Path $sdkRoot "tensorrt-oss"
        TensorRtRuntimeRoot = Join-Path $sdkRoot "tensorrt-runtime"
        AnnotationToolsRoot = Join-Path $toolsRoot "annotation-tools"
    }
}

function Join-AITrainPythonExecutableCandidates {
    param([string[]]$EnvironmentRoots)

    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($root in $EnvironmentRoots) {
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }
        $candidates.Add((Join-Path $root "Scripts\python.exe"))
        $candidates.Add((Join-Path $root "python.exe"))
    }
    return @($candidates)
}

function Get-AITrainPythonCandidates {
    param(
        [ValidateSet("General", "Yolo", "Ocr", "OcrCpu", "OcrGpu")]
        [string]$Role = "General",
        [string]$Root = ""
    )

    $layout = Get-AITrainDepsLayout -Root $Root
    $repoRoot = $layout.RepoRoot
    $legacyEmbed = Join-Path $repoRoot ".deps\python-3.13.13-embed-amd64"
    $legacyOcr = Join-Path $repoRoot ".deps\python-3.13.13-ocr-amd64"
    $legacyOcrGpu = Join-Path $repoRoot ".deps\rtx4090-validation\python-ocr-gpu"
    $legacyOcrCpu = Join-Path $repoRoot ".deps\rtx4090-validation\python-ocr"
    $legacyYoloCuda = Join-Path $repoRoot ".deps\rtx4090-validation\python-yolo-cuda"
    $legacyYoloVenv = Join-Path $repoRoot ".deps\rtx4090-validation\python-yolo-venv"

    switch ($Role) {
        "Yolo" {
            return Join-AITrainPythonExecutableCandidates @(
                $layout.YoloCudaEnv,
                $layout.Yolo26Env,
                $layout.PythonEmbed,
                $legacyYoloCuda,
                $legacyYoloVenv,
                $legacyEmbed,
                $legacyOcr
            )
        }
        "OcrGpu" {
            return Join-AITrainPythonExecutableCandidates @(
                $layout.OcrGpuEnv,
                $legacyOcrGpu,
                $legacyOcr,
                $layout.OcrCpuEnv,
                $layout.PythonEmbed,
                $legacyEmbed
            )
        }
        "OcrCpu" {
            return Join-AITrainPythonExecutableCandidates @(
                $layout.OcrCpuEnv,
                $legacyOcr,
                $layout.PythonEmbed,
                $legacyEmbed
            )
        }
        "Ocr" {
            return Join-AITrainPythonExecutableCandidates @(
                $layout.OcrGpuEnv,
                $layout.OcrCpuEnv,
                $legacyOcrGpu,
                $legacyOcrCpu,
                $legacyOcr,
                $layout.PythonEmbed,
                $legacyEmbed
            )
        }
        default {
            return Join-AITrainPythonExecutableCandidates @(
                $layout.PythonEmbed,
                $layout.OcrCpuEnv,
                $layout.OcrGpuEnv,
                $legacyEmbed,
                $legacyOcr
            )
        }
    }
}

function Resolve-AITrainExistingPython {
    param(
        [ValidateSet("General", "Yolo", "Ocr", "OcrCpu", "OcrGpu")]
        [string]$Role = "General",
        [string]$Root = "",
        [string]$RequestedPath = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $resolved = Resolve-AITrainRepoPath -Root $Root -Path $RequestedPath
        if (Test-Path -LiteralPath $resolved) {
            return $resolved
        }
        throw "Python executable was not found: $resolved"
    }

    foreach ($candidate in (Get-AITrainPythonCandidates -Role $Role -Root $Root)) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }
    return ""
}

function Get-AITrainPaddleOcrRepoCandidates {
    param([string]$Root = "")

    $layout = Get-AITrainDepsLayout -Root $Root
    $candidates = New-Object System.Collections.Generic.List[string]
    $envValue = [Environment]::GetEnvironmentVariable("AITRAIN_PADDLEOCR_REPO", "Process")
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $candidates.Add($envValue)
    }
    $envValue = [Environment]::GetEnvironmentVariable("AITRAIN_PADDLEOCR_ROOT", "Process")
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        $candidates.Add($envValue)
    }
    $candidates.Add($layout.PaddleOcrRepo)
    $candidates.Add((Join-Path $layout.RepoRoot ".deps\PaddleOCR"))
    return @($candidates)
}

function Resolve-AITrainPaddleOcrRepo {
    param(
        [string]$Root = "",
        [string]$RequestedPath = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $requested = Resolve-AITrainRepoPath -Root $Root -Path $RequestedPath
        if (Test-Path -LiteralPath (Join-Path $requested "tools\train.py")) {
            return $requested
        }
    }
    foreach ($candidate in (Get-AITrainPaddleOcrRepoCandidates -Root $Root)) {
        $resolved = Resolve-AITrainRepoPath -Root $Root -Path $candidate
        if (Test-Path -LiteralPath (Join-Path $resolved "tools\train.py")) {
            return $resolved
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        return Resolve-AITrainRepoPath -Root $Root -Path $RequestedPath
    }
    return (Get-AITrainDepsLayout -Root $Root).PaddleOcrRepo
}

function Get-AITrainPortablePythonZipCandidates {
    param([string]$Root = "")

    $layout = Get-AITrainDepsLayout -Root $Root
    return @(
        (Join-Path $layout.ArchivesRoot "python-3.13.13-embed-amd64.zip"),
        (Join-Path $layout.RepoRoot ".deps\python-3.13.13-embed-amd64.zip")
    )
}

function Get-AITrainGetPipCandidates {
    param([string]$Root = "")

    $layout = Get-AITrainDepsLayout -Root $Root
    return @(
        (Join-Path $layout.ArchivesRoot "get-pip.py"),
        (Join-Path $layout.RepoRoot ".deps\get-pip.py")
    )
}

function Resolve-AITrainFirstExistingPath {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    return ""
}
