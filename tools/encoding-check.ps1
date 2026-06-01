param(
    [switch]$IncludeUntracked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$binaryExtensions = @(
    ".bin",
    ".bmp",
    ".dll",
    ".docx",
    ".engine",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".lib",
    ".onnx",
    ".pdb",
    ".pdparams",
    ".pdf",
    ".png",
    ".pptx",
    ".psd",
    ".pt",
    ".qm",
    ".tar",
    ".webp",
    ".xlsx",
    ".7z",
    ".zip"
)

function Test-Utf8Valid {
    param([byte[]]$Bytes)

    $i = 0
    while ($i -lt $Bytes.Length) {
        $b = [int]$Bytes[$i]
        if ($b -le 0x7F) {
            $i++
            continue
        }

        if (($b -band 0xE0) -eq 0xC0) {
            if ($b -lt 0xC2 -or $i + 1 -ge $Bytes.Length) {
                return $false
            }
            if ((([int]$Bytes[$i + 1]) -band 0xC0) -ne 0x80) {
                return $false
            }
            $i += 2
            continue
        }

        if (($b -band 0xF0) -eq 0xE0) {
            if ($i + 2 -ge $Bytes.Length) {
                return $false
            }
            $b1 = [int]$Bytes[$i + 1]
            $b2 = [int]$Bytes[$i + 2]
            if (($b1 -band 0xC0) -ne 0x80 -or ($b2 -band 0xC0) -ne 0x80) {
                return $false
            }
            if ($b -eq 0xE0 -and $b1 -lt 0xA0) {
                return $false
            }
            if ($b -eq 0xED -and $b1 -ge 0xA0) {
                return $false
            }
            $i += 3
            continue
        }

        if (($b -band 0xF8) -eq 0xF0) {
            if ($i + 3 -ge $Bytes.Length) {
                return $false
            }
            $b1 = [int]$Bytes[$i + 1]
            $b2 = [int]$Bytes[$i + 2]
            $b3 = [int]$Bytes[$i + 3]
            if (($b1 -band 0xC0) -ne 0x80 -or
                ($b2 -band 0xC0) -ne 0x80 -or
                ($b3 -band 0xC0) -ne 0x80) {
                return $false
            }
            if ($b -eq 0xF0 -and $b1 -lt 0x90) {
                return $false
            }
            if ($b -eq 0xF4 -and $b1 -ge 0x90) {
                return $false
            }
            if ($b -gt 0xF4) {
                return $false
            }
            $i += 4
            continue
        }

        return $false
    }

    return $true
}

function Get-TrackedFileEncoding {
    param([string]$Path)

    $fullPath = (Resolve-Path -LiteralPath $Path).ProviderPath
    $bytes = [System.IO.File]::ReadAllBytes($fullPath)
    $length = $bytes.Length
    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    $isKnownBinary = $binaryExtensions -contains $extension

    if ($length -eq 0) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "Empty"
            IsProblem = $false
            Detail = ""
        }
    }

    if ($length -ge 4 -and
        $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE -and
        $bytes[2] -eq 0x00 -and $bytes[3] -eq 0x00) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "UTF-32 LE BOM"
            IsProblem = $true
            Detail = "Text must be UTF-8, not UTF-32."
        }
    }

    if ($length -ge 4 -and
        $bytes[0] -eq 0x00 -and $bytes[1] -eq 0x00 -and
        $bytes[2] -eq 0xFE -and $bytes[3] -eq 0xFF) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "UTF-32 BE BOM"
            IsProblem = $true
            Detail = "Text must be UTF-8, not UTF-32."
        }
    }

    if ($length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "UTF-16 LE BOM"
            IsProblem = $true
            Detail = "Text must be UTF-8, not UTF-16."
        }
    }

    if ($length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "UTF-16 BE BOM"
            IsProblem = $true
            Detail = "Text must be UTF-8, not UTF-16."
        }
    }

    $nulCount = 0
    foreach ($byte in $bytes) {
        if ($byte -eq 0) {
            $nulCount++
        }
    }

    if ($nulCount -gt 0) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "Binary or UTF-16 without BOM"
            IsProblem = (-not $isKnownBinary)
            Detail = if ($isKnownBinary) { "Known binary file." } else { "NUL bytes found in a non-binary path." }
        }
    }

    if (-not (Test-Utf8Valid -Bytes $bytes)) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "Non-UTF8 no BOM"
            IsProblem = $true
            Detail = "Text is not valid UTF-8."
        }
    }

    if ($length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        return [pscustomobject]@{
            Path = $Path
            Encoding = "UTF-8 BOM"
            IsProblem = $false
            Detail = ""
        }
    }

    $hasNonAscii = $false
    foreach ($byte in $bytes) {
        if ($byte -gt 0x7F) {
            $hasNonAscii = $true
            break
        }
    }

    return [pscustomobject]@{
        Path = $Path
        Encoding = if ($hasNonAscii) { "UTF-8 no BOM" } else { "ASCII subset (valid UTF-8)" }
        IsProblem = $false
        Detail = ""
    }
}

$files = @(git -c core.quotepath=false ls-files)
if ($IncludeUntracked) {
    $files += @(git -c core.quotepath=false ls-files --others --exclude-standard)
    $files = @($files | Sort-Object -Unique)
}

if ($files.Count -eq 0) {
    throw "No Git files found to check."
}

$results = foreach ($file in $files) {
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        Get-TrackedFileEncoding -Path $file
    }
}

Write-Host "Encoding check summary:" -ForegroundColor Cyan
$results |
    Group-Object Encoding |
    Sort-Object Count -Descending |
    ForEach-Object {
        Write-Host ("  {0}: {1}" -f $_.Name, $_.Count)
    }

$nonAsciiPaths = @($results | Where-Object { $_.Path -cmatch "[^\x00-\x7F]" })
if ($nonAsciiPaths.Count -gt 0) {
    Write-Host "Non-ASCII Git paths:" -ForegroundColor Cyan
    foreach ($entry in $nonAsciiPaths) {
        Write-Host ("  {0}" -f $entry.Path)
    }
}

$problems = @($results | Where-Object { $_.IsProblem })
if ($problems.Count -gt 0) {
    Write-Host "Encoding problems:" -ForegroundColor Red
    foreach ($problem in $problems) {
        Write-Host ("  {0}: {1} ({2})" -f $problem.Path, $problem.Encoding, $problem.Detail) -ForegroundColor Red
    }
    throw "Encoding check failed."
}

Write-Host "Encoding check passed." -ForegroundColor Green
