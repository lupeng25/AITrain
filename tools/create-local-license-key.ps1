param(
    [string]$OutputPath,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $root ".deps\local-license\aitrain-license-private-key.json"
}

$outputFullPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $OutputPath))
}

if ((Test-Path -LiteralPath $outputFullPath) -and -not $Force) {
    $existing = Get-Content -LiteralPath $outputFullPath -Raw | ConvertFrom-Json
    if (-not $existing.publicKey -or -not $existing.privateKey) {
        throw "Existing key file is missing publicKey/privateKey: $outputFullPath"
    }
    Write-Host "Local license key already exists: $outputFullPath"
    Write-Host ("Public key length: {0}" -f ([string]$existing.publicKey).Length)
    return
}

function ConvertTo-UInt32LittleEndianBytes {
    param([UInt32]$Value)
    return [BitConverter]::GetBytes($Value)
}

function ConvertTo-Fixed32Bytes {
    param(
        [byte[]]$Bytes,
        [string]$Name
    )

    if (-not $Bytes -or $Bytes.Length -eq 0) {
        throw "Missing EC parameter: $Name"
    }
    if ($Bytes.Length -eq 32) {
        return $Bytes
    }
    if ($Bytes.Length -gt 32) {
        return $Bytes[($Bytes.Length - 32)..($Bytes.Length - 1)]
    }

    $output = New-Object byte[] 32
    [Array]::Copy($Bytes, 0, $output, 32 - $Bytes.Length, $Bytes.Length)
    return $output
}

$directory = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null

$ecdsa = [System.Security.Cryptography.ECDsa]::Create([System.Security.Cryptography.ECCurve+NamedCurves]::nistP256)
try {
    $parameters = $ecdsa.ExportParameters($true)
    [byte[]]$x = ConvertTo-Fixed32Bytes -Bytes $parameters.Q.X -Name "x"
    [byte[]]$y = ConvertTo-Fixed32Bytes -Bytes $parameters.Q.Y -Name "y"
    [byte[]]$d = ConvertTo-Fixed32Bytes -Bytes $parameters.D -Name "d"

    [byte[]]$publicBlob = @()
    $publicBlob += ConvertTo-UInt32LittleEndianBytes 0x31534345
    $publicBlob += ConvertTo-UInt32LittleEndianBytes 32
    $publicBlob += $x
    $publicBlob += $y

    [byte[]]$privateBlob = @()
    $privateBlob += ConvertTo-UInt32LittleEndianBytes 0x32534345
    $privateBlob += ConvertTo-UInt32LittleEndianBytes 32
    $privateBlob += $x
    $privateBlob += $y
    $privateBlob += $d

    $publicKey = [Convert]::ToBase64String($publicBlob)
    $privateKey = [Convert]::ToBase64String($privateBlob)
    $payload = [ordered]@{
        type = "aitrain-license-key"
        curve = "P-256"
        createdAt = [DateTime]::UtcNow.ToString("o")
        publicKey = $publicKey
        privateKey = $privateKey
    } | ConvertTo-Json

    Set-Content -LiteralPath $outputFullPath -Value $payload -Encoding UTF8
    Write-Host "Generated local license key: $outputFullPath"
    Write-Host ("Public key length: {0}" -f $publicKey.Length)
} finally {
    $ecdsa.Dispose()
}
