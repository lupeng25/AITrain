param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,

    [string]$OutputPath = "",

    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath($PackageRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "Package root does not exist: $root"
}

$manifestPath = Join-Path $root "plugin.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing plugin.json in package root: $root"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $manifest.id) {
    throw "plugin.json id is required."
}
if (-not $manifest.version) {
    throw "plugin.json version is required."
}
if (-not $manifest.entrypoints -or -not $manifest.entrypoints.qtModelPlugin) {
    throw "plugin.json entrypoints.qtModelPlugin is required."
}

if (-not $manifest.files) {
    $manifest | Add-Member -NotePropertyName "files" -NotePropertyValue @($manifest.entrypoints.qtModelPlugin)
}

$hashes = [ordered]@{}
foreach ($relativePath in @($manifest.files)) {
    if (-not $relativePath) {
        continue
    }
    $normalizedRelativePath = ([string]$relativePath).Replace("\", "/")
    $filePath = Join-Path $root $normalizedRelativePath
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        throw "Listed package file is missing: $normalizedRelativePath"
    }
    $hashes[$normalizedRelativePath] = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
}
$manifest.hashes = [pscustomobject]$hashes
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $safeId = ([string]$manifest.id) -replace "[^A-Za-z0-9_.-]", "_"
    $safeVersion = ([string]$manifest.version) -replace "[^A-Za-z0-9_.-]", "_"
    $OutputPath = Join-Path (Split-Path -Parent $root) "$safeId-$safeVersion.aitrain-plugin.zip"
}
$zipPath = [System.IO.Path]::GetFullPath($OutputPath)
if ((Test-Path -LiteralPath $zipPath) -and -not $Force) {
    throw "Output package already exists: $zipPath. Use -Force to overwrite."
}

$requiredLicense = Join-Path $root "LICENSE"
$requiredPayload = Join-Path $root "payload"
if (-not (Test-Path -LiteralPath $requiredLicense -PathType Leaf)) {
    throw "Missing required LICENSE file."
}
if (-not (Test-Path -LiteralPath $requiredPayload -PathType Container)) {
    throw "Missing required payload directory."
}

$zipParent = Split-Path -Parent $zipPath
if (-not (Test-Path -LiteralPath $zipParent)) {
    New-Item -ItemType Directory -Path $zipParent | Out-Null
}
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $root "*") -DestinationPath $zipPath -Force
$packageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()

Write-Host "Plugin package created:" -ForegroundColor Green
Write-Host "  $zipPath"
Write-Host "Package SHA256:"
Write-Host "  $packageHash"
