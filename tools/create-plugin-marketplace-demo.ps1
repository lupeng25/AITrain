param(
    [string]$BuildDir = "build-vscode",
    [string]$OutputRoot = ".deps\plugin-marketplace-demo",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$buildPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
$packageRoot = Join-Path $outputPath "dataset-interop-package"
$packageZip = Join-Path $outputPath "com.aitrain.plugins.dataset_interop-0.1.0.aitrain-plugin.zip"
$marketplaceJson = Join-Path $outputPath "marketplace.json"
$demoDllName = "DatasetInteropMarketplaceDemo.dll"

$pluginCandidates = @(
    (Join-Path $buildPath "plugins\models\DatasetInteropPlugin.dll"),
    (Join-Path $buildPath "package-smoke\plugins\models\DatasetInteropPlugin.dll"),
    (Join-Path $repoRoot "build-vscode\plugins\models\DatasetInteropPlugin.dll")
)
$pluginDll = $pluginCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $pluginDll) {
    throw "DatasetInteropPlugin.dll was not found. Build the project first with .\tools\harness-check.ps1."
}

if ((Test-Path -LiteralPath $outputPath) -and $Force) {
    $fullOutput = [System.IO.Path]::GetFullPath($outputPath)
    $expectedParent = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".deps"))
    if (-not $fullOutput.StartsWith($expectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove demo output outside .deps: $fullOutput"
    }
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}

New-Item -ItemType Directory -Path (Join-Path $packageRoot "payload\plugins\models") -Force | Out-Null
Copy-Item -LiteralPath $pluginDll -Destination (Join-Path $packageRoot "payload\plugins\models\$demoDllName") -Force

@"
AITrain Studio demo package license.

This package reuses the built-in DatasetInteropPlugin.dll for local marketplace smoke and GUI validation only.
"@ | Set-Content -LiteralPath (Join-Path $packageRoot "LICENSE") -Encoding UTF8

@"
# Dataset Interop Marketplace Demo

This package reuses the built-in DatasetInteropPlugin.dll under a demo-only DLL filename so the marketplace can be validated with a real Qt plugin package without overwriting the built-in DLL.

Use only in development or disposable validation layouts. It has the same plugin id as the built-in plugin.
"@ | Set-Content -LiteralPath (Join-Path $packageRoot "README.md") -Encoding UTF8

$manifest = [ordered]@{
    schemaVersion = 1
    id = "com.aitrain.plugins.dataset_interop"
    name = "Dataset Interop Marketplace Demo"
    version = "0.1.0"
    description = "Development demo package that reuses the built-in Dataset Interop Qt plugin."
    publisher = "AITrain Studio"
    license = "Demo"
    category = "dataset_interop"
    capabilities = @("dataset_interop")
    entrypoints = @{
        qtModelPlugin = "payload/plugins/models/$demoDllName"
    }
    compatibility = @{
        minAitrainVersion = "0.1.0"
        qtAbi = "Qt5.12"
        msvcRuntime = "msvc1950"
        requiresGpu = $false
    }
    files = @("payload/plugins/models/$demoDllName")
    hashes = @{
        "payload/plugins/models/$demoDllName" = "replace-with-sha256"
    }
}
$manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $packageRoot "plugin.json") -Encoding UTF8

& (Join-Path $repoRoot "tools\create-plugin-package.ps1") -PackageRoot $packageRoot -OutputPath $packageZip -Force:$Force
if (-not (Test-Path -LiteralPath $packageZip -PathType Leaf)) {
    throw "create-plugin-package.ps1 did not create the expected package: $packageZip"
}

$packageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $packageZip).Hash.ToLowerInvariant()
$marketplace = [ordered]@{
    schemaVersion = 1
    generatedBy = "tools/create-plugin-marketplace-demo.ps1"
    generatedAt = (Get-Date).ToUniversalTime().ToString("o")
    plugins = @(
        [ordered]@{
            id = "com.aitrain.plugins.dataset_interop"
            name = "Dataset Interop Marketplace Demo"
            version = "0.1.0"
            description = "Development demo package that reuses the built-in Dataset Interop Qt plugin."
            publisher = "AITrain Studio"
            license = "Demo"
            category = "dataset_interop"
            capabilities = @("dataset_interop")
            minAitrainVersion = "0.1.0"
            qtAbi = "Qt5.12"
            msvcRuntime = "msvc1950"
            requiresGpu = $false
            dependencies = @()
            packageSha256 = $packageHash
            signature = ""
            downloadUrl = $packageZip
        }
    )
}
$marketplace | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $marketplaceJson -Encoding UTF8

Write-Host "Plugin marketplace demo created:" -ForegroundColor Green
Write-Host "  Package: $packageZip"
Write-Host "  Index:   $marketplaceJson"
Write-Host "  SHA256:  $packageHash"
Write-Host ""
Write-Host "Note: this demo reuses the built-in DatasetInteropPlugin id. Use it for development validation only." -ForegroundColor Yellow
