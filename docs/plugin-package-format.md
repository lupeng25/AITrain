# AITrain Plugin Package Format

Plugin packages use the `.aitrain-plugin.zip` extension. The same layout can also be used as an expanded directory during local development and tests.

## Required Layout

```text
plugin.json
LICENSE
payload/
  plugins/
    models/
      YourPlugin.dll
README.md        optional
examples/         optional
```

The Qt DLL must implement the existing `IModelPlugin` interface and declare the same plugin id as `plugin.json`.

## plugin.json

```json
{
  "schemaVersion": 1,
  "id": "com.example.aitrain.plugin",
  "name": "Example Plugin",
  "version": "0.1.0",
  "description": "Short plugin description.",
  "publisher": "Example Publisher",
  "license": "Proprietary or SPDX id",
  "category": "dataset_interop",
  "capabilities": ["dataset_interop"],
  "entrypoints": {
    "qtModelPlugin": "payload/plugins/models/YourPlugin.dll"
  },
  "compatibility": {
    "minAitrainVersion": "0.1.0",
    "qtAbi": "Qt5.12",
    "msvcRuntime": "msvc1950",
    "requiresGpu": false
  },
  "files": [
    "payload/plugins/models/YourPlugin.dll"
  ],
  "hashes": {
    "payload/plugins/models/YourPlugin.dll": "<sha256>"
  }
}
```

## Categories

The v1 marketplace recognizes these category values:

- `model_backend`
- `dataset_interop`
- `exporter`
- `postprocess`
- `report_template`
- `pipeline_template`

Resource-only report and pipeline packages can be listed by the marketplace, but v1 does not add new C++ plugin interfaces for them. Runtime model capabilities continue to use `IModelPlugin`.

## Compatibility

- `minAitrainVersion` uses semantic version comparison.
- `qtAbi` must match the current application ABI, for example `Qt5.12`.
- `msvcRuntime` is recorded for diagnostics.
- `requiresGpu` is displayed to users but does not bypass hardware checks.

## Signing

Mandatory signature verification is reserved for a later release. v1 records manually imported local packages as unsigned local packages and shows that status in the installed plugin table.

## Build a Package

After placing a real Qt plugin DLL under `payload/plugins/models/` and updating `plugin.json` so `id` matches the DLL manifest id, use the helper script to refresh file hashes and create the zip:

```powershell
.\tools\create-plugin-package.ps1 -PackageRoot .\examples\plugin-package-template -OutputPath .\.deps\Example.aitrain-plugin.zip -Force
```

The script validates `plugin.json`, `LICENSE`, `payload/`, every listed file, updates `hashes`, and writes a `.aitrain-plugin.zip` package.
