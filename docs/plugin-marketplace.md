# AITrain Studio Plugin Marketplace

AITrain Studio plugin marketplace v1 is local and offline first. It installs trusted plugin packages into the application plugin area, then lets the existing Qt `IModelPlugin` loader expose those capabilities to the workbench.

## Scope

- Supported sources: local `marketplace.json`, expanded package directories, and `.aitrain-plugin.zip` packages.
- Supported install target: `plugins/marketplace/<plugin-id>/<version>`.
- Supported active plugin target: `plugins/models`.
- Supported runtime interface: existing `IModelPlugin` / `com.aitrainstudio.IModelPlugin/1.0`.
- Signature status: v1 records unsigned local packages as `verified-local-unsigned`; mandatory publisher signing is reserved for a later release.

The marketplace does not add accounts, ratings, payment, remote code execution, cloud scheduling, or new training algorithms. Training, evaluation, export, inference, and reports still run through the existing core, Worker, plugin, and Python trainer boundaries.

## User Flow

1. Open the Plugins page.
2. Use `Load Index` to read `plugins/marketplace/marketplace.json` or a local/static index path.
3. Use `Import Plugin Package` to select a `.aitrain-plugin.zip` file.
4. The marketplace validates `plugin.json`, package layout, hashes, compatibility, and Qt plugin identity.
5. When enabled, the Qt plugin DLL is copied into `plugins/models`.
6. Run plugin rescan; enabled plugins appear in the existing plugin matrix and training controls.

## State

The local state file is:

```text
plugins/marketplace/plugin_marketplace_state.json
```

It records installed plugin id, version, enabled state, install path, source path, verification status, active files, and the package manifest. This is application-level state, not project SQLite metadata.

## Safety Rules

- One plugin id can have multiple installed versions, but only one enabled version.
- Disable removes active DLL entries from `plugins/models` but keeps the installed package.
- Uninstall removes only the marketplace install directory and state record.
- User projects, datasets, model artifacts, training runs, and `.deps` environments are never removed by marketplace actions.
- Incompatible packages are rejected before activation.

## Verification

Run:

```powershell
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
```

Worker plugin smoke includes marketplace state in its JSON output:

```powershell
.\build-vscode\bin\aitrain_worker.exe --plugin-smoke .\build-vscode\plugins\models
```

After copying a real plugin DLL into an expanded package directory and updating `plugin.json`, create a local test package with:

```powershell
.\tools\create-plugin-package.ps1 -PackageRoot .\examples\plugin-package-template -OutputPath .\.deps\Example.aitrain-plugin.zip -Force
```

For GUI validation, generate a disposable marketplace index and real zip package from the built-in Dataset Interop plugin:

```powershell
.\tools\create-plugin-marketplace-demo.ps1 -Force
```

Then load `.deps\plugin-marketplace-demo\marketplace.json` from the Plugins page. This demo package reuses the built-in `com.aitrain.plugins.dataset_interop` id and DLL name, so use it only in development or disposable validation layouts.
