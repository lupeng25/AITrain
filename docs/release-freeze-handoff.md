# AITrain Studio Release Freeze Handoff

This checklist is the Phase 44 Lite local release-freeze handoff path. It creates a traceable package identity before external clean Windows and RTX / SM 75+ TensorRT acceptance.

## Scope

Release freeze handoff covers:

- Local RC closeout gate.
- CPack ZIP generation.
- SHA256 hash recording for generated packages.
- Source commit and dirty-worktree recording.
- Inclusion of external acceptance docs and templates in the handoff manifest.

It does not mark clean Windows acceptance or TensorRT acceptance as passed. Those statuses require returned evidence from the external machines described in `docs\external-acceptance-handoff.md`.

## Command

From the repository root:

```powershell
.\tools\release-freeze-handoff.ps1
```

For a faster manifest refresh after gates have already passed:

```powershell
.\tools\release-freeze-handoff.ps1 -SkipLocalRc
```

## Outputs

The script writes:

- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

The manifest records:

- UTC timestamp.
- Source commit.
- Whether the worktree was dirty.
- Build directory.
- Generated ZIP package paths.
- SHA256 hashes.
- External acceptance docs/templates to send with the package.
- Required external result artifacts to return.

## Inno Setup Installers

The repo also includes Inno Setup 6 scripts for producing Windows installers from the same verified package layout used by package smoke. The default installer build is split into three files:

- Product installer: AITrain Studio executables, built-in plugins, product scripts, docs, examples, translations, and Python trainer adapter files.
- Dependency installer: Qt/VC runtime files, Qt runtime plugin folders, ONNX Runtime, NCNN, TensorRT, and other runtime dependency DLLs.
- Python AI environment installer: a prepared Python environment for official YOLO/OBB, SMP, Anomalib, and OCR adapters plus an optional PaddleOCR source checkout under `python_env\PaddleOCR`.

```powershell
.\tools\build-inno-installer.ps1
```

The command refreshes `build-vscode\package-smoke`, verifies the layout, locates `ISCC.exe`, and writes:

- `build-vscode\inno\AITrainStudio-0.1.0-Product-Setup.exe`
- `build-vscode\inno\AITrainStudio-0.1.0-Dependencies-Setup.exe`
- `build-vscode\inno\AITrainStudio-0.1.0-PythonEnv-Setup.exe`

For a faster compile after `package-smoke` has already passed:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke
```

To build only one side of the split:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -PackageMode Product
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -PackageMode Dependencies
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -PackageMode PythonEnv -PythonEnvSourceDir <prepared-python-env-root> -PaddleOcrSourceDir <paddleocr-source-root>
```

For a smaller dependency installer that leaves TensorRT redistribution to a separate GPU bundle:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -PackageMode Dependencies -ExcludeTensorRt
```

For legacy all-in-one installer output:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -PackageMode Full
```

The product and native dependency installers wrap the package-smoke directory only. The Python AI environment installer wraps only the explicitly selected Python environment source and optional PaddleOCR source checkout; it must not include datasets, model weights, run outputs, or external acceptance evidence. The PaddleOCR `.git` directory and Python cache files are excluded by the installer script.

Install all three packages into the same target directory. They write to separate roots and should not overwrite one another:

- Product: application root, `plugins`, `docs`, `examples`, `python_trainers`, `tools`, `translations`.
- Native dependencies: application root runtime DLLs, Qt runtime plugin folders, `runtimes`.
- Python AI environment: `python_env` and optional `python_env\PaddleOCR`.

The product installer is not self-contained until the dependency installer and Python AI environment installer, or equivalent runtime dependencies, are installed into the same target directory. For clean-machine release candidates, build the Python AI environment package from a prepared relocatable staging directory rather than an ad hoc developer venv.

## External Follow-Up

Send the generated ZIP package plus:

- `docs\external-acceptance-handoff.md`
- `docs\acceptance-templates\clean-windows-acceptance-result.md`
- `docs\acceptance-templates\tensorrt-acceptance-result.md`
- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

Then collect the filled templates and evidence before updating `docs\harness\current-status.md`.
