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

## Inno Setup Installer

The repo also includes an Inno Setup 6 script for producing a Windows installer from the same verified package layout used by package smoke:

```powershell
.\tools\build-inno-installer.ps1
```

The command refreshes `build-vscode\package-smoke`, verifies the layout, locates `ISCC.exe`, and writes:

- `build-vscode\inno\AITrainStudio-0.1.0-Setup.exe`

For a faster compile after `package-smoke` has already passed:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke
```

For a smaller CPU/ONNX installer that leaves TensorRT redistribution to the ZIP package or a separate GPU bundle:

```powershell
.\tools\build-inno-installer.ps1 -SkipPackageSmoke -ExcludeTensorRt
```

The installer wraps the package-smoke directory only. It does not add `.deps`, generated datasets, downloaded tools, model weights, ONNX smoke outputs, TensorRT engines, or external acceptance evidence.

## External Follow-Up

Send the generated ZIP package plus:

- `docs\external-acceptance-handoff.md`
- `docs\acceptance-templates\clean-windows-acceptance-result.md`
- `docs\acceptance-templates\tensorrt-acceptance-result.md`
- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

Then collect the filled templates and evidence before updating `docs\harness\current-status.md`.
