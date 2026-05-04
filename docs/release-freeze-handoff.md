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

## External Follow-Up

Send the generated ZIP package plus:

- `docs\external-acceptance-handoff.md`
- `docs\acceptance-templates\clean-windows-acceptance-result.md`
- `docs\acceptance-templates\tensorrt-acceptance-result.md`
- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

Then collect the filled templates and evidence before updating `docs\harness\current-status.md`.
