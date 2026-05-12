# Next Stage RC Delivery Plan

Last updated: 2026-05-12

This document is the execution plan for the stage after RTX 4090 D validation. It does not replace `docs/harness/current-status.md` as the project status source of truth.

## Summary

- Goal: freeze the current RTX 4090 D validation result into a release-candidate baseline and prepare a traceable local handoff package. External clean-machine/package-root collection is deferred for this lane unless explicitly reopened.
- Current baseline: RTX 4090 D validation is complete. TensorRT, Phase 45 YOLO11/YOLO12 matrix, Phase 47 PaddleOCR Det ONNX, and Production OCR acceptance all have passing evidence under `.deps\rtx4090-validation`. The current local RC handoff was refreshed from commit `6dac361b83de78f38853e0649fd00f1355b7ccd7`; after the offline-license public-key rebuild, the ZIP SHA256 is `206C10CDB291D8D9A0FA5B5E8B51D30F07CD22B695EE8206E41ED5D08BA70F03`.
- Production OCR gate: the current default gate accepts Rec when `accuracy > 0.7`; CER is recorded but not blocking unless `-RequireRecCer` is used.
- Artifact rule: do not commit `.deps`, build outputs, datasets, model weights, ONNX files, TensorRT engines, generated ZIP files, or other generated binaries.

## Execution Plan

1. Close the current diff.
   - Review the current working tree and confirm only intended gate/documentation/script changes are present.
   - Run `git diff --check`.
   - Run `.\tools\harness-check.ps1`.
   - Commit only source-controlled script and documentation changes after validation.

2. Run the local RC gate.
   - Run `.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode`.
   - Run `.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild -WorkDir .deps\release-rc\acceptance-local-package`.
   - Require every summary JSON to report `passed`.

3. Generate the release handoff package.
   - Run `.\tools\release-freeze-handoff.ps1`.
   - Record the ZIP path, SHA256, commit HEAD, generation timestamp, and dirty-worktree flag from the generated manifest.
   - If the manifest reports a dirty worktree, resolve it before treating the package as a final RC handoff.

4. Defer the external acceptance bundle for this lane.
   - Include the generated ZIP package.
   - Include `build-vscode\release-freeze-handoff\release_handoff_manifest.json`.
   - Include `build-vscode\release-freeze-handoff\release_handoff_summary.md`.
   - Include `docs\external-acceptance-handoff.md`.
   - Include the templates under `docs\acceptance-templates\`.
   - Do not send the bundle unless a new priority explicitly reopens external collection.

5. Defer clean-machine package acceptance for this lane.
   - On a clean Windows machine, unzip the package and run `.\tools\acceptance-smoke.ps1 -Package` from the package root.
   - Collect `acceptance_summary.json`, full console output, package layout evidence, and the filled `clean-windows-acceptance-result.md` template.
   - Mark clean-machine acceptance as passed only when the package-root summary is `passed`.
   - Do not mark this passed without returned external evidence.

6. Defer optional package-root TensorRT acceptance for this lane.
   - On an RTX / SM 75+ machine, run `.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt` from the package root.
   - Collect GPU model, driver, CUDA runtime, Worker self-check JSON, TensorRT console output, and the filled `tensorrt-acceptance-result.md` template.
   - Keep GTX 1060 / SM 61 results as `hardware-blocked`; they must not override the RTX 4090 D passing evidence.
   - Do not mark this passed without returned external evidence.

7. Update status after returned evidence is reviewed.
   - If all required external evidence passes, update `docs/harness/current-status.md` to mark the RC handoff and clean-machine package acceptance as passed.
   - If any external check is blocked or failed, record the blocker and evidence without weakening gates.
   - Keep the RTX 4090 D validation evidence path separate from external clean-machine package evidence.

## Acceptance Criteria

- Local validation passes:
  - `git diff --check`
  - `.\tools\harness-check.ps1`
  - `.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode`
  - `.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild`
- Release handoff exists and records:
  - ZIP path
  - SHA256
  - commit HEAD
  - generation timestamp
  - dirty-worktree status
- External clean-machine package acceptance is deferred and must not be marked `passed` without returned evidence.
- Optional TensorRT rerun is deferred and must not be marked `passed` without returned package-root RTX / SM 75+ evidence.
- No generated artifacts are added to source control.

## Risk Boundaries

- The current Production OCR result is accepted only under the lowered `accuracy > 0.7` gate. It is public Total-Text evidence, not customer-domain production OCR quality proof.
- The old `accuracy >= 0.90` and `CER <= 0.10` target is not the default blocking gate. If reinstated, it requires a separate model/data/training improvement plan.
- Phase 47 validates PaddleOCR Det ONNX conversion and C++ DB-style postprocess wiring, not PP-OCRv5 accuracy parity.
- Phase 45 validates YOLO detection and segmentation wiring only. It does not add classification, pose, OBB, anomaly, YOLO-World, or YOLOE support.
- Plugin marketplace v1 remains local/offline-first and unsigned unless a later phase explicitly adds publisher signature enforcement.
- External acceptance is deferred for this lane. Reopening it requires returned evidence before changing status to passed.

## Verification Commands

For this document-only change:

```powershell
git diff --check
.\tools\harness-context.ps1
```

Before final RC handoff:

```powershell
.\tools\harness-check.ps1
.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke -BuildDir build-vscode
.\tools\release-freeze-handoff.ps1
```
