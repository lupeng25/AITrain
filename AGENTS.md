# AITrain Studio Agent Instructions

This repository uses a harness-first workflow. In every new AI coding conversation, read these files before making a plan or editing code:

1. `HARNESS.md`
2. `docs/harness/current-status.md`
3. `docs/harness/project-context.md`
4. `docs/harness/implementation-checklist.md`
5. `docs/harness/quality-gates.md`

If the task touches UI, also read:

6. `docs/harness/ui-guidelines.md`

If the task is broad or implementation-heavy, also inspect:

7. `docs/product-roadmap-local-training-platform.md`

Archived roadmap notes are not current implementation plans. Use `docs/harness/current-status.md` for phase status and `docs/product-roadmap-local-training-platform.md` for the current broad direction.

## Operating Rules

- Do not start implementation from memory. Ground in the harness files first.
- Keep changes scoped to the task.
- Preserve the current Qt Widgets workbench architecture:
  - left sidebar
  - top status bar
  - central `QStackedWidget`
  - `AppStyle`, `Sidebar`, `InfoPanel`, `StatusPill`
- Do not put long-running work in the GUI thread.
- Do not put model training logic in `MainWindow`.
- Use Worker messages for long tasks.
- Use `ProjectStore` for SQLite metadata; GUI reads go through Query Service/Presenter and writes go through `ProjectWorkspace`/Worker.
- Use the compile-time `CapabilityRegistry` and explicit Worker/core adapters for model, dataset, export, inference, and validation extensions; do not reintroduce dynamic plugin interfaces.
- Keep Qt 5.12+ compatibility unless explicitly asked to upgrade.
- Use `QStringLiteral` for UI text.
- Avoid Chinese mojibake. Source files must compile with UTF-8.
- Treat project text files as UTF-8. In Windows PowerShell, read Chinese or mixed-language files with an explicit encoding, for example `Get-Content -Encoding UTF8`; do not judge file corruption from mojibake console output alone.
- When listing Git paths that may contain Chinese, use `git -c core.quotepath=false ...` or configure `core.quotepath=false`, so filenames are not shown as octal escape sequences.
- Production training entry points are official/upstream-backend only: Ultralytics YOLO detection/segmentation/OBB, SMP semantic segmentation, Anomalib PatchCore/EfficientAD, and PaddleOCR Det/Rec official adapters. Do not reintroduce or describe removed diagnostic paths (`tiny_linear_detector`, shipped `python_mock`, small PaddleOCR Rec CTC, C++ segmentation/OCR scaffold training) as product backends.
- YOLO, SMP, anomaly, and OCR have different runtime boundaries: YOLO uses official Ultralytics for training, first ONNX export, and `val()` evaluation while AITrain C++ runtime owns supported packaged inference, benchmark, deployment validation, overlays, and delivery reports; OBB v1 is ONNX Runtime-only for product deployment. SMP semantic segmentation uses the SMP adapter and AITrain ONNX Runtime only; NCNN/TensorRT export is outside SMP scope. Anomaly v1 uses Worker-managed Python/Anomalib artifacts and does not claim AITrain C++ ONNX/TensorRT/NCNN anomaly runtime. OCR acceptance is official-only through PaddleOCR Det/Rec/System reports.
- If a feature is only a scaffold, smoke, diagnostic helper, or report-only workflow, label it clearly. Do not claim customer-domain OCR production readiness, clean Windows acceptance, package-root TensorRT rerun, unsupported-hardware TensorRT success, or new algorithm support without returned evidence.

## Verification

For code changes, run:

```powershell
.\tools\harness-check.ps1
```

For context inspection, run:

```powershell
.\tools\harness-context.ps1
```

For encoding validation, run:

```powershell
.\tools\encoding-check.ps1
```

Final responses should include:

- What changed.
- Key files changed.
- Verification command and result.
- Any known scaffold or unfinished part.
