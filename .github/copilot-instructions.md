# Copilot Instructions for AITrain Studio

This project uses a harness-first workflow. Before generating substantial code, use these files as the source of truth:

- `HARNESS.md`
- `docs/harness/current-status.md`
- `docs/harness/project-context.md`
- `docs/harness/implementation-checklist.md`
- `docs/harness/quality-gates.md`

For UI work, follow:

- `docs/harness/ui-guidelines.md`

## Project Rules

- C++20, CMake, Qt Widgets.
- Keep Qt 5.12+ compatibility; the current verified environment is Qt 5.12.9.
- Long-running tasks belong in `aitrain_worker`, not the GUI thread.
- GUI should orchestrate and display state only.
- SQLite access should go through `ProjectStore`; GUI reads through Query Service/Presenter and writes through `ProjectWorkspace`/Worker.
- Model, dataset, validation, export, and inference behavior should use the compile-time `CapabilityRegistry` and explicit Worker/core adapters, not dynamic plugins.
- Preserve the left-sidebar workbench UI.
- Use UTF-8 and `QStringLiteral` for Chinese UI text.
- When reading project text in Windows PowerShell, specify UTF-8 explicitly, for example `Get-Content -Encoding UTF8`; mojibake in terminal output is not proof that the file is corrupt.
- When listing Git paths with possible Chinese filenames, use `git -c core.quotepath=false ...` or set `core.quotepath=false`.
- Production training is official/upstream-backend only: Ultralytics YOLO detection/segmentation/OBB, SMP semantic segmentation, Anomalib PatchCore/EfficientAD, and PaddleOCR Det/Rec official adapters.
- Do not reintroduce or describe removed diagnostic paths (`tiny_linear_detector`, shipped `python_mock`, small PaddleOCR Rec CTC, C++ segmentation/OCR scaffold training) as product backends.
- Keep runtime boundaries explicit: YOLO uses official Ultralytics for training/export/`val()` evaluation plus AITrain C++ runtime for supported packaged inference/benchmark/deployment validation; OBB v1 is ONNX Runtime-only for product deployment. SMP uses ONNX Runtime only for product inference/deployment validation. Anomaly v1 uses Worker-managed Python/Anomalib artifacts, not AITrain C++ ONNX/TensorRT/NCNN runtime. OCR acceptance uses PaddleOCR Det/Rec/System official reports.
- Do not claim customer-domain OCR readiness, clean Windows acceptance, package-root TensorRT reruns, unsupported-hardware TensorRT success, or new algorithm support without returned evidence.

## Verification

Use:

```powershell
.\tools\harness-check.ps1
```

For encoding-only checks, use:

```powershell
.\tools\encoding-check.ps1
```

Expected successful result:

```text
Harness check passed.
```
