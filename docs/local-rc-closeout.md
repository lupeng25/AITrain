# AITrain Studio Local RC Closeout

This checklist is the local, non-external release-candidate closeout path after Phase 39B, Phase 39C, and Phase 41 Lite. It does not add new model backends. It exists to make the current detection, segmentation, OCR, pipeline, benchmark, model registry, delivery report, and environment profile loop repeatable on the development machine.

## Scope

Local closeout covers:

- Source build and CTest through the harness.
- Packaged layout smoke from the source tree.
- Worker self-check, plugin smoke, and package documentation/script presence.
- Optional local baseline acceptance and CPU training smoke.
- Manual GUI walkthrough for the current workbench.
- Documentation language check for scaffold, official backend, and TensorRT hardware-blocked boundaries.

Out of scope:

- Clean Windows machine acceptance.
- RTX / SM 75+ TensorRT engine acceptance.
- New classification, pose, OBB, or anomaly training backends.
- Embedding Python training or annotation tools into the GUI process.

## Command Plan

Fast local RC closeout:

```powershell
.\tools\local-rc-closeout.ps1
```

This runs:

```powershell
git diff --check
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
```

Fuller local closeout, still without external hardware:

```powershell
.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke
```

This additionally runs:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke -SkipOfficialOcr
```

The CPU training smoke is intentionally heavier. It validates integration and artifacts, not production model accuracy.

## Manual GUI Walkthrough

Start with generated sample data:

```powershell
python examples\create-minimal-datasets.py --output .deps\next-smoke
.\build-vscode\bin\AITrainStudio.exe
```

Walk through these screens:

| Area | Check |
|---|---|
| Project | Create or open a project; dashboard should show project, task, dataset, model, plugin, and environment summaries. |
| Environment | Run environment check; YOLO, OCR, and TensorRT profile rows should appear with repair hints. GTX 1060 / SM 61 TensorRT must read as `hardware-blocked` / hardware limited, not passed. |
| Dataset | Import generated YOLO detection, YOLO segmentation, PaddleOCR Rec, and PaddleOCR Det datasets; auto-detection and validation should be visible. |
| Annotation | X-AnyLabeling remains an external tool; launch/detect actions should not block the GUI or imply embedded annotation. |
| Training | Official YOLO / OCR backends should be preferred where applicable; tiny/scaffold or mock backends must remain diagnostic/scaffold choices. |
| Task Artifacts | Select recent tasks and preview JSON/TXT/CSV/image/ONNX/model artifacts; unsupported artifacts should show a clear message. |
| Evaluation / Benchmark | Run evaluation and benchmark from model artifacts when available; reports should be recorded as Worker artifacts. |
| Model Registry | Registered model versions should show lineage, evaluation, benchmark, artifact, and limitation summaries. |
| Delivery Report | Generate a delivery report and confirm HTML, model card, and artifact inventory are present and previewable. |

## Boundary Wording Checklist

Before marking the local RC closeout done, check docs and UI text for:

- Tiny detector, segmentation baseline, OCR baseline, and `python_mock` are scaffold/demo/diagnostic only.
- Ultralytics YOLO official backends require installed official Python packages and license review before redistribution.
- PaddleOCR Rec CTC backend is a small PaddlePaddle CTC trainer, not a full PP-OCRv4 official pipeline.
- PaddleOCR System is official `predict_system.py` tool orchestration, not C++ DB ONNX postprocess.
- TensorRT on GTX 1060 / SM 61 is `hardware-blocked`; RTX / SM 75+ is still required for real TensorRT acceptance.
- Phase 40 classification / pose / OBB / anomaly backends remain deferred until priorities are reset.

## Completion Record

After closeout passes locally, update `docs/harness/current-status.md` with:

- Phase 42 Lite local RC closeout status.
- Commands run and whether CPU training smoke was included.
- Any skipped heavy smoke and why.
- External clean Windows and RTX TensorRT acceptance remaining pending.
