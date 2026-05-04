# AITrain Studio Acceptance Runbook

This runbook is the Phase 17-47 acceptance path. It freezes the local baseline, validates the packaged layout, prepares TensorRT external acceptance, runs small training smoke checks, covers the current local usability additions, includes the external acceptance handoff package, records a traceable release-freeze package identity, validates newer YOLO detection/segmentation model-family candidates, adds C++ PaddleOCR Det DB-style ONNX postprocess coverage, and prepares real exported PaddleOCR Det ONNX wiring evidence collection before clean Windows and RTX / SM 75+ TensorRT validation.

## Acceptance Modes

Run the unified smoke script from the repository root:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
.\tools\acceptance-smoke.ps1 -PublicDatasets
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke -SkipOfficialOcr
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\acceptance-smoke.ps1 -TensorRT
```

The same script is also installed into packaged builds under `tools\acceptance-smoke.ps1`. From a package root, run:

```powershell
.\tools\acceptance-smoke.ps1 -Package
.\tools\acceptance-smoke.ps1 -TensorRT
```

All generated datasets, official downloads, trainer outputs, and smoke artifacts must stay under `.deps\` or another explicitly supplied work directory. Do not add them to source control.

Every `acceptance-smoke.ps1` run writes `acceptance_summary.json` to its work directory. The summary records requested modes, status, work directory, start/end timestamps, failure reason, and hardware-blocked reason when applicable.

For the current local release-candidate closeout path, use `docs\local-rc-closeout.md`. It is the Phase 42 Lite entry point for source build, package smoke, optional local baseline, optional CPU training smoke, Phase 41 environment profile GUI walkthrough, and boundary wording checks.

For external handoff, use `docs\external-acceptance-handoff.md` and the result templates under `docs\acceptance-templates`. These files define the package-root commands, TensorRT commands, required returned evidence, and the clean separation between local RC closeout and external acceptance.

For release-freeze package identity, use `docs\release-freeze-handoff.md` and `tools\release-freeze-handoff.ps1`. This generates the CPack ZIP, SHA256 hashes, and a handoff manifest without marking external acceptance as passed.

For YOLO model-family productization, use `docs\yolo-model-support-matrix.md` and `tools\phase45-yolo-model-matrix-smoke.ps1`. Phase 45 validates newer Ultralytics detection/segmentation model names only; it does not expand the productized scope to classification, pose, OBB, anomaly, YOLO-World, YOLOE, or TensorRT.

## Phase 17: Local Baseline Freeze

Use this mode before changing release or acceptance documentation:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
```

Expected result:

- `harness-check.ps1` configures, builds, and passes CTest.
- No build outputs, downloaded packages, model weights, datasets, or `.deps` files are staged.
- Scaffold/baseline wording remains explicit in docs and UI text.

For the isolated official PaddleOCR smoke, run:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

This validates official PaddleOCR train/export/inference wiring on a tiny generated dataset. The script checks out a pinned PaddleOCR source ref and records the requested/resolved ref in the report. It does not validate OCR accuracy.

For the full official PaddleOCR Det + Rec + System chain, run:

```powershell
.\tools\phase31-paddleocr-full-official-smoke.ps1
```

This validates official Det train/export, official Rec train/export, and official `predict_system.py` inference with `use_angle_cls=false`. It checks the Det and Rec inference configs, official reports, `official_system_prediction.json`, `system_results.txt`, and visualized output images. It is still a tiny CPU smoke run, so it validates wiring and artifacts rather than OCR quality.

## Phase 18: Package Acceptance

From the source tree, validate the install layout with:

```powershell
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

From a packaged build directory, validate the already-installed layout with:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

Expected result:

- `AITrainStudio.exe` and `aitrain_worker.exe` exist.
- The three built-in plugins load through `aitrain_worker.exe --plugin-smoke`.
- Runtime folders, docs, examples, Python trainers, requirements, and this acceptance script are present.
- Worker self-check emits JSON and reports missing optional runtimes clearly.

## Phase 19: TensorRT External Acceptance

The current GTX 1060 / SM 61 laptop is not a valid TensorRT 10 acceptance machine. On this machine, `-TensorRT` should report `hardware-blocked` instead of pretending the engine path passed.

Run on an RTX / SM 75+ Windows machine or matching cloud GPU:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

Acceptance requires:

- Worker self-check resolves CUDA, cuDNN, TensorRT, TensorRT Plugin, TensorRT ONNX Parser, and ONNX Runtime.
- `aitrain_worker.exe --tensorrt-smoke <work-dir>` builds an engine and runs inference.
- The result is recorded back in `docs\harness\current-status.md` only after a real RTX / SM 75+ pass.

## Phase 43 Lite: External Acceptance Handoff

Before sending a package to an external tester, read:

```text
docs\external-acceptance-handoff.md
docs\acceptance-templates\clean-windows-acceptance-result.md
docs\acceptance-templates\tensorrt-acceptance-result.md
```

Clean Windows package acceptance from the unpacked package root:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

RTX / SM 75+ TensorRT acceptance from a package root or source tree:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

Return the filled template, `acceptance_summary.json`, full console output, Worker self-check JSON, package layout summary, GPU/driver evidence for TensorRT, and the exact TensorRT smoke pass/fail/hardware-blocked output. Do not update TensorRT status as passed until a real supported GPU run succeeds.

## Phase 44 Lite: Release Freeze Handoff

From the source tree, generate a traceable package handoff:

```powershell
.\tools\release-freeze-handoff.ps1
```

This runs the local RC closeout by default, generates the CPack ZIP, computes SHA256 hashes, and writes:

- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

Send those files with the ZIP package and the Phase 43 external acceptance templates. This is still a local handoff preparation step; clean Windows and TensorRT acceptance remain external until evidence is returned.

## Phase 45: YOLO New-Version Productization

Run the required YOLO11 detection/segmentation model matrix:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

Optional YOLO12 candidate coverage:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

Expected artifacts:

- `yolo_model_matrix_summary.json` under `.deps\phase45-yolo-model-matrix` by default.
- For each required model: `ultralytics_training_report.json`, `best.pt`, and exported ONNX.
- If CTest is available, the script runs C++ ONNX Runtime regression checks with `AITRAIN_ACCEPTANCE_SMOKE_ROOT` pointed at the Phase 45 work directory.

Required Phase 45 models are `yolo11n.yaml` and `yolo11n-seg.yaml`. `yolo12n.yaml` and `yolo12n-seg.yaml` are optional candidates until explicitly recorded as passed. This is a wiring/artifact/productization smoke, not an accuracy benchmark.

## Phase 46: PaddleOCR Det C++ ONNX Postprocess

Phase 46 adds C++ DB-style postprocess for PaddleOCR Det ONNX probability maps. The required local acceptance is the normal harness check:

```powershell
.\tools\harness-check.ps1
```

The CTest suite covers the deterministic postprocess path with a synthetic probability map. Expected behavior:

- single connected text region becomes one `ocr_detection` prediction;
- small noise regions are filtered by `minArea`;
- predictions include four-point polygons, normalized boxes, confidence, and pixel area;
- overlay rendering returns a valid image.

This phase validates C++ postprocess wiring only. It does not mark PP-OCRv5 official training/export accuracy as accepted, and the full official PaddleOCR Det+Rec+System path remains `tools\phase31-paddleocr-full-official-smoke.ps1` / official `predict_system.py`.

## Phase 47: Real PaddleOCR Det ONNX Wiring Smoke

Phase 47 attempts to convert the official PaddleOCR Det inference model from Phase 31 into ONNX through the official PaddleX `--paddle2onnx` path, writes an AITrain sidecar with `modelFamily=ocr_detection`, then runs the C++ ONNX Runtime DB-style postprocess through `aitrain_worker --ocr-det-onnx-smoke`.

```powershell
.\tools\phase47-paddleocr-det-onnx-smoke.ps1
```

Expected passing artifacts under `.deps\phase47-paddleocr-det-onnx-smoke`:

- `paddleocr_det_official.onnx`
- `paddleocr_det_official.onnx.aitrain-export.json`
- `cpp_onnx_smoke\ocr_det_onnx_predictions.json`
- `cpp_onnx_smoke\ocr_det_onnx_overlay.png`
- `paddleocr_det_onnx_smoke_summary.json`

The script reuses existing Phase 31 Det artifacts when available. If the official Det inference model is missing or failed, it runs `tools\phase31-paddleocr-full-official-smoke.ps1` first. Conversion uses a separate Python 3.12 embeddable environment because the local Python 3.13 OCR environment only sees old `paddle2onnx` wheels that are not compatible with PaddlePaddle 3.x PIR inference exports. By default the conversion environment uses Paddle's Windows nightly CPU package source, which matches the official Windows guidance for Paddle2ONNX conversion; `-UseStablePaddleForConversion` can be used to reproduce the stable-wheel path.

If conversion is blocked, the script writes `paddleocr_det_onnx_smoke_summary.json` with `status=conversion-blocked`, including Python/Paddle/PaddleX/Paddle2ONNX versions, pip index evidence, the Phase 31 Det inference artifact paths, and the PaddleX conversion output. The current local machine reaches this blocked state even with Python 3.12.10, PaddlePaddle `3.4.0.dev20260407`, PaddleX 3.5.1, and Paddle2ONNX 2.1.0 because `paddle2onnx_cpp2py_export` fails to load. Paddle2ONNX 1.3.1 can import on this machine, but it cannot parse PaddlePaddle 3 PIR `inference.json`, so it is not a valid fallback for current Phase 31 Det exports.

A passing Phase 47 run is real exported Det ONNX wiring evidence for the C++ postprocess path on a tiny CPU smoke model. It is still not PP-OCRv5 official accuracy parity or a production OCR benchmark.

## Production OCR Acceptance

Use this gate only with representative, non-tiny OCR data and returned official reports:

```powershell
.\tools\production-ocr-acceptance.ps1 `
  -DetDataset <paddleocr-det-dataset> `
  -RecDataset <paddleocr-rec-dataset> `
  -SystemImages <end-to-end-image-folder> `
  -OfficialDetReport <paddleocr_official_det_report.json> `
  -OfficialRecReport <paddleocr_official_rec_report.json> `
  -OfficialSystemReport <paddleocr_official_system_report.json> `
  -OcrDetOnnxSummary <paddleocr_det_onnx_smoke_summary.json> `
  -RequireDetOnnxEvidence
```

The script writes:

- `production_ocr_acceptance_report.json`
- `production_ocr_acceptance_summary.md`

Default thresholds are intentionally higher than tiny smoke data: at least 100 Det images, 1000 Rec samples, 100 System images, Rec accuracy >= 0.90, and CER <= 0.10. If evidence is missing, the script exits blocked and records the missing checks instead of marking production OCR as accepted.

## Phase 20: Small Training Smoke

Run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

The script tries to materialize Ultralytics official COCO8 / COCO8-seg datasets through the installed `ultralytics` package. If that fails, it falls back to the generated minimal datasets from `examples\create-minimal-datasets.py`.

To require real public materialization instead of fallback, run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets -SkipOfficialOcr
```

Materialization details are written as JSON reports next to the generated datasets. The reports include the source yaml, download URL, Ultralytics version, output `data.yaml`, and fallback state.

The CTest step receives `AITRAIN_ACCEPTANCE_SMOKE_ROOT`, so segmentation and OCR ONNX Runtime tests use the artifacts generated by this smoke run, including custom `-WorkDir` values.

Expected artifacts:

- YOLO detection: `best.pt`, ONNX export, and `ultralytics_training_report.json`.
- YOLO segmentation: `best.pt`, ONNX export, and `ultralytics_training_report.json` with mask metrics when exposed.
- OCR Rec small CTC: `paddleocr_rec_ctc.pdparams`, `paddleocr_rec_ctc.onnx`, `dict.txt`, and `paddleocr_rec_training_report.json`.
- Official PaddleOCR Rec: `official_model\best_accuracy.pdparams`, `official_inference\inference.yml`, `official_prediction.json`, and `paddleocr_official_rec_report.json` when `phase16-ocr-official-smoke.ps1` is available.
- Official PaddleOCR full chain: Det `official_model\best_accuracy.pdparams`, Det `official_inference\inference.yml`, Rec `official_inference\inference.yml`, `official_system_prediction.json`, `system_results.txt`, visualization images, `paddleocr_official_det_report.json`, `paddleocr_official_rec_report.json`, and `paddleocr_official_system_report.json` when `phase31-paddleocr-full-official-smoke.ps1` is run.

If a public dataset requires interactive registration, record it as an external dataset blocker. Do not block the required smoke path as long as the generated minimal dataset path passes.

## Phase 33: Local CPU Small/Medium Training Smoke

Run:

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke -SkipOfficialOcr
```

This mode does not download public datasets and does not run TensorRT. It generates deterministic `--profile cpu-smoke` data under `<WorkDir>\cpu-training-smoke\generated`, trains YOLO detection and YOLO segmentation for 3 CPU epochs at image size 128, trains the small PaddlePaddle OCR Rec CTC backend for 8 CPU epochs, exports ONNX artifacts, and runs CTest against those artifacts.

Expected artifacts:

- `cpu_training_smoke_summary.json` with dataset counts, parameters, report paths, artifact paths, metrics, and elapsed time.
- YOLO detection and segmentation: `best.pt`, `best.onnx`, and `ultralytics_training_report.json`.
- OCR Rec small CTC: `paddleocr_rec_ctc.pdparams`, `paddleocr_rec_ctc.onnx`, `dict.txt`, and `paddleocr_rec_training_report.json`.

This is a stronger local integration smoke than `-PublicDatasets`, but it is still not a model accuracy benchmark.

## Phase 22-30: Local Usability Baseline

These phases do not change TensorRT acceptance. They make the local RC easier to use and re-check:

- Task history: GUI-started inference, dataset validation, and dataset split create SQLite task records and record Worker artifacts.
- Artifact browsing: task details show artifacts, metrics, and exports; JSON/text/image/ONNX/model-directory artifacts have a preview or a clear unsupported message.
- Dataset management: the GUI auto-detects YOLO detection, YOLO segmentation, PaddleOCR Rec, and PaddleOCR Det layouts, and split supports all four.
- PaddleOCR Det data: the GUI auto-detects `det_gt.txt` / `det_gt_train.txt`, validates PaddleOCR native detection rows, and split writes train/val/test label files plus `split_report.json`.
- Public datasets: COCO8 / COCO8-seg materialization is a standalone machine-readable script with required and fallback modes.
- Official OCR: `phase16-ocr-official-smoke.ps1` covers official Rec train, export, and recognition inference. `phase31-paddleocr-full-official-smoke.ps1` covers official Det + Rec train/export and official System inference.
- Annotation: the dataset page launches X-AnyLabeling as the default external annotation tool, detects its local path, and provides a post-labeling refresh/revalidation action.
- UX closeout: task history can be filtered by category, status, and search text; failed tasks show a short diagnostic next-step summary.

Suggested manual GUI walkthrough:

```powershell
python examples\create-minimal-datasets.py --output .deps\next-smoke
.\build-vscode\bin\Release\AITrainStudio.exe
```

In the GUI, create or open a project, import the generated detection, segmentation, OCR Rec, and OCR Det datasets, launch X-AnyLabeling from the dataset page, use "标注后刷新 / 重新校验", validate and split each dataset, run one training/export/inference path, then confirm the task queue detail view lists report, checkpoint/model, ONNX, overlay, visualized OCR image, and prediction JSON/TXT artifacts.

X-AnyLabeling is detected from `AITRAIN_XANYLABELING_EXE`, the app directory, `tools\x-anylabeling`, `.deps\annotation-tools\X-AnyLabeling`, or `PATH`. Keep downloaded binaries in `.deps\` unless a separate redistribution review is completed.

## Phase 21: Release Closeout

Before marking a release baseline:

```powershell
.\tools\harness-context.ps1
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package
.\tools\phase31-paddleocr-full-official-smoke.ps1
```

Then check:

- `docs\harness\current-status.md` remains the source of truth.
- `docs\yolo-model-support-matrix.md` remains the source of truth for productized YOLO model-family status.
- Phase 7 / Phase 10 TensorRT status stays external pending unless RTX / SM 75+ smoke passed.
- Third-party backend license notes remain visible, especially Ultralytics AGPL / Enterprise constraints.
- C++ tiny detector, segmentation baseline, and OCR baseline are still marked as scaffold/demo/test backends.
- C++ PaddleOCR Det DB-style ONNX postprocess has a Phase 46 v1 path for probability-map outputs. Phase 47 prepares real exported Det ONNX wiring evidence collection, but the current local PaddleX/Paddle2ONNX conversion chain is blocked; full OCR system acceptance remains official `predict_system.py` until production-quality Det+Rec/System accuracy acceptance is recorded.
