# Current Project Status

Last updated: 2026-05-01

This file is the source of truth for phase status in new AI coding conversations. Read it before using `AITrainStudio_后续实施方案.md`, because that document is the long-range roadmap and may contain historical phase descriptions.

## Phase Status

| Phase | Status | Current interpretation |
|---|---|---|
| Phase 1: Platform stabilization | Done as platform scaffold | Task state transitions, Worker commands, SQLite metadata, artifacts/exports/environment records, and harness checks are in place. |
| Phase 2: Dataset system | Done as initial system | YOLO detection, YOLO segmentation, and PaddleOCR Rec validation exist. Dataset split now covers YOLO detection, YOLO segmentation, and PaddleOCR Rec layouts. |
| Phase 3: YOLO detection training | Scaffold done | Tiny linear detector can train on small data, report loss/mAP50-style metrics, save checkpoint, resume, and generate preview. This is not real LibTorch/CUDA YOLO. |
| Phase 4: ONNX export and inference | Tiny detector path done | Tiny detector ONNX export, ONNX Runtime inference, checkpoint/ONNX consistency, Worker inference, prediction JSON, and overlay output are covered by tests. Full YOLO/OCR postprocess and TensorRT are not done. |
| Phase 5: YOLO segmentation training | Scaffold done | `SegmentationDataset`, `SegmentationDataLoader`, polygon-to-mask, letterbox-aligned masks, multi-polygon/multi-class masks, overlay preview, mask preview artifact, scaffold `maskIoU`/`segmentationMap50`, Worker `taskType=segmentation`, checkpoint, and tests are in place. Real mask head, real mask loss, and CUDA training are not done. |
| Phase 6: OCR recognition training | Scaffold done | `OcrRecDataset`, dictionary loading, label encode/decode, resize/pad batching, `OcrRecTrainer` scaffold, Worker `taskType=ocr_recognition`, `ctcLoss`/`accuracy`/`editDistance`, checkpoint, preview artifact, and tests are in place. Real CRNN/CTC loss and CUDA training are not done. |
| Phase 7: TensorRT and packaging | Code complete; local TensorRT smoke hardware-blocked | Install layout, package smoke, ZIP packaging target, packaged Worker/plugin smoke, runtime DLL discovery, CUDA runtime initialization self-check, downloaded local CUDA/cuDNN/TensorRT redistributables, and SDK-backed TensorRT ONNX->engine export/inference code are in place. After updating the local NVIDIA driver to 582.28, CUDA 13 runtime self-check passes, but TensorRT 10 engine build is still blocked because the local GTX 1060 is SM 61 and this TensorRT release requires a newer target GPU. |
| Phase 8: Python Trainer Adapter and environment management | Done | Worker can launch a Python trainer subprocess, pass a JSON request, read JSON Lines log/progress/metric/artifact/completed/failed messages, propagate failures, and report Python/Ultralytics/PaddleOCR/PaddlePaddle availability in environment checks. The shipped `python_mock` trainer is a scaffold protocol fixture, not real model training. |
| Phase 9: Official YOLO detection training integration | Done on local CPU smoke | Worker routes `trainingBackend=ultralytics_yolo_detect` / `ultralytics_yolo` to `python_trainers/detection/ultralytics_trainer.py`. The adapter normalizes YOLO data yaml, calls official `ultralytics.YOLO(...).train()`, exports ONNX, forwards metrics/artifacts, and has deterministic Worker coverage with a fake official API package. Local `.deps` Python has Ultralytics 8.4.45, Torch 2.11.0 CPU, ONNX 1.21.0, and ONNX Runtime 1.25.1; a 1-epoch CPU smoke with `yolov8n.yaml` produced `best.pt`, `best.onnx`, and `ultralytics_training_report.json`. |
| Phase 10: Real detection ONNX inference and conversion | Done for ONNX Runtime; TensorRT external GPU pending | C++ ONNX Runtime can now run Ultralytics YOLO detection ONNX models with letterbox preprocessing, YOLOv8-style output decode, confidence filtering, NMS, class mapping from the Phase 9 data yaml/report, prediction JSON, and overlay rendering. `exportDetectionCheckpoint(..., format=onnx)` can also copy a real YOLO ONNX and write an AITrain sidecar; TensorRT conversion accepts ONNX sources but remains hardware-blocked on this GTX 1060 / SM 61 machine. |
| Phase 11: Official YOLO segmentation training integration | Done on local CPU smoke | Worker routes `trainingBackend=ultralytics_yolo_segment` to `python_trainers/segmentation/ultralytics_trainer.py`, which reuses the official Ultralytics adapter with segmentation defaults. A 1-epoch CPU smoke with a minimal polygon dataset and `yolov8n-seg.yaml` produced `best.pt`, `best.onnx`, `ultralytics_training_report.json`, and mask metrics (`maskPrecision`, `maskRecall`, `maskMap50`, `maskMap50_95`). C++ ONNX Runtime now decodes YOLOv8-seg boxes, mask coefficients, prototype masks, NMS, prediction JSON, and mask overlay for local smoke artifacts; current C++ segmentation training remains scaffold. |
| Phase 12: PaddlePaddle OCR Rec training integration | Done on local CPU smoke | Worker routes `trainingBackend=paddleocr_rec` to `python_trainers/ocr_rec/paddleocr_trainer.py`. Local `.deps` Python has PaddlePaddle 3.3.1 and PaddleOCR 3.5.0 installed. A minimal PaddleOCR-style Rec dataset trained for 1 epoch on CPU and produced `paddleocr_rec_ctc.pdparams`, `paddleocr_rec_ctc.onnx`, `dict.txt`, `paddleocr_rec_training_report.json`, and real `loss`/`accuracy`/`editDistance` metrics. C++ ONNX Runtime can run the exported CTC model and perform greedy decode. This remains a small PaddlePaddle CTC Rec trainer, not a full PP-OCRv4 official config/export pipeline yet. |
| Phase 13: Productization and acceptance | Local package/docs/examples complete; external acceptance pending | Added Python requirements, training backend docs, hardware compatibility docs, minimal dataset generator, package layout checks for docs/examples/requirements, and README/protocol updates. Generated Phase 13 examples ran direct CPU trainer smoke for YOLO detection, YOLO segmentation, and PaddlePaddle OCR Rec. `harness-check.ps1` and `package-smoke.ps1 -SkipBuild` pass locally. Clean Windows machine validation and RTX / SM 75+ TensorRT acceptance remain external tasks. |
| Phase 14: Official PaddleOCR Rec adapter | Done for adapter/config smoke; official long run environment-isolated | Worker routes `trainingBackend=paddleocr_rec_official` / `paddleocr_ppocrv4_rec` to `python_trainers/ocr_rec/paddleocr_official_adapter.py`. The adapter turns AITrain PaddleOCR-style Rec data into PP-OCRv4 train/val lists, dictionary copy, generated config, report, and reproducible train/export command files. `prepareOnly=true` passed locally. Full official `tools/train.py` is available through `runOfficial=true` and a PaddleOCR source checkout, but should be run in an isolated OCR Python environment; the shared `.deps` Python currently exposes a PaddlePaddle/PyTorch DLL conflict through `albumentations` when official training imports both stacks. |
| Phase 15: Inference result UI summary | Done | GUI inference status now reads existing `inference_predictions.json` and summarizes task type, count, first detection/segmentation class and confidence, segmentation mask area, or OCR text/confidence. Worker inference protocol, overlay artifact paths, and model postprocess code are unchanged. |
| Phase 16: Isolated official PaddleOCR Rec smoke | Done on local CPU smoke | Added `tools/phase16-ocr-official-smoke.ps1`, which creates/uses an isolated OCR Python 3.13 embeddable environment, checks out a pinned PaddleOCR source ref, installs pinned OCR smoke constraints, verifies `tools/train.py`, generates a minimal OCR Rec dataset, runs official PP-OCRv4 Rec training for 1 epoch on CPU through `paddleocr_rec_official`, exports the official inference model with `tools/export_model.py`, runs official `tools/infer/predict_rec.py` on one sample image, and verifies `best_accuracy.pdparams`, `official_inference/inference.yml`, `official_prediction.json`, report metrics, and the requested/resolved PaddleOCR ref. |
| Phase 17: Local baseline freeze | Done locally | Reviewed the dirty worktree as Phase 16 / inference / docs work, checked key diffs, ran `git diff --check`, `tools\harness-check.ps1`, `tools\package-smoke.ps1 -SkipBuild`, and `tools\phase16-ocr-official-smoke.ps1`. Only real local passes are recorded; scaffold/baseline wording remains explicit and TensorRT stays external pending. |
| Phase 18: Acceptance runbook and unified smoke script | Done locally | Added `docs\acceptance-runbook.md` and `tools\acceptance-smoke.ps1`, installed them into the package layout, and extended package smoke to assert their presence. Source and packaged `-Package` modes pass, and failure paths distinguish missing layout, missing Python modules, missing GPU, and hardware-blocked TensorRT. |
| Phase 19: TensorRT external acceptance preparation | Prepared; local hardware-blocked | `tools\acceptance-smoke.ps1 -TensorRT` runs Worker self-check, detects GPU compute capability, and stops with `hardware-blocked` on the local GTX 1060 / SM 61. Phase 7 / Phase 10 TensorRT acceptance must remain pending until a real RTX / SM 75+ `--tensorrt-smoke` pass is recorded. |
| Phase 20: Small training / inference / conversion acceptance | Done on local smoke; public COCO8 materialization passed locally | `tools\acceptance-smoke.ps1 -PublicDatasets` generates minimal datasets, materializes official Ultralytics COCO8 / COCO8-seg when available, and falls back to generated datasets only when public materialization fails. The latest local smoke materialized COCO8 and COCO8-seg, trained YOLO detection and segmentation for 1 epoch through the official Ultralytics adapters, exported ONNX artifacts and reports, trained/exported the PaddlePaddle OCR CTC backend, and ran C++ ONNX Runtime tests against the current acceptance WorkDir via `AITRAIN_ACCEPTANCE_SMOKE_ROOT`. The isolated official PaddleOCR train/export/inference smoke also passes. |
| Phase 21: Release closeout | Done locally; external machine checks pending | README, training backend docs, hardware compatibility notes, acceptance runbook, current status, and the long-range implementation plan now describe the Phase 17-21 delivery baseline. No Worker JSON protocol, SQLite schema, plugin interface, or GUI architecture changes were made for this phase set. Clean Windows packaged validation and RTX / SM 75+ TensorRT remain external acceptance items. |
| Phase 22: Task history and artifact index | Done locally | GUI-triggered inference, dataset validation, and dataset split now create SQLite task records, pass task ids into Worker requests, and record Worker artifacts. `ProjectRepository` now exposes read-only task artifact/metric/export queries and dataset version history queries. `tools\acceptance-smoke.ps1` writes `acceptance_summary.json` for each run, including modes, status, work directory, timing, failure reason, and hardware-blocked reason. |
| Phase 23: Unified GUI artifact browser | Done locally | The task queue page now has a task detail area showing artifacts, metrics, and exports for the selected task. JSON/YAML/TXT previews are read-only and size-limited, image overlays render inline, ONNX artifacts show model-family inference, directory/model artifacts show metadata, and actions can open the directory, copy a path, or reuse an artifact as inference/export input. |
| Phase 24: Dataset management enhancements | Done locally | The dataset page now includes a registered dataset list, format auto-detection for YOLO detection, YOLO segmentation, and PaddleOCR Rec, and Worker-backed split support for all three formats. Split reports are recorded as artifacts and dataset versions remain in SQLite. |
| Phase 25: Public dataset materialization stabilization | Done locally | Added `tools\materialize-ultralytics-dataset.py` and wired `acceptance-smoke.ps1 -PublicDatasets` to materialize COCO8 / COCO8-seg from the installed Ultralytics package yaml and download URL. Reports record source yaml, download URL, Ultralytics version, output `data.yaml`, and fallback state; `-RequirePublicDatasets` fails clearly instead of falling back. |
| Phase 26: Official PaddleOCR Rec chain enhancement | Done locally | The official PaddleOCR adapter supports explicit train/val label files, dictionary file, official config, pretrained/resume checkpoints, export-only mode, post-export official inference, inference image, and `recImageShape`. Reports now record Python/Paddle/PaddleOCR versions, PaddleOCR source refs, commands, configs, labels, dictionary, checkpoint/model paths, metrics, exit codes, log paths, and `official_prediction.json` when inference is requested. |
| Phase 27: Mature local workbench UI | Done locally; manual GUI walkthrough still recommended | The Qt Widgets shell has been reorganized into grouped workbench navigation, a project/data/task/model/environment dashboard, dataset library/details workflow, official-backend-first training launch controls, task-history-centered artifact browsing, and less demo-like export/inference copy. This is UI information architecture only; Worker JSON, SQLite schema, plugin interfaces, and training implementation remain unchanged. |

## Local Hardware Note

Recorded on 2026-04-30:

- Machine: Lenovo Legion Y7000P-1060 / model 81LE.
- CPU: Intel Core i5-8300H, 4 cores / 8 threads.
- Memory: 16 GB DDR4-2666 installed as 2x8 GB; firmware reports 2 slots and 32 GB maximum.
- Storage: 512 GB NVMe SSD.
- GPU: NVIDIA GeForce GTX 1060 6 GB. Driver was upgraded from 398.27 to 582.28 on 2026-04-30.
- Phase 7 impact: this machine should not be treated as the final TensorRT acceptance machine. The driver update makes CUDA 13 runtime initialization pass, but TensorRT 10 smoke fails with `Target GPU SM 61 is not supported by this TensorRT release`.
- Practical upgrade path: optionally upgrade memory to 32 GB and storage to 1-2 TB. Do not plan around replacing the laptop GPU. Use a CUDA 13/TensorRT-compatible RTX machine or cloud GPU for real Phase 7 TensorRT verification.
- Python note: Python 3.13.13 embeddable x64 was downloaded under `.deps/python-3.13.13-embed-amd64` for Phase 8 adapter verification. `.deps/` is ignored and should not be committed.

## Current Next Task

Current local follow-up: perform a manual GUI walkthrough for Phase 27, use the GUI for the daily workflow, and keep external TensorRT blocked until an RTX / SM 75+ machine is available:

- keep Phase 7 marked as code complete but hardware-blocked on this GTX 1060 / SM 61 machine
- Phase 8 is complete as an adapter/protocol layer; `python_mock` remains a scaffold protocol fixture
- Phase 9 is complete on this machine for a CPU smoke using the installed official Ultralytics package; keep license constraints visible before redistribution
- Phase 10 ONNX Runtime inference is complete for real YOLO detection ONNX; TensorRT engine acceptance still needs RTX / SM 75+
- Phase 11 official YOLO segmentation training is complete on local CPU smoke
- Phase 12 PaddlePaddle OCR Rec training and C++ ONNX CTC greedy decode are complete on local CPU smoke; full PP-OCRv4 config/export is still future work
- Phase 13 local productization is complete: docs, generated sample datasets, Python dependency files, hardware compatibility notes, package smoke verification
- Phase 14 official PaddleOCR Rec adapter/config generation is complete; full official training should use an isolated OCR Python environment or a clean machine to avoid PaddlePaddle/PyTorch DLL conflicts in the shared `.deps` Python
- Phase 15 GUI inference summaries are complete for detection, segmentation, and OCR recognition prediction JSON
- Phase 16 official PaddleOCR Rec train/export smoke is complete in an isolated OCR Python environment with a pinned PaddleOCR source ref
- Phase 17 local baseline freeze is complete; the checked worktree remains a local delivery baseline with TensorRT still external pending
- Phase 18 acceptance runbook and `tools\acceptance-smoke.ps1` are complete and installed into the package layout
- Phase 19 TensorRT preparation is complete; local `-TensorRT` correctly reports `hardware-blocked` on SM 61
- Phase 20 training / inference / conversion acceptance is complete locally; CTest consumes the current acceptance WorkDir, and public Ultralytics COCO8 / COCO8-seg materialization now succeeds when network and package metadata are available
- Phase 21 release closeout documentation is complete locally; clean Windows and RTX TensorRT remain external acceptance tasks
- Phase 22 task history and artifact indexing is complete locally; repository query tests cover artifacts, metrics, exports, and dataset versions
- Phase 23 GUI artifact browsing is implemented; it still needs normal manual GUI walkthrough whenever UI layout changes are made
- Phase 24 dataset management now supports detection, segmentation, and OCR Rec import/validation/split flows
- Phase 25 public dataset materialization is split into a standalone script and has required/fallback modes
- Phase 26 official PaddleOCR Rec smoke now covers train, export, and official inference on one generated sample
- Phase 27 local automated checks pass; run a manual GUI walkthrough after future layout changes because screenshot/interaction QA is not covered by `harness-check.ps1`
- keep training logic inside core/plugin/Worker boundaries, not in `MainWindow`
- keep C++ tiny detector as a scaffold/demo/test backend until the Python path is stable
- preserve the external TensorRT acceptance checklist for a future RTX / SM 75+ machine

For local baseline acceptance, run:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
```

For packaging smoke checks, run:

```powershell
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

For small public/generated-data training smoke, run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

To require successful public COCO8 / COCO8-seg materialization instead of fallback, run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets -SkipOfficialOcr
```

For official PaddleOCR Rec train/export/inference smoke in an isolated OCR environment, run:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

For ZIP package generation, run:

```powershell
cmake --build build-vscode --target package
```

Minimal example datasets can be generated with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

When an RTX / SM 75+ machine is available, resume Phase 7 acceptance with:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

## Non-Negotiable Notes

- Do not describe the C++ tiny detector, segmentation baseline, or OCR baseline as real YOLO/OCR training.
- The `ultralytics_yolo_detect` backend is the first official YOLO detection training integration, but it requires an installed official Python package and license review before redistribution.
- Current detection and segmentation training are scaffold/baseline workflows.
- Real training should now be implemented through Worker-managed Python trainer subprocesses; do not embed Python inside the GUI process.
- Phase 5 is complete only as a segmentation scaffold/baseline loop; it is not real YOLO segmentation training.
- Phase 6 is complete only as an OCR recognition scaffold/baseline loop; it is not real CRNN/CTC OCR training.
- Phase 11 adds official YOLO segmentation training/export and C++ ONNX Runtime mask postprocess for YOLOv8-seg smoke models.
- Phase 12 adds a real PaddlePaddle CTC OCR Rec trainer and C++ ONNX CTC greedy decode for PaddleOCR-style data, but it is not a full PaddleOCR PP-OCRv4 official training/export pipeline yet.
- Phase 13 local productization is complete, but clean-machine and RTX TensorRT acceptance still require external hardware/environment.
- Phase 14 adds the official PaddleOCR PP-OCRv4 Rec adapter. Prepare-only artifacts are not trained model artifacts.
- Phase 15 is UI-only and must continue to read Worker artifacts instead of duplicating inference logic in `MainWindow`.
- Phase 16 validates official PaddleOCR train/export/inference smoke only on a tiny dataset; it proves wiring and artifacts, not OCR accuracy.
- Phase 17-21 add acceptance scripts, docs, and smoke coverage only; they do not change public Worker JSON protocol, SQLite schema, plugin interfaces, or GUI architecture.
- Phase 20 generated-data smoke proves integration and artifacts, not useful model accuracy.
- Phase 22-26 improve local usability and repeatability; they do not change the TensorRT acceptance requirement.
- Phase 27 is UI-only: it may reorganize pages, labels, and helper widgets, but must not move training/export/inference work into `MainWindow`.
- Keep long-running execution in `aitrain_worker`.
- Keep model-specific behavior behind core/plugin/Worker boundaries, not in `MainWindow`.
