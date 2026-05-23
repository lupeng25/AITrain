# AITrain Studio Training Backends

AITrain Studio keeps training out of the GUI process. Real training is launched by `aitrain_worker` as a Python subprocess and reports newline-delimited JSON events back to the existing Worker protocol.

## Backend Summary

Production training is official-backend only. The GUI training page and Worker production whitelist expose these training backends:

| Backend | Task | Status | Notes |
|---|---|---|---|
| `ultralytics_yolo_detect` | Detection | Official Ultralytics adapter | Uses Ultralytics YOLO detection training and ONNX export. Review AGPL-3.0 / Enterprise license before redistribution. |
| `ultralytics_yolo_segment` | Segmentation | Official Ultralytics adapter | Uses Ultralytics YOLO segmentation training and ONNX export. C++ ONNX Runtime can decode boxes, mask coefficients, prototypes, and render mask overlays. |
| `paddleocr_det_official` | OCR detection | Official PaddleOCR adapter | Generates a PP-OCRv4 detection config from PaddleOCR Det data and can run official PaddleOCR `tools/train.py` and `tools/export_model.py`. Artifacts include `aitrain_ppocrv4_det.yml`, `official_model/best_accuracy.pdparams`, `official_inference/inference.yml`, and `paddleocr_official_det_report.json`. |
| `paddleocr_rec_official` / `paddleocr_ppocrv4_rec` | OCR recognition | Official PaddleOCR adapter | Generates a PP-OCRv4 recognition config from AITrain PaddleOCR-style Rec data and runs official PaddleOCR `tools/train.py`, `tools/export_model.py`, and optional `tools/infer/predict_rec.py` when `runOfficial=true` and `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` points to a checkout. Production GUI requests set `runOfficial=true` and `prepareOnly=false`. |

`paddleocr_system_official` remains the official OCR System inference/validation adapter. It is not shown as a "train model" backend because it runs official `predict_system.py` against exported Det and Rec inference model directories.

Legacy diagnostic training implementations have been physically removed from production packages and training routing. `paddleocr_rec` remains a dataset format, not a training backend. Protocol tests that need a Python trainer now create an explicit temporary fixture through `pythonTrainerScript` and require `AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS=1`; no shipped `python_mock` trainer is provided.

## Environment Setup

Use an isolated Python environment. The local development machine used Python 3.13 embeddable under `.deps`, but a regular venv is preferred for users.

Detection and segmentation:

```powershell
python -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r python_trainers\requirements-yolo.txt
```

YOLO model-family status is tracked in `docs\yolo-model-support-matrix.md`. The product defaults remain `yolov8n.yaml` for detection and `yolov8n-seg.yaml` for segmentation. Phase 45 adds a repeatable acceptance matrix for newer detection/segmentation families:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

The required Phase 45 targets are `yolo11n.yaml`, `yolo11n-seg.yaml`, `yolo12n.yaml`, and `yolo12n-seg.yaml`. Classification, pose, OBB, anomaly, YOLO-World, and YOLOE remain out of the current productized training path.

OCR recognition:

```powershell
python -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install -r python_trainers\requirements-ocr.txt
```

Optional official PaddleOCR Det/Rec training and System inference require a PaddleOCR source checkout because the installed `paddleocr` package exposes inference pipelines, not the legacy `tools/train.py` training scripts:

```powershell
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git .deps\PaddleOCR
$env:AITRAIN_PADDLEOCR_REPO = (Resolve-Path .deps\PaddleOCR).Path
```

The reproducible local smoke command is:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

That script uses an isolated OCR Python embeddable environment under `.deps\python-3.13.13-ocr-amd64`, checks out a pinned PaddleOCR source ref, installs pinned OCR smoke constraints unless disabled, runs official PP-OCRv4 Rec training for 1 epoch on CPU, exports the official inference model, runs official recognition inference on one generated sample image, and checks the checkpoint, inference config, prediction report, resolved source ref, and metrics report.

The full official PaddleOCR Det + Rec + System smoke is:

```powershell
.\tools\phase31-paddleocr-full-official-smoke.ps1
```

That script reuses the isolated OCR environment and pinned PaddleOCR checkout, generates minimal PaddleOCR Det and Rec datasets, runs 1-epoch CPU official Det and Rec train/export, then calls official `predict_system.py` with `use_angle_cls=false`. It validates reports, exported `official_inference/inference.yml` files, `official_system_prediction.json`, and visualized system output images. The run proves toolchain wiring and task artifacts, not useful OCR accuracy.

For offline deployment, build a wheelhouse on a connected machine:

```powershell
python -m pip download -r python_trainers\requirements-yolo.txt -d wheelhouse-yolo
python -m pip download -r python_trainers\requirements-ocr.txt -d wheelhouse-ocr
```

Then install offline:

```powershell
python -m pip install --no-index --find-links wheelhouse-yolo -r python_trainers\requirements-yolo.txt
python -m pip install --no-index --find-links wheelhouse-ocr -r python_trainers\requirements-ocr.txt
```

## Dataset Inputs

YOLO detection and segmentation use standard YOLO folder layout with `images/train`, `images/val`, `labels/train`, `labels/val`, and a `data.yaml`.

OCR recognition uses a PaddleOCR-style Rec layout:

```text
dataset/
  dict.txt
  rec_gt.txt
  images/
    sample.png
```

`rec_gt.txt` contains one image path and label per line:

```text
images/sample.png<TAB>label
```

OCR detection uses a PaddleOCR-style Det layout:

```text
dataset/
  det_gt.txt
  images/
    sample.png
```

`det_gt.txt` contains one image path and one JSON array per line:

```text
images/sample.png<TAB>[{"transcription":"text","points":[[1,1],[30,1],[30,20],[1,20]]}]
```

`det_gt_train.txt` and `det_gt_val.txt` are also accepted. Validation checks that referenced images exist, the JSON parses as an array, each box has `transcription`, each box has at least four non-negative points, and duplicate image rows are rejected. `###` and `*` are preserved as PaddleOCR ignore transcriptions.

Dataset validation and split are Worker-backed flows. The GUI can now auto-detect YOLO detection, YOLO segmentation, PaddleOCR Rec, and PaddleOCR Det layouts; split outputs are recorded as SQLite dataset versions and task artifacts. PaddleOCR Rec split writes `rec_gt_train.txt`, `rec_gt_val.txt`, `rec_gt_test.txt`, and keeps a compatible `rec_gt.txt`. PaddleOCR Det split writes `det_gt_train.txt`, `det_gt_val.txt`, `det_gt_test.txt`, keeps a compatible `det_gt.txt`, copies images into split folders, and writes `split_report.json`.

## Model Export Formats

The GUI model export page routes conversion through `aitrain_worker`; conversion logic must not run in `MainWindow`.

- `onnx`: copies or creates an ONNX model and writes an AITrain sidecar report.
- `ncnn`: converts an official ONNX source into NCNN `.param` and `.bin` files through the external `onnx2ncnn` tool. Configure it with `AITRAIN_NCNN_ONNX2NCNN` or an NCNN install root in `AITRAIN_NCNN_ROOT`.
- `tensorrt`: RTX 4090 D acceptance has passed for the current validation lane; unsupported GPUs such as GTX 1060 / SM 61 still report `hardware-blocked`.

NCNN export creates `.param/.bin` deployment artifacts and writes an AITrain sidecar. Deployment validation can run NCNN CPU inference for supported YOLO detection and segmentation models when the build is configured with an NCNN SDK/runtime and a sample image is supplied. External NCNN models are not guessed blindly: provide the sidecar or explicit `modelFamily`, `classNames`, `inputBlob`, `outputBlobs`, and `decoder` settings.

NCNN runtime smoke:

```powershell
.\tools\phase-ncnn-runtime-smoke.ps1 -NcnnRoot <ncnn-sdk-root> -OnnxPath <best.onnx> -SampleImagePath <sample.png> -OutputDir <smoke-output> -TaskType detection
```

For existing external NCNN `.param/.bin` artifacts, provide an AITrain sidecar or explicit blob/decoder settings, then use the Worker helper without forcing an ONNX conversion:

```powershell
.\build-vscode\bin\aitrain_worker.exe --ncnn-param-smoke <model.param> --image <sample.png> --output <smoke-output> --task-type segmentation
```

Current local NCNN evidence on 2026-05-16:

- Detection passed with Hyuto YOLOv8 ONNX converted through `onnx2ncnn`; output is under `.deps\github-ncnn-smoke\hyuto-yolov8\runtime-output` and reported `predictionCount=14`.
- Segmentation passed with nihui `ncnn-android-yolov8` preconverted `yolov8n_seg.ncnn.param/.bin` plus an explicit AITrain sidecar using `decoder=dfl`; output is under `.deps\github-ncnn-smoke\nihui-yolov8n-seg-ncnn\runtime-output\deployment-validation` and reported `predictionCount=100`.
- Hyuto and X-AnyLabeling YOLOv8-seg ONNX conversion attempts currently leave unsupported NCNN `Shape` layers. AITrain records these as failed deployment validation reports instead of crashing Worker; they are not passing segmentation ONNX-conversion evidence.

## Acceptance Smoke

The unified Phase 17-21 acceptance entry point is:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

This mode checks required Python modules, generates tiny local datasets under `.deps\acceptance-smoke` by default, tries to materialize Ultralytics COCO8 / COCO8-seg through the installed official package, then runs official-adapter smoke training for YOLO detection, YOLO segmentation, and PaddleOCR Rec. During the CTest step, it sets `AITRAIN_ACCEPTANCE_SMOKE_ROOT` so inference tests consume artifacts generated in the current WorkDir rather than relying on older local smoke outputs.

Public dataset materialization is handled by `tools\materialize-ultralytics-dataset.py`. It reads the installed Ultralytics dataset yaml, resolves the official download URL, downloads into `.deps\datasets\downloads`, extracts into `.deps\datasets\materialized`, rewrites a local absolute-path `data.yaml`, and writes a machine-readable materialization report. Use `-RequirePublicDatasets` to fail if COCO8 / COCO8-seg cannot be materialized:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets
```

Every `acceptance-smoke.ps1` run writes `acceptance_summary.json` into its work directory with modes, status, timing, failure reason, and hardware-blocked reason when applicable.

For a longer local-only CPU exercise that avoids public downloads and TensorRT, run:

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
```

This mode generates deterministic small/medium datasets with `examples\create-minimal-datasets.py --profile cpu-smoke`, trains YOLO detection and segmentation for 3 epochs at image size 128 on CPU, runs official PaddleOCR Rec train/export/inference through `phase16-ocr-official-smoke.ps1`, runs CTest with `AITRAIN_ACCEPTANCE_SMOKE_ROOT` pointed at the new artifacts, and writes `cpu_training_smoke_summary.json`. If the official OCR source checkout or isolated OCR environment is unavailable, the mode must fail or block with an explicit environment error instead of falling back to diagnostic CTC training. It validates wiring and artifacts; it is not an accuracy benchmark.

For the Phase 45 newer-YOLO-family matrix, run:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

This validates the required YOLO11 and YOLO12 detection/segmentation candidates through the same official Ultralytics adapters, checks report/checkpoint/ONNX artifacts, and runs CTest against the generated work directory when a build tree is available. The legacy `-IncludeYolo12` switch is still accepted, but YOLO12 is now included by default.

## Official PaddleOCR Adapter Parameters

The official Rec adapter accepts these extra parameters in addition to the common Python trainer request fields:

- `trainLabelFile`, `valLabelFile`, and `dictionaryFile` to use explicit PaddleOCR Rec materials.
- `officialConfig` to start from a specific PaddleOCR recognition config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `runInferenceAfterExport=true` plus `inferenceImage` to run official `predict_rec.py` after export and write `official_prediction.json`.
- `recImageShape` to override the generated recognition image shape, for example `3,48,320`.

The final `paddleocr_official_rec_report.json` records PaddleOCR requested/resolved refs, Python/Paddle/PaddleOCR versions, train/export/predict commands, config and label paths, dictionary path, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths.

The official Det adapter accepts these extra parameters:

- `trainLabelFile` and `valLabelFile` to use explicit PaddleOCR Det label files.
- `officialConfig` to start from a specific PaddleOCR detection config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `imageSize` to override generated detection image size.

The final `paddleocr_official_det_report.json` records PaddleOCR requested/resolved refs, Python/Paddle/PaddleOCR versions, train/export commands, config and label paths, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths.

The official System adapter accepts these parameters:

- `detModelDir`: exported PaddleOCR Det inference model directory.
- `recModelDir`: exported PaddleOCR Rec inference model directory.
- `dictionaryFile`: recognition dictionary file.
- `inferenceImage`: image or directory to pass to official `predict_system.py`.
- `dropScore`: optional recognition score threshold.
- `useGpu`: default `false`.

The final `paddleocr_official_system_report.json` records Python/Paddle/PaddleOCR versions, source checkout ref, command, exit code, log path, model directories, dictionary path, `official_system_prediction.json`, `system_results.txt`, and the visualization directory.

## C++ PaddleOCR Det ONNX Postprocess

Phase 46 adds a C++ ONNX Runtime path for PaddleOCR Det DB-style detection maps. AITrain can identify `ocr_detection` ONNX sidecars/reports, run a single-output DB probability map shaped `[1,1,H,W]`, `[1,H,W]`, or `[H,W]`, threshold connected components, emit four-point text-box polygons, write `ocr_detection` prediction JSON, and render an overlay through the existing inference artifact path.

Phase 47 adds `tools\phase47-paddleocr-det-onnx-smoke.ps1`, which attempts to convert the official PaddleOCR Det inference model into ONNX through PaddleX `--paddle2onnx`, writes an AITrain sidecar when conversion succeeds, and calls `aitrain_worker --ocr-det-onnx-smoke` to produce C++ predictions and overlay artifacts.

RTX 4090 validation unblocked Phase 47 by exporting a Paddle 2.6 old-IR PaddleOCR Det inference model, converting it with Paddle2ONNX, writing the AITrain sidecar, and running `aitrain_worker --ocr-det-onnx-smoke`. The passing evidence is under `.deps/rtx4090-validation/phase47-paddleocr-det-onnx`, including `paddleocr_det_onnx_smoke_summary.json`, C++ predictions, and overlay PNG.

This is a v1 DB-style postprocess plus prepared real exported ONNX wiring smoke path. It is not a replacement for the full official PaddleOCR System acceptance path, and it does not claim PP-OCRv5 accuracy parity. Use official `predict_system.py` for complete Det+Rec system validation until production-quality OCR accuracy acceptance is separately recorded.

If public dataset materialization fails or requires external interaction, the generated minimal datasets remain the required smoke baseline. The failure reason should be recorded as an external data acquisition blocker, not hidden as a successful public dataset run.

## Known Boundaries

- TensorRT engine building has passing RTX 4090 D acceptance evidence under `.deps/rtx4090-validation/acceptance-tensorrt`; older unsupported GPUs should still report `hardware-blocked`.
- Phase 45 covers newer YOLO detection/segmentation model names only; it does not productize YOLO classification, pose, OBB, anomaly, YOLO-World, or YOLOE.
- Historical GTX 1060 / SM 61 machines can run CPU training smoke and ONNX Runtime checks, but they cannot validate TensorRT 10 engine building and must not override RTX 4090 acceptance evidence.
- C++ segmentation mask ONNX postprocess and OCR ONNX CTC greedy decode are available for the current smoke models.
- C++ OCR Det ONNX Runtime DB-style postprocess is available as a Phase 46 v1 path for single-output probability maps, and Phase 47 now has real exported Det ONNX wiring smoke evidence. End-to-end PaddleOCR validation still uses official `predict_system.py`; per the current RTX 4090 validation decision, Rec accuracy is not considered for this pass and remains a future production-quality gate if reinstated.
- Official third-party backend licensing must be reviewed before commercial redistribution.
- The official PaddleOCR adapter should be run in an isolated OCR Python environment. Mixing PaddlePaddle and PyTorch in one Windows Python process can trigger DLL conflicts through newer `albumentations` builds.
- The official PP-OCRv4 smoke uses a tiny generated dataset; it validates train/export/inference wiring and artifacts, not useful OCR accuracy.
