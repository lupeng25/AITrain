# AITrain Studio Training Backends

AITrain Studio keeps training out of the GUI process. Real training is launched by `aitrain_worker` as a Python subprocess and reports newline-delimited JSON events back to the existing Worker protocol.

## Backend Summary

Production training is official-backend only. The GUI training page and Worker production whitelist expose these training backends:

| Backend | Task | Status | Notes |
|---|---|---|---|
| `ultralytics_yolo_detect` | Detection | Official Ultralytics adapter | Uses Ultralytics YOLO detection training and official export. P1 exposes YOLOv8 / YOLO11 / YOLO12 `n/s/m/l/x` `.yaml` and `.pt` presets plus YOLOv8 P2/P6 detection YAML architectures. Product evaluation uses Ultralytics official `val()`; product inference, benchmark, and deployment validation run against official artifacts through the AITrain C++ runtime. Review AGPL-3.0 / Enterprise license before redistribution. |
| `ultralytics_yolo_segment` | Segmentation | Official Ultralytics adapter | Uses Ultralytics YOLO instance-segmentation training and official export. P1 exposes YOLOv8 / YOLO11 / YOLO12 `n/s/m/l/x` `-seg.yaml` and `-seg.pt` presets. Product evaluation uses Ultralytics official `val()`; product inference, benchmark, and deployment validation run against official artifacts through the AITrain C++ runtime, including mask postprocess and overlays. |
| `paddleocr_det_official` | OCR detection | Official PaddleOCR adapter | Generates PP-OCRv4 or PP-OCRv5 detection configs from PaddleOCR Det data and can run official PaddleOCR `tools/train.py` and `tools/export_model.py`. The default preset is `PP-OCRv5_mobile_det`; `PP-OCRv4_mobile_det` and `PP-OCRv5_server_det` remain selectable. |
| `paddleocr_rec_official` / `paddleocr_ppocrv4_rec` | OCR recognition | Official PaddleOCR adapter | Generates PP-OCRv4 or PP-OCRv5 recognition configs from AITrain PaddleOCR-style Rec data and runs official PaddleOCR `tools/train.py`, `tools/export_model.py`, and optional `tools/infer/predict_rec.py` when `runOfficial=true` and `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` points to a checkout. The default preset is `PP-OCRv5_mobile_rec`; `PP-OCRv4_mobile_rec`, `PP-OCRv5_server_rec`, and `en_PP-OCRv5_mobile_rec` remain selectable. |

`paddleocr_system_official` remains the official OCR System inference/validation adapter. It is not shown as a "train model" backend because it runs official `predict_system.py` against exported Det and Rec inference model directories.

Legacy diagnostic training implementations have been physically removed from production packages and training routing. `paddleocr_rec` remains a dataset format, not a training backend. Protocol tests that need a Python trainer now create an explicit temporary fixture through `pythonTrainerScript` and require `AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS=1`; no shipped `python_mock` trainer is provided.

## YOLO Runtime Boundary

YOLO detection and segmentation are not end-to-end official-only routes. AITrain uses official Ultralytics code for training, ONNX export, and detection/segmentation evaluation via `YOLO(...).val()`, then treats the exported ONNX / TensorRT / NCNN artifacts as deployment inputs for its local C++ runtime. C++ ONNX Runtime, TensorRT, and NCNN paths handle prediction JSON, overlays, benchmark summaries, and deployment validation so the packaged Windows product can run without embedding Python in the GUI process.

AITrain no longer computes local YOLO AP/mAP, mask IoU, TP/FP/FN, local evaluation error samples, or local evaluation overlays. The `evaluation_report.json` wrapper remains for the GUI and task history, with `evaluationSource=ultralytics_official_val` and links to official plots/predictions when Ultralytics writes them. OCR is different: current OCR product inference, evaluation, benchmark, deployment validation, and acceptance are official-only through PaddleOCR Det/Rec/System reports.

## Environment Setup

Use an isolated Python environment. The local development machine used Python 3.13 embeddable under `.deps`, but a regular venv is preferred for users.

Detection and segmentation:

```powershell
python -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r python_trainers\requirements-yolo.txt
```

YOLO model-family status is tracked in `docs\yolo-model-support-matrix.md`. The product defaults remain `yolov8n.yaml` for detection and `yolov8n-seg.yaml` for segmentation. P1 adds editable GUI presets for YOLOv8 / YOLO11 / YOLO12 detection and instance segmentation across `n/s/m/l/x`, both `.yaml` and `.pt`, plus YOLOv8 P2/P6 detection YAML architectures. `.pt` weights are not bundled; the installed Ultralytics package may download them into the user environment.

Run the full P1 productization matrix with:

```powershell
.\tools\phase-p1-yolo-full-matrix-smoke.ps1
```

The previous Phase 45 matrix remains useful as a faster YOLO11/YOLO12 nano wiring check:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

The P1 matrix does not productize YOLO26, semantic segmentation, tracking, classification, pose, OBB, anomaly, YOLO-World, or YOLOE.

Official YOLO export parameters are recorded as `ultralyticsExportArgs` and are accepted by both training and model export flows:

```json
{
  "format": "onnx",
  "dynamic": false,
  "half": false,
  "int8": false,
  "imgsz": 640,
  "batch": 1,
  "device": "cpu"
}
```

`dynamic` and `half` apply to official ONNX export. `int8=true` is TensorRT-only and requires calibration data; training uses the normalized YOLO `data.yaml`, while the model export page sends optional `data` when exporting `.pt` to TensorRT INT8. Existing `.onnx` inputs still use AITrain C++ copy / NCNN / TensorRT paths; `.pt` inputs use Worker-managed official Ultralytics export for ONNX/TensorRT, and `.pt -> ncnn` first creates a static FP32 official ONNX intermediate before `onnx2ncnn`.

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
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv4
```

That script reuses the isolated OCR environment and pinned PaddleOCR checkout, generates minimal PaddleOCR Det and Rec datasets, runs 1-epoch official Det and Rec train/export, then calls official `predict_system.py` with `use_angle_cls=false`. The default is PP-OCRv5 mobile Det/Rec; `-OcrVersion PP-OCRv4` switches back to the legacy v4 mobile presets. It validates reports, exported `official_inference/inference.yml` files, `official_system_prediction.json`, and visualized system output images. The run proves toolchain wiring and task artifacts, not useful OCR accuracy.

The PP-OCRv5 GPU production-chain wrapper is:

```powershell
.\tools\phase50-paddleocr-v5-gpu-official-chain.ps1 -UseGpu
```

This wrapper first checks that the selected OCR Python environment uses a CUDA-enabled PaddlePaddle build, then runs the production Det + Rec + System official chain with PP-OCRv5 presets. GPU mode is the default; `-UseGpu` is accepted as an explicit switch. Missing GPU support is recorded as `blocked`; the script must not downgrade to CPU and call the GPU gate passed.

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

This validates the historical YOLO11 and YOLO12 nano detection/segmentation candidates through the same official Ultralytics adapters, checks report/checkpoint/ONNX artifacts, and runs CTest against the generated work directory when a build tree is available. The P1 full matrix should be used for full model-family acceptance.

## Official PaddleOCR Adapter Parameters

The official Rec adapter accepts these extra parameters in addition to the common Python trainer request fields:

- `trainLabelFile`, `valLabelFile`, and `dictionaryFile` to use explicit PaddleOCR Rec materials.
- `modelPreset` to select `PP-OCRv4_mobile_rec`, `PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`, or `en_PP-OCRv5_mobile_rec`.
- `officialConfig` to start from a specific PaddleOCR recognition config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `runInferenceAfterExport=true` plus `inferenceImage` to run official `predict_rec.py` after export and write `official_prediction.json`.
- `recImageShape` to override the generated recognition image shape, for example `3,48,320`.

The final `paddleocr_official_rec_report.json` records PaddleOCR requested/resolved refs, `ocrVersion`, `modelPreset`, `resolvedOfficialConfig`, `resolvedModelName`, `configSource`, `presetDictionaryPath`, `recAlgorithm`, Python/Paddle/PaddleOCR versions, train/export/predict commands, config and label paths, dictionary path, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths.

The official Det adapter accepts these extra parameters:

- `trainLabelFile` and `valLabelFile` to use explicit PaddleOCR Det label files.
- `modelPreset` to select `PP-OCRv4_mobile_det`, `PP-OCRv5_mobile_det`, or `PP-OCRv5_server_det`.
- `officialConfig` to start from a specific PaddleOCR detection config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `imageSize` to override generated detection image size.

The final `paddleocr_official_det_report.json` records PaddleOCR requested/resolved refs, `ocrVersion`, `modelPreset`, `resolvedOfficialConfig`, `resolvedModelName`, `configSource`, Python/Paddle/PaddleOCR versions, train/export commands, config and label paths, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths.

The official System adapter accepts these parameters:

- `detModelDir`: exported PaddleOCR Det inference model directory.
- `recModelDir`: exported PaddleOCR Rec inference model directory.
- `dictionaryFile`: recognition dictionary file.
- `inferenceImage`: image or directory to pass to official `predict_system.py`.
- `dropScore`: optional recognition score threshold.
- `useGpu`: default `false`.
- `detModelPreset`, `recModelPreset`, and `recReportPath`: optional metadata used to select the correct official `predict_system.py` recognition algorithm. `PP-OCRv5_server_rec` uses `SVTR_HGNet`; mobile and v4 presets use `SVTR_LCNet`.

The final `paddleocr_official_system_report.json` records Python/Paddle/PaddleOCR versions, source checkout ref, Det/Rec presets, recognition algorithm, command, exit code, log path, model directories, dictionary path, `official_system_prediction.json`, `system_results.txt`, and the visualization directory.

## Historical OCR ONNX Wiring Evidence

AITrain's OCR product route is official-only. Training, export, prediction, evaluation, and customer acceptance should use the PaddleOCR official Det, Rec, and System adapters and their official reports.

Phase 46/47 C++ OCR ONNX work is retained only as historical wiring evidence for older validation lanes. It is not a production OCR inference, benchmark, deployment, or acceptance route. The `aitrain_worker --ocr-det-onnx-smoke` compatibility option now reports `blocked` with an official-only message instead of running AITrain C++ OCR postprocess.

Historical RTX 4090 Phase 47 evidence may remain in delivery archives to explain past wiring coverage, but new OCR closeout must cite PaddleOCR official Det/Rec/System reports and customer-domain acceptance outputs.

Use official `predict_system.py` for complete Det+Rec system validation. Use `paddleocr_official_rec_report.json`, `paddleocr_official_det_report.json`, and `paddleocr_official_system_report.json` as the OCR evidence set.

If public dataset materialization fails or requires external interaction, the generated minimal datasets remain the required smoke baseline. The failure reason should be recorded as an external data acquisition blocker, not hidden as a successful public dataset run.

## Known Boundaries

- TensorRT engine building has passing RTX 4090 D acceptance evidence under `.deps/rtx4090-validation/acceptance-tensorrt`; older unsupported GPUs should still report `hardware-blocked`.
- P1 covers YOLOv8 / YOLO11 / YOLO12 detection and instance-segmentation presets only; it does not productize YOLO26, semantic segmentation, tracking, classification, pose, OBB, anomaly, YOLO-World, or YOLOE.
- Historical GTX 1060 / SM 61 machines can run CPU training smoke and ONNX Runtime checks, but they cannot validate TensorRT 10 engine building and must not override RTX 4090 acceptance evidence.
- C++ segmentation mask ONNX postprocess is available for YOLO segmentation smoke models.
- OCR product inference, benchmark, evaluation, and acceptance are official-only. Historical C++ OCR ONNX wiring evidence must not be used as a current production OCR route.
- PP-OCRv5 support in this phase covers official Det/Rec/System OCR only. It does not productize PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document orientation classification, document unwarping, text-line orientation classification, or PaddleOCR C++ local deployment.
- Official third-party backend licensing must be reviewed before commercial redistribution.
- The official PaddleOCR adapter should be run in an isolated OCR Python environment. Mixing PaddlePaddle and PyTorch in one Windows Python process can trigger DLL conflicts through newer `albumentations` builds.
- The official PP-OCRv4/v5 smoke uses a tiny generated dataset; it validates train/export/inference wiring and artifacts, not useful OCR accuracy.
