# AITrain Studio Training Backends

> 2026-07-16 破坏性重构说明：本文中早期 Phase 的独立矩阵脚本和 Worker 裸路径 smoke 命令均为历史记录，相关脚本已删除。当前训练只能通过注册的 `TrainingWorkflowProfile` 和 Worker  Workflow 执行；模型交付只能通过 `runRuntimeDeliveryWorkflow`，不得直接向 Worker 传入模型/样本路径。

AITrain Studio 将训练置于 GUI 进程之外。真实训练由 `aitrain_worker` 作为 Python 子进程启动，并通过 Worker  协议回传换行 JSON 事件。

## Backend Summary

Production training is limited to Worker-managed official or upstream-maintained adapters. The GUI training page and Worker production whitelist expose these training backends:

| Backend | Task | Status | Notes |
|---|---|---|---|
| `ultralytics_yolo_detect` | Detection | Official Ultralytics adapter | Uses Ultralytics YOLO detection training and official export. P1 exposes YOLOv5u standard P5 detection presets, YOLOv8 / YOLO11 / YOLO12 `n/s/m/l/x` `.yaml` and `.pt` presets, plus YOLOv8 P2/P6 detection YAML architectures. YOLO26 detection `n/s/m/l/x` `.yaml` and `.pt` presets are tracked in the separate YOLO26 compatibility matrix; the 2026-06-15 isolated targeted full matrix passed training/ONNX/TensorRT, and NCNN is not offered for YOLO26. Product evaluation uses Ultralytics official `val()`; product inference, benchmark, and deployment validation run against official artifacts through the AITrain C++ runtime. Review AGPL-3.0 / Enterprise license before redistribution. |
| `ultralytics_yolo_segment` | Segmentation | Official Ultralytics adapter | Uses Ultralytics YOLO instance-segmentation training and official export. P1 exposes YOLOv8 / YOLO11 `n/s/m/l/x` `-seg.yaml` and `-seg.pt` presets, plus YOLO12 `n/s/m/l/x` `-seg.yaml` presets. YOLO12 `-seg.pt` depends on upstream official weights and is currently blocked in the recorded Ultralytics 8.3.171 lifecycle run because `yolo12n-seg.pt` could not be resolved. YOLO26 instance-segmentation `n/s/m/l/x-seg` `.yaml` and `.pt` presets are tracked in the separate YOLO26 compatibility matrix; the 2026-06-15 isolated targeted full matrix passed training/ONNX/TensorRT, and NCNN is not offered for YOLO26. Product evaluation uses Ultralytics official `val()`; product inference, benchmark, and deployment validation run against official artifacts through the AITrain C++ runtime, including mask postprocess and overlays. |
| `ultralytics_yolo_obb` | OBB rotated-box detection | Official Ultralytics adapter | Uses Ultralytics YOLO OBB training, official `val()`, and ONNX export for `taskType=obb_detection`, `datasetFormat=yolo_obb`, and `modelFamily=yolo_obb`. Datasets use 9-column labels: `class x1 y1 x2 y2 x3 y3 x4 y4`, with normalized corner coordinates. Product inference, overlay, benchmark, and deployment validation run through AITrain C++ ONNX Runtime with rotated-polygon output JSON (`xywhr`, `points[4]`, and external `bbox`). NCNN is not an OBB v1 deployment target; TensorRT engine export is optional status evidence only. Default GUI preset is `yolo11n-obb.pt`, with YOLO11 OBB `.pt` and `.yaml` presets selectable. |
| `smp_semantic_segmentation` | Semantic segmentation | SMP adapter | Uses `segmentation_models.pytorch` for dedicated per-pixel semantic segmentation, separate from YOLO instance segmentation. The public presets are `smp_unet_resnet34`, `smp_unetplusplus_resnet34`, `smp_fpn_resnet34`, `smp_deeplabv3plus_resnet50`, and `smp_segformer_mit_b0`. Datasets use `classes.txt` plus `images/{train,val,test}` and `masks/{train,val,test}` single-channel PNG class-id masks. Training exports `best.pt`, `best.onnx`, `smp_training_report.json`, and `semantic_segmentation_sidecar.json`; evaluation reports mIoU, mean Dice, pixel accuracy, per-class metrics, confusion matrix, low-quality samples, and overlays. Product inference, overlay, benchmark, and deployment validation are ONNX Runtime only; NCNN/TensorRT export is not part of the SMP capability scope and is not required for SMP acceptance. Review SMP, Torch, timm, ONNX, and ONNX Runtime licenses before redistribution. |
| `anomalib_patchcore` | Anomaly detection | Anomalib adapter | Default anomaly backend for `taskType=anomaly_detection`, `datasetFormat=anomaly_folder`, and `modelFamily=anomaly_detection`. The default preset is `anomalib_patchcore_wide_resnet50_2` with `backbone=wide_resnet50_2`, `layers=layer2,layer3`, `coresetSamplingRatio=0.1`, and `numNeighbors=9`. Training/inference/evaluation/benchmark use Worker-managed Python/Anomalib artifacts and write `anomalib_training_report.json`, `anomaly_sidecar.json`, `evaluation_report.json`, `inference_predictions.json`, heatmap, overlay, and binary mask outputs. |
| `anomalib_efficientad` | Anomaly detection | Anomalib adapter | Secondary anomaly backend. The default preset is `anomalib_efficientad_s` with `modelSize=small`, `lr=0.0001`, `weightDecay=0.00001`, and Anomalib 2.5 training `batchSize=1`. AITrain accepts legacy `s/m` input but sends `small/medium` to Anomalib. EfficientAD requires `imagenetDir`, `AITRAIN_ANOMALIB_IMAGENET_DIR`, or `.deps/anomalib/imagenette`; missing data is reported as `efficientad_imagenet_dir_missing` / blocked and AITrain does not auto-download external data. |
| `paddleocr_det_official` | OCR detection | Official PaddleOCR adapter | Generates PP-OCRv4, PP-OCRv5, or PP-OCRv6 detection configs from PaddleOCR Det data and can run official PaddleOCR `tools/train.py` and `tools/export_model.py`. The default preset remains `PP-OCRv5_mobile_det`; `PP-OCRv4_mobile_det`, `PP-OCRv5_server_det`, and PP-OCRv6 `tiny/small/medium` Det presets are selectable. |
| `paddleocr_rec_official` | OCR recognition | Official PaddleOCR adapter | Generates PP-OCRv4, PP-OCRv5, or PP-OCRv6 recognition configs from AITrain PaddleOCR-style Rec data and runs official PaddleOCR `tools/train.py`, `tools/export_model.py`, and optional `tools/infer/predict_rec.py` when `runOfficial=true` and `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` points to a checkout. The default preset remains `PP-OCRv5_mobile_rec`; PP-OCRv6 `tiny/small/medium` Rec presets are selectable. |

`paddleocr_system_official` remains the official OCR System inference/validation adapter. It is not shown as a "train model" backend because it runs official `predict_system.py` against exported Det and Rec inference model directories.

Legacy diagnostic training implementations have been physically removed from production packages and training routing. `paddleocr_rec` remains a dataset format, not a training backend. Protocol tests that need a Python trainer now create an explicit temporary fixture through `pythonTrainerScript` and require `AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS=1`; no shipped `python_mock` trainer is provided.

## YOLO Runtime Boundary

YOLO detection, instance segmentation, and OBB are not end-to-end official-only routes. AITrain uses official Ultralytics code for training, ONNX export, and YOLO evaluation via `YOLO(...).val()`, then treats the exported artifacts as deployment inputs for its local runtime checks. C++ ONNX Runtime handles detection, segmentation, and OBB single-image prediction JSON, overlays, benchmark summaries, and deployment validation. NCNN remains limited to supported YOLO detection/segmentation artifacts and is explicitly rejected for OBB v1. TensorRT currently covers engine export and deployment validation status, not single-image runtime decoding in the GUI.

AITrain no longer computes local YOLO AP/mAP, mask IoU, TP/FP/FN, local evaluation error samples, or local evaluation overlays. The `evaluation_report.json` wrapper remains for the GUI and task history, with `evaluationSource=ultralytics_official_val` and links to official plots/predictions when Ultralytics writes them. OCR is different: current OCR product inference, evaluation, benchmark, deployment validation, and acceptance are official-only through PaddleOCR Det/Rec/System reports. Anomaly detection is also different from YOLO C++ runtime: v1 uses `runtime=anomalib_python` only and does not create AITrain C++ ONNX/TensorRT/NCNN anomaly deployment support.

## Environment Setup

Use an isolated Python environment. The local development machine used Python 3.13 embeddable under `.deps`, but a regular venv is preferred for users.

YOLO detection, instance segmentation, and OBB:

```powershell
python -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r python_trainers\requirements-yolo.txt
```

GPU YOLO training with `device=0` requires the selected Python to provide CUDA-enabled PyTorch. A CPU-only YOLO environment must use `device=cpu`; otherwise Ultralytics fails before training with an invalid CUDA device error. For packaged or validation runs, set training parameter `pythonExecutable` or `AITRAIN_PYTHON_EXECUTABLE` to the CUDA-capable YOLO Python path and verify `torch.cuda.is_available()` before starting long jobs.

YOLO model-family status is tracked in `docs\yolo-model-support-matrix.md`. The product defaults remain `yolov8n.yaml` for detection, `yolov8n-seg.yaml` for segmentation, and `yolo11n-obb.pt` for OBB. P1 adds editable GUI presets for YOLOv5u standard P5 detection, YOLOv8 / YOLO11 / YOLO12 detection across `n/s/m/l/x` with `.yaml` and `.pt`, YOLOv8 / YOLO11 instance segmentation with `-seg.yaml` and `-seg.pt`, YOLO12 instance segmentation with `-seg.yaml`, plus YOLOv8 P2/P6 detection YAML architectures. OBB v1 exposes YOLO11 OBB `n/s/m/l/x` `.pt` and `.yaml` presets; YOLO26 OBB remains manual/future compatibility only and is not a default preset. YOLO12 `-seg.pt` is currently a blocked upstream-weight case in Ultralytics 8.3.171, not a data or AITrain runtime failure. YOLO26 detection and instance-segmentation presets are imported as an independent compatibility phase, not into P1; the isolated YOLO26 full matrix gates training/ONNX/TensorRT only. YOLO26 NCNN export/conversion is not a product option. YOLOv5u uses `yolov5n/s/m/l/x.yaml` architecture entries and `yolov5nu/su/mu/lu/xu.pt` pretrained entries; original `ultralytics/yolov5` repository weights are not a compatibility promise. `.pt` weights are not bundled; the installed Ultralytics package may download them into the user environment.

Run the full P1 productization matrix with:

```powershell
.\tools\acceptance-smoke.ps1
```

The previous Phase 45 matrix remains useful as a faster YOLO11/YOLO12 nano wiring check:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

The P1 matrix does not productize YOLOv5 segmentation, YOLOv5 P6, YOLO26, semantic segmentation, tracking, classification, pose, anomaly, YOLO-World, or YOLOE. OBB is covered by its own Ultralytics OBB route, not by P1. YOLO26 records are archival only and do not include semantic segmentation, classification, pose, OBB, tracking, or YOLOE-26.

OBB v1 smoke and public matrix:

```powershell
```

`phase-obb-ultralytics-smoke.ps1` tries to materialize a public Ultralytics DOTA OBB dataset first, then falls back to generated OBB workflow data unless `-RequirePublicDataset` is set. A passing row must produce `best.pt`, `best.onnx`, `ultralytics_training_report.json`, official `evaluation_report.json`, and Worker `--obb-onnx-smoke` output with prediction JSON, overlay, benchmark report, and deployment validation report. Generated fallback rows validate workflow wiring only. DOTA/DOTA-subset rows are public benchmark evidence and are not customer-domain industrial precision evidence.

Dedicated semantic segmentation is a separate SMP route, not part of the YOLO P1/YOLO26 matrix:

```powershell
python -m venv .venv-smp
.\.venv-smp\Scripts\python.exe -m pip install -r python_trainers\requirements-smp.txt
.\tools\phase-smp-semantic-segmentation-smoke.ps1 -Python .\.venv-smp\Scripts\python.exe
```

Use `phase-smp-semantic-segmentation-smoke.ps1 -SkipTraining` when only verifying package layout and Python script compilation. A passing minimal smoke must produce `best.pt`, `best.onnx`, `smp_training_report.json`, `semantic_segmentation_sidecar.json`, `evaluation_report.json`, and non-empty overlay output. If SMP dependencies are missing, the smoke writes a blocked summary instead of treating generated data as product evidence.

The archived RTX 4090D SMP evidence under `.deps\smp-realtest\gpu-4090d` covers CUDA PyTorch training, exported ONNX evaluation, AITrain C++ ONNX Runtime inference, overlay, timing, and deployment validation. The historical GPU smoke script and the removed `--semantic-onnx-smoke` entry point are not part of the current product and must not be invoked; current acceptance uses the Runtime Delivery workflow.

Oxford-IIIT Pet comparison remains historical evidence only; its former matrix script was removed. Do not treat the archived quality files as a current product gate or customer-domain precision evidence.

Anomaly detection uses Anomalib:

```powershell
python -m venv .venv-anomaly
.\.venv-anomaly\Scripts\python.exe -m pip install -r python_trainers\requirements-anomaly.txt
.\tools\phase-anomaly-anomalib-smoke.ps1 -PythonExecutable .\.venv-anomaly\Scripts\python.exe
.\tools\phase-anomaly-mvtec-quality-matrix.ps1 -PythonExecutable .\.deps\envs\anomalib\python.exe
```

Datasets use `anomaly_folder`, compatible with MVTec-style folders: required `train/good/*`; optional `val/good`, `val/anomaly`, `test/good`, `test/anomaly`; optional masks under `masks/<split>/anomaly/<stem>.png`; and MVTec aliases `test/<defect_type>/*` plus `ground_truth/<defect_type>/<stem>_mask.png`. Good-only data can train one-class anomaly models but evaluation is marked `limited`. Passing anomaly evidence must include the Anomalib training report, sidecar, evaluation or limited report, inference predictions with OK/NG score/threshold, heatmap/overlay/mask artifacts, and a benchmark/deployment validation report with `runtime=anomalib_python`. `.ckpt` inference uses Anomalib `Engine.predict(..., ckpt_path=...)`, not the removed legacy `TorchInferencer` path.

Use `phase-anomaly-mvtec-quality-matrix.ps1` for reproducible public MVTec evidence across `bottle`, `hazelnut`, and `leather` with both `anomalib_patchcore` and `anomalib_efficientad`. The script keeps its isolated Conda environment under `.deps\envs\anomalib`, matrix outputs under `.deps\anomaly-mvtec-quality-matrix`, MVTec data under `.deps\datasets`, and EfficientAD Imagenette data under `.deps\anomalib\imagenette`. If the official MVTec archive is missing, the summary is `blocked`; it must not be counted as a passed training row. On 2026-06-18, the default three-category matrix passed 6/6 rows locally and wrote `.deps\anomaly-mvtec-quality-matrix\anomaly_mvtec_quality_matrix_summary.json`; this is public-dataset workflow/quality evidence, not customer-domain production precision.

Current YOLO26 status: the 2026-06-14/15 shared lifecycle run showed all YOLO26 detection and instance-segmentation rows fail in Ultralytics 8.3.171 before useful training starts. `.yaml` entries report missing files, most `.pt` entries report missing official weights, and nano `.pt` entries expose package/code incompatibility. The isolated 2026-06-15 targeted full matrix passed 20/20 rows for training, official ONNX, AITrain C++ ONNX inference, and TensorRT validation. YOLO26 NCNN remains removed from supported export/deployment targets.

Official YOLO export parameters are recorded as `ultralyticsExportArgs` and are accepted by both training and model export flows:

```json
{
  "format": "onnx",
  "dynamic": false,
  "half": false,
  "int8": false,
  "end2end": false,
  "imgsz": 640,
  "batch": 1,
  "device": "cpu"
}
```

`dynamic` and `half` apply to official ONNX export. `int8=true` is TensorRT-only and requires calibration data; training uses the normalized YOLO `data.yaml`, while the model export page sends optional `data` when exporting `.pt` to TensorRT INT8. `end2end=auto` uses the loaded Ultralytics model config when it exposes an `end2end` default, then falls back to YOLO26 detection=true and other models=false; reports store the final boolean. Treat this as AITrain metadata until the installed Ultralytics export confirms the argument is supported for the requested model and format. The 2026-06-14 lifecycle run showed Ultralytics 8.3.171 rejects unsupported `end2end` arguments on generic ONNX export, so unsupported combinations must be recorded as failed or blocked rather than silently downgraded. Existing `.onnx` inputs still use AITrain C++ copy / NCNN conversion and TensorRT export/deployment validation paths; `.pt` inputs use Worker-managed official Ultralytics export for ONNX/TensorRT, and non-YOLO26 `.pt -> ncnn` first creates a static FP32 traditional official ONNX intermediate before `onnx2ncnn`. NCNN rejects `end2end=true`; YOLO26 rejects `format=ncnn` entirely.

OCR recognition:

```powershell
python -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install -r python_trainers\requirements-ocr.txt
```

`requirements-ocr.txt` requires PaddleOCR 3.7+ so the PP-OCRv6 config and model family are available.

Optional official PaddleOCR Det/Rec training and System inference require a PaddleOCR source checkout because the installed `paddleocr` package exposes inference pipelines, not the legacy `tools/train.py` training scripts:

```powershell
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git .deps\repos\PaddleOCR
$env:AITRAIN_PADDLEOCR_REPO = (Resolve-Path .deps\repos\PaddleOCR).Path
```

Reusable Python environments, source checkouts, SDKs, and external tools should follow `docs/deps-layout.md`. The canonical OCR Python paths are `.deps\envs\ocr-cpu` and `.deps\envs\ocr-gpu`; legacy paths remain fallback-only for older local worktrees.

The reproducible local smoke command is:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

That script uses an isolated OCR Python embeddable environment under `.deps\envs\ocr-cpu`, checks out a pinned PaddleOCR source ref under `.deps\repos\PaddleOCR`, installs pinned OCR smoke constraints unless disabled, runs official PP-OCRv4 Rec training for 1 epoch on CPU, exports the official inference model, runs official recognition inference on one generated sample image, and checks the checkpoint, inference config, prediction report, resolved source ref, and metrics report.

The full official PaddleOCR Det + Rec + System smoke is:

```powershell
.\tools\phase31-paddleocr-full-official-smoke.ps1
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv4
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv6 -PPOCRv6Tier tiny
.\tools\phase-ppocrv6-model-matrix-smoke.ps1
```

That script reuses the isolated OCR environment and PaddleOCR checkout, generates minimal PaddleOCR Det and Rec datasets, runs 1-epoch official Det and Rec train/export, then calls official `predict_system.py` with `use_angle_cls=false`. The default is PP-OCRv5 mobile Det/Rec; `-OcrVersion PP-OCRv4` switches back to the legacy v4 mobile presets, and `-OcrVersion PP-OCRv6 -PPOCRv6Tier tiny|small|medium` selects matching v6 Det/Rec presets. `phase-ppocrv6-model-matrix-smoke.ps1` checks all six v6 Det/Rec presets in prepare-only mode and runs one v6 tiny Det+Rec+System full-chain smoke. These runs prove toolchain wiring and task artifacts, not useful OCR accuracy.

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
- `modelPreset` to select `PP-OCRv4_mobile_rec`, `PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`, `en_PP-OCRv5_mobile_rec`, `PP-OCRv6_tiny_rec`, `PP-OCRv6_small_rec`, or `PP-OCRv6_medium_rec`.
- `officialConfig` to start from a specific PaddleOCR recognition config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `runInferenceAfterExport=true` plus `inferenceImage` to run official `predict_rec.py` after export and write `official_prediction.json`.
- `recImageShape` to override the generated recognition image shape, for example `3,48,320`.

The final `paddleocr_official_rec_report.json` records PaddleOCR requested/resolved refs, `ocrVersion`, `modelPreset`, `resolvedOfficialConfig`, `resolvedModelName`, `configSource`, `presetDictionaryPath`, `dictionarySource`, `recAlgorithm`, Python/Paddle/PaddleOCR versions, train/export/predict commands, config and label paths, dictionary path, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths. PP-OCRv6 uses the official config dictionary path when no explicit `dictionaryFile` is provided.

The official Det adapter accepts these extra parameters:

- `trainLabelFile` and `valLabelFile` to use explicit PaddleOCR Det label files.
- `modelPreset` to select `PP-OCRv4_mobile_det`, `PP-OCRv5_mobile_det`, `PP-OCRv5_server_det`, `PP-OCRv6_tiny_det`, `PP-OCRv6_small_det`, or `PP-OCRv6_medium_det`.
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
- `detModelPreset`, `recModelPreset`, and `recReportPath`: optional metadata used to select the correct official `predict_system.py` recognition algorithm. `PP-OCRv5_server_rec` uses `SVTR_HGNet`; PP-OCRv4/v5 mobile presets use `SVTR_LCNet`; PP-OCRv6 reads the algorithm from the Rec report, Rec `inference.yml`, or explicit `recAlgorithm`.

The final `paddleocr_official_system_report.json` records Python/Paddle/PaddleOCR versions, source checkout ref, Det/Rec presets, recognition algorithm, command, exit code, log path, model directories, dictionary path, `official_system_prediction.json`, `system_results.txt`, and the visualization directory.

## Historical OCR ONNX Wiring Evidence

AITrain's OCR product route is official-only. Training, export, prediction, evaluation, and customer acceptance should use the PaddleOCR official Det, Rec, and System adapters and their official reports.

Phase 46/47 C++ OCR ONNX work is retained only as historical wiring evidence for older validation lanes. It is not a production OCR inference, benchmark, deployment, or acceptance route. The `aitrain_worker --ocr-det-onnx-smoke` compatibility option now reports `blocked` with an official-only message instead of running AITrain C++ OCR postprocess.

Historical RTX 4090 Phase 47 evidence may remain in delivery archives to explain past wiring coverage, but new OCR closeout must cite PaddleOCR official Det/Rec/System reports and customer-domain acceptance outputs.

Use official `predict_system.py` for complete Det+Rec system validation. Use `paddleocr_official_rec_report.json`, `paddleocr_official_det_report.json`, and `paddleocr_official_system_report.json` as the OCR evidence set.

If public dataset materialization fails or requires external interaction, the generated minimal datasets remain the required smoke baseline. The failure reason should be recorded as an external data acquisition blocker, not hidden as a successful public dataset run.

## Known Boundaries

- TensorRT engine building has passing RTX 4090 D acceptance evidence archived in `docs/validation/rtx4090-validation-evidence-20260615.json`; older unsupported GPUs should still report `hardware-blocked`.
- P1 covers YOLOv5u standard P5 detection plus YOLOv8 / YOLO11 / YOLO12 detection and instance-segmentation presets only; it does not productize YOLOv5 segmentation, YOLOv5 P6, tracking, classification, pose, OBB, anomaly, YOLO-World, or YOLOE. Dedicated semantic segmentation is covered separately by the SMP route and remains ONNX Runtime-only in this first version.
- YOLO12 segmentation `.pt` rows are currently blocked by missing upstream official `yolo12*-seg.pt` resolution in the recorded Ultralytics 8.3.171 environment. Keep YOLO12 segmentation `.yaml` and YOLO12 detection `.pt` evidence separate.
- YOLO26 detection/segmentation is a separate compatibility phase. The shared Ultralytics 8.3.171 environment blocked/failed all 20 YOLO26 rows, but the isolated 2026-06-15 targeted full matrix passed 20/20 training, official ONNX export, AITrain C++ ONNX inference, and TensorRT validation. YOLO26 NCNN is removed from supported export/deployment targets.
- Progress dashboards that aggregate existing `row_summary.json` files can show historical failures from previous runs. Operators must filter by the current run start time or run id before interpreting failed counts as live failures.
- Historical GTX 1060 / SM 61 machines can run CPU training smoke and ONNX Runtime checks, but they cannot validate TensorRT 10 engine building and must not override RTX 4090 acceptance evidence.
- Worker self-check 不再检查 LibTorch；C++ LibTorch 训练路线已删除。YOLO、SMP、Anomalib 与 PaddleOCR 训练均由 Worker 管理的官方 Python 适配器负责，环境门禁只报告当前产品运行时依赖。
- C++ segmentation mask ONNX postprocess is available for YOLO instance-segmentation smoke models and for SMP semantic segmentation ONNX argmax masks.
- OCR product inference, benchmark, evaluation, and acceptance are official-only. Historical C++ OCR ONNX wiring evidence must not be used as a current production OCR route.
- PP-OCRv5 and PP-OCRv6 support in this phase covers official Det/Rec/System OCR only. It does not productize PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document orientation classification, document unwarping, text-line orientation classification, or PaddleOCR C++ local deployment. PP-OCRv6 tiny follows the official language-coverage limitation and is not a customer-domain production claim.
- Official third-party backend licensing must be reviewed before commercial redistribution.
- The official PaddleOCR adapter should be run in an isolated OCR Python environment. Mixing PaddlePaddle and PyTorch in one Windows Python process can trigger DLL conflicts through newer `albumentations` builds.
- The official PP-OCRv4/v5/v6 smoke uses a tiny generated dataset; it validates train/export/inference wiring and artifacts, not useful OCR accuracy.
