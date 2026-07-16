# Python Trainer Protocol

Phase 8 uses a subprocess boundary for real training. `aitrain_worker` launches a Python trainer, passes one request JSON file, then reads newline-delimited JSON from stdout.

## Request

The Worker writes `python_trainer_request.json` in the task output directory and launches:

```powershell
python <selected-official-trainer> --request <request-json>
```

For Phase 9 official YOLO detection training, Worker routes `trainingBackend=ultralytics_yolo_detect` or `trainingBackend=ultralytics_yolo` to:

```powershell
python python_trainers/detection/ultralytics_trainer.py --request <request-json>
```

For official YOLO segmentation training, Worker routes `trainingBackend=ultralytics_yolo_segment` to:

```powershell
python python_trainers/segmentation/ultralytics_trainer.py --request <request-json>
```

For official YOLO OBB training, Worker routes `trainingBackend=ultralytics_yolo_obb` to:

```powershell
python python_trainers/obb/ultralytics_trainer.py --request <request-json>
```

OBB uses `taskType=obb_detection`, `datasetFormat=yolo_obb`, and `modelFamily=yolo_obb`. Training, first ONNX export, and evaluation are official Ultralytics OBB operations. Packaged OBB inference, overlay, benchmark, and deployment validation use AITrain C++ ONNX Runtime with rotated-box output. NCNN is not an OBB v1 target, and TensorRT runtime OBB inference is not claimed.

YOLO detection/segmentation/OBB 的评估由  八步训练 Workflow 的 `Evaluate` 步骤启动官方 evaluator；已删除独立的裸模型路径 `evaluateModel` Worker 命令：

```powershell
python python_trainers/yolo/ultralytics_evaluator.py --request <request-json>
```

The evaluator calls official `YOLO(modelPath).val()` and writes an AITrain-compatible `evaluation_report.json` with `evaluationSource=ultralytics_official_val`. AITrain no longer computes local YOLO AP/mAP, rotated AP, mask IoU, TP/FP/FN, local error samples, local confusion CSV, or local evaluation overlays.

For dedicated semantic segmentation, Worker routes `trainingBackend=smp_semantic_segmentation` to:

```powershell
python python_trainers/semantic_segmentation/smp_trainer.py --request <request-json>
```

SMP evaluation uses:

```powershell
python python_trainers/semantic_segmentation/smp_evaluator.py --request <request-json>
```

SMP uses `taskType=semantic_segmentation`, `datasetFormat=semantic_segmentation_mask`, and `modelFamily=semantic_segmentation`. Datasets use `classes.txt` plus single-channel PNG class-id masks. Training exports `best.pt`, `best.onnx`, `smp_training_report.json`, and `semantic_segmentation_sidecar.json`; product inference, overlay, benchmark, and deployment validation are ONNX Runtime-only. NCNN/TensorRT export is outside SMP scope.

For anomaly detection/localization, Worker routes `trainingBackend=anomalib_patchcore` or `trainingBackend=anomalib_efficientad` to:

```powershell
python python_trainers/anomaly/anomalib_adapter.py --request <request-json>
```

Anomaly v1 uses `taskType=anomaly_detection`, `datasetFormat=anomaly_folder`, `modelFamily=anomaly_detection`, and `runtime=anomalib_python`. It writes Worker-managed Python/Anomalib artifacts such as `anomalib_training_report.json`, `anomaly_sidecar.json`, `evaluation_report.json`, `inference_predictions.json`, heatmaps, overlays, masks, and benchmark/deployment reports. It does not add AITrain C++ ONNX/TensorRT/NCNN anomaly runtime.

For the official PaddleOCR PP-OCRv4 / PP-OCRv5 / PP-OCRv6 Rec adapter, Worker routes `trainingBackend=paddleocr_rec_official` or the compatibility alias `trainingBackend=paddleocr_ppocrv4_rec` to:

```powershell
python python_trainers/ocr_rec/paddleocr_official_adapter.py --request <request-json>
```

Use `modelPreset` to select the official family. The default is `PP-OCRv5_mobile_rec`; `PP-OCRv4_mobile_rec`, `PP-OCRv5_server_rec`, `en_PP-OCRv5_mobile_rec`, and PP-OCRv6 `tiny/small/medium` Rec presets remain selectable. Use `prepareOnly=true` to generate and validate the selected official config, label lists, dictionary selection, report, and reproducible command files without running official training. PP-OCRv5/v6 built-in presets are resolved from a PaddleOCR source checkout and fail with `paddleocr_repo_missing` when no repo is available. PP-OCRv6 reads dictionary and algorithm metadata from the official config. Use `runOfficial=true` or `prepareOnly=false` with `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` pointing at a PaddleOCR source checkout to execute official `tools/train.py` and `tools/export_model.py`. Set `runInferenceAfterExport=true` with `inferenceImage` to run official `tools/infer/predict_rec.py` after export and write `official_prediction.json`.

The former small CTC route for `trainingBackend=paddleocr_rec` has been removed. `paddleocr_rec` remains a dataset format only; use `paddleocr_rec_official` or `paddleocr_ppocrv4_rec` for production OCR Rec training.

For the official PaddleOCR PP-OCRv4 / PP-OCRv5 / PP-OCRv6 Det adapter, Worker routes `trainingBackend=paddleocr_det_official` to:

```powershell
python python_trainers/ocr_det/paddleocr_det_official_adapter.py --request <request-json>
```

Use `modelPreset` to select `PP-OCRv5_mobile_det`, `PP-OCRv4_mobile_det`, `PP-OCRv5_server_det`, or PP-OCRv6 `tiny/small/medium` Det presets; the default is `PP-OCRv5_mobile_det`. Use `prepareOnly=true` to generate and validate the selected official detection config, label lists, report, and reproducible command files without running official training. PP-OCRv5/v6 built-in presets are resolved from a PaddleOCR source checkout and fail with `paddleocr_repo_missing` when no repo is available. Use `runOfficial=true` or `prepareOnly=false` with `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` pointing at a PaddleOCR source checkout to execute official `tools/train.py` and `tools/export_model.py`.

For official PaddleOCR end-to-end System inference, the Phase 31 smoke and official OCR toolchain call the inference adapter directly:

```powershell
python python_trainers/ocr_system/paddleocr_system_official_adapter.py --request <request-json>
```

This adapter does not train and is not exposed as a production training backend. It calls official `tools/infer/predict_system.py` with exported Det and Rec inference model directories, a recognition dictionary, and an image or image directory. It keeps `use_angle_cls=false` by default. The recognition algorithm is selected from `recModelPreset`, `recReportPath`, `recModelDir/inference.yml`, or explicit `recAlgorithm`; `PP-OCRv5_server_rec` uses `SVTR_HGNet`, v4/v5 mobile presets use `SVTR_LCNet`, and PP-OCRv6 requires Rec report, Rec `inference.yml`, or explicit `recAlgorithm` metadata.

The local isolated official smoke is:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

The full official PaddleOCR Det + Rec + System smoke is:

```powershell
.\tools\phase31-paddleocr-full-official-smoke.ps1
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv4
.\tools\phase50-paddleocr-v5-gpu-official-chain.ps1 -UseGpu
```

Request shape:

```json
{
  "protocolVersion": 1,
  "taskId": "task-id",
  "taskType": "detection",
  "datasetPath": "dataset-root",
  "outputPath": "run-root",
  "backend": "ultralytics_yolo_detect",
  "parameters": {},
  "request": {}
}
```

## Stdout Messages

Each stdout line must be a compact JSON object:

```json
{"type":"metric","payload":{"name":"loss","value":0.5,"step":1,"epoch":1}}
```

Supported message types:

- `log`
- `progress`
- `metric`
- `artifact`
- `completed`
- `failed`

The Worker adds `taskId` when a payload omits it, then forwards the business event to the GUI inside the Protocol  control envelope described below.

## GUI ↔ Worker Protocol  外层控制面

GUI 启动 Worker 时必须同时传入 `--server`、`--request-id` 与 `--task-id`。Worker 连接本地 Socket 后先发送 `event.ready`；GUI 随后只发送一次 `command.start_task`，取消只使用 `command.cancel_task`。旧 Dataset、Annotation、Report 等业务命令在迁移期放入 `command.start_task.payload.businessCommand/businessPayload`，不再直接作为 Socket 顶层 `type`。

Worker 事件统一使用 `event.ready`、`event.progress`、`event.metric`、`event.artifact`、`event.result`、`event.log`、`event.succeeded`、`event.failed` 或 `event.canceled`，原业务事件名与 payload 暂放在 `businessEvent/businessPayload`。双向各自维护严格递增的 `sequence`，每帧必须有唯一 `messageId`，且 request/task 身份必须与启动参数完全一致。控制帧上限为 1 MiB，日志帧上限为 64 KiB；跨请求/任务、重复、乱序、未知 kind、超限帧或终态后的事件会被拒绝并唯一收口。

## Cancellation

The Worker owns cancellation. If the GUI sends Protocol  `command.cancel_task`, the Worker terminates the Python subprocess and emits `event.canceled`；业务映射层再向现有 GUI signal 暴露 `canceled`。

## Official Backends

Official Python packages are adapted behind this protocol:

- `ultralytics_yolo_detect`: Ultralytics YOLO detection training. The adapter writes normalized `aitrain_yolo_data.yaml`, calls official `YOLO(...).train()`, exports ONNX, and forwards `best.pt`, `last.pt`, `results.csv`, `args.yaml`, `model.onnx`, and `ultralytics_training_report.json`.
- `ultralytics_yolo_segment`: Ultralytics YOLO segmentation training. It reuses the detection adapter with segmentation defaults such as `yolov8n-seg.yaml` and forwards mask metrics when the official results expose them.
- `ultralytics_yolo_obb`: Ultralytics YOLO OBB training. It uses official OBB training, ONNX export, and `val()` evaluation over YOLO OBB 9-column labels. Product deployment validation is ONNX Runtime-only for OBB v1.
- `smp_semantic_segmentation`: SMP dedicated semantic segmentation training and evaluation. It is separate from YOLO instance segmentation and uses Mask PNG datasets plus ONNX Runtime product inference/deployment validation.
- `anomalib_patchcore` / `anomalib_efficientad`: Anomalib anomaly detection/localization. It uses Worker-managed Python/Anomalib artifacts and `runtime=anomalib_python`; missing Anomalib or EfficientAD external data is blocked, not downgraded to a pass.
- `paddleocr_rec_official` / `paddleocr_ppocrv4_rec`: PaddleOCR official-recognition adapter. It prepares a PP-OCRv4, PP-OCRv5, or PP-OCRv6 config selected by `modelPreset` and can run official PaddleOCR training/export/inference from a source checkout. `prepareOnly` artifacts are configuration validation, not trained model artifacts.
  When official training runs, the adapter parses stdout metrics such as `loss`, `ctcLoss`, `nrtrLoss`, `accuracy`, and `normalizedEditDistance` into Worker `metric` events and the final report.
- `paddleocr_det_official`: PaddleOCR official-detection adapter. It prepares a PP-OCRv4, PP-OCRv5, or PP-OCRv6 Det config selected by `modelPreset` and can run official PaddleOCR training/export from a source checkout. `prepareOnly` artifacts are configuration validation, not trained model artifacts.
  When official training runs, the adapter parses stdout metrics such as `loss`, `hmean`, `precision`, and `recall` into Worker `metric` events and the final report.
- `paddleocr_system_official`: PaddleOCR official-system inference adapter for the official OCR toolchain, not a production training backend. It calls official `predict_system.py` with exported Det and Rec inference model directories and emits `official_system_prediction.json`, `system_results.txt`, and visualization-image artifacts. It is the current full OCR acceptance path; it is not C++ DB detection ONNX postprocess.

Diagnostic/test-only protocol coverage no longer uses shipped backend ids. Tests that need to exercise the subprocess protocol create a temporary trainer script, pass it through `parameters.pythonTrainerScript`, set a production backend id such as `ultralytics_yolo_detect`, and require `AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS=1`. Production requests should not set `pythonTrainerScript`.

Common Phase 9 detection parameters:

- `model`: default `yolov8n.pt`
- `epochs`: default `1`
- `batchSize` / `batch`: default `1`
- `imageSize` / `imgsz`: default `320`
- `device`: default `cpu`
- `workers`: default `0`
- `runName`: optional Ultralytics run name
- `exportOnnx`: default `true`
- `pythonPathPrepend`: optional test/dev-only module path injection
- `ultralyticsTrainArgs`: optional JSON object of whitelisted official training args such as `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, `patience`, `cos_lr`, `amp`, `cache`, `classes`, `freeze`, `mosaic`, `mixup`, `copy_paste`, `overlap_mask`, and `mask_ratio`

Common Phase 11 segmentation parameters are the same as detection, with default `model=yolov8n-seg.yaml`.

Common YOLO OBB parameters are the same as detection, with default `model=yolo11n-obb.pt`, `taskType=obb_detection`, and `datasetFormat=yolo_obb`.

Common SMP semantic segmentation parameters:

- `model`: default `smp_unet_resnet34`
- `epochs`: default adapter-dependent smoke value
- `batchSize` / `batch`
- `imageSize` / `imgsz`
- `device`: `cpu` or CUDA device
- `numClasses`, `ignoreIndex`, and class metadata from `classes.txt`
- `pythonExecutable`: optional Python path for SMP training/evaluation

Common Anomalib parameters:

- `modelPreset`: `anomalib_patchcore_wide_resnet50_2`, `anomalib_efficientad_s`, or compatible explicit settings
- `backbone`, `layers`, `coresetSamplingRatio`, and `numNeighbors` for PatchCore
- `modelSize=small|medium` for EfficientAD; legacy `s/m` may be normalized for compatibility
- `imagenetDir` or `AITRAIN_ANOMALIB_IMAGENET_DIR` for EfficientAD auxiliary data
- `threshold`, `imageSize`, `batchSize`, `device`, and `pythonExecutable`

Common YOLO evaluation options:

- `ultralyticsValArgs`: optional JSON object passed through the official evaluator after whitelist validation. Supported keys include `split`, `batch`, `imgsz`, `device`, `workers`, `conf`, `iou`, `max_det`, `half`, `dnn`, `plots`, `save_json`, `save_txt`, `save_conf`, `rect`, `classes`, `single_cls`, `augment`, `agnostic_nms`, `visualize`, and `end2end`.
- `pythonExecutable`: optional Python path for official evaluation.
- `pythonPathPrepend`: optional test/dev-only module path injection.

Common official PaddleOCR Rec parameters:

- `trainLabelFile`: optional explicit training label file.
- `valLabelFile`: optional explicit validation label file.
- `dictionaryFile`: optional explicit recognition dictionary.
- `modelPreset`: `PP-OCRv4_mobile_rec`, `PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`, `en_PP-OCRv5_mobile_rec`, `PP-OCRv6_tiny_rec`, `PP-OCRv6_small_rec`, or `PP-OCRv6_medium_rec`.
- `officialConfig`: optional official config source path.
- `pretrainedModel`: optional pretrained/export input checkpoint.
- `resumeCheckpoint`: optional official resume checkpoint.
- `exportOnly`: skip train and run export from an existing checkpoint.
- `runInferenceAfterExport`: run official recognition inference after export.
- `inferenceImage`: sample image for official recognition inference.
- `recImageShape`: generated recognition image shape, for example `3,48,320`.

Common official PaddleOCR Det parameters:

- `trainLabelFile`: optional explicit training label file.
- `valLabelFile`: optional explicit validation label file.
- `modelPreset`: `PP-OCRv4_mobile_det`, `PP-OCRv5_mobile_det`, `PP-OCRv5_server_det`, `PP-OCRv6_tiny_det`, `PP-OCRv6_small_det`, or `PP-OCRv6_medium_det`.
- `officialConfig`: optional official config source path.
- `pretrainedModel`: optional pretrained/export input checkpoint.
- `resumeCheckpoint`: optional official resume checkpoint.
- `exportOnly`: skip train and run export from an existing checkpoint.
- `imageSize`: generated detection image size.

Common official PaddleOCR System parameters:

- `detModelDir`: exported official Det inference model directory.
- `recModelDir`: exported official Rec inference model directory.
- `dictionaryFile`: recognition dictionary file.
- `inferenceImage`: image or directory for official system inference.
- `detModelPreset` / `recModelPreset`: optional metadata used for report lineage and recognition algorithm selection.
- `recReportPath`: optional Rec adapter report used to inherit `modelPreset`, `resolvedModelName`, dictionary, and `recAlgorithm`.
- `recAlgorithm`: optional explicit official recognition algorithm override.
- `dropScore`: optional recognition score threshold.
- `useGpu`: defaults to `false`.

The Det and Rec adapter reports include `ocrVersion`, `modelPreset`, `resolvedOfficialConfig`, `resolvedModelName`, `configSource`, and `presetDictionaryPath` (where applicable). When `officialConfig` overrides a built-in preset, `configSource` is `officialConfig_override`; reports must not describe override runs as built-in preset evidence.

Generate minimal smoke datasets and request JSON files with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```
