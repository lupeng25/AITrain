# Python Trainer Protocol

Phase 8 uses a subprocess boundary for real training. `aitrain_worker` launches a Python trainer, passes one request JSON file, then reads newline-delimited JSON from stdout.

## Request

The Worker writes `python_trainer_request.json` in the task output directory and launches:

```powershell
python python_trainers/mock_trainer.py --request <request-json>
```

For Phase 9 official YOLO detection training, Worker routes `trainingBackend=ultralytics_yolo_detect` or `trainingBackend=ultralytics_yolo` to:

```powershell
python python_trainers/detection/ultralytics_trainer.py --request <request-json>
```

For official YOLO segmentation training, Worker routes `trainingBackend=ultralytics_yolo_segment` to:

```powershell
python python_trainers/segmentation/ultralytics_trainer.py --request <request-json>
```

For PaddlePaddle OCR Rec training, Worker routes `trainingBackend=paddleocr_rec` to:

```powershell
python python_trainers/ocr_rec/paddleocr_trainer.py --request <request-json>
```

For the official PaddleOCR PP-OCRv4 Rec adapter, Worker routes `trainingBackend=paddleocr_rec_official` or `trainingBackend=paddleocr_ppocrv4_rec` to:

```powershell
python python_trainers/ocr_rec/paddleocr_official_adapter.py --request <request-json>
```

Use `prepareOnly=true` to generate and validate the PP-OCRv4 config, label lists, dictionary copy, report, and reproducible command files without running official training. Use `runOfficial=true` or `prepareOnly=false` with `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` pointing at a PaddleOCR source checkout to execute official `tools/train.py` and `tools/export_model.py`. Set `runInferenceAfterExport=true` with `inferenceImage` to run official `tools/infer/predict_rec.py` after export and write `official_prediction.json`.

The local isolated official smoke is:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

Request shape:

```json
{
  "protocolVersion": 1,
  "taskId": "task-id",
  "taskType": "detection",
  "datasetPath": "dataset-root",
  "outputPath": "run-root",
  "backend": "python_mock",
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

The Worker adds `taskId` when a payload omits it, then forwards the message to the GUI over the existing Worker protocol.

## Cancellation

The Worker owns cancellation. If the GUI sends `cancel`, the Worker terminates the Python subprocess and emits `canceled`.

## Official Backends

Official Python packages are adapted behind this protocol:

- `ultralytics_yolo_detect`: Ultralytics YOLO detection training. The adapter writes normalized `aitrain_yolo_data.yaml`, calls official `YOLO(...).train()`, exports ONNX, and forwards `best.pt`, `last.pt`, `results.csv`, `args.yaml`, `model.onnx`, and `ultralytics_training_report.json`.
- `ultralytics_yolo_segment`: Ultralytics YOLO segmentation training. It reuses the detection adapter with segmentation defaults such as `yolov8n-seg.yaml` and forwards mask metrics when the official results expose them.
- `paddleocr_rec`: PaddlePaddle CTC OCR recognition training for PaddleOCR-style Rec data. It produces `paddleocr_rec_ctc.pdparams`, `dict.txt`, and `paddleocr_rec_training_report.json`. This is not a full PP-OCRv4 official config/export pipeline yet.
- `paddleocr_rec_official` / `paddleocr_ppocrv4_rec`: PaddleOCR official-recognition adapter. It prepares a PP-OCRv4 config and can run official PaddleOCR training/export/inference from a source checkout. `prepareOnly` artifacts are configuration validation, not trained model artifacts.
  When official training runs, the adapter parses stdout metrics such as `loss`, `ctcLoss`, `nrtrLoss`, `accuracy`, and `normalizedEditDistance` into Worker `metric` events and the final report.

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

Common Phase 11 segmentation parameters are the same as detection, with default `model=yolov8n-seg.yaml`.

Common Phase 12 OCR Rec parameters:

- `epochs`: default `1`
- `batchSize`: default `2`
- `imageWidth`: default `96`
- `imageHeight`: default `32`
- `maxTextLength`: default `32`
- `learningRate`: default `0.001`

Common official PaddleOCR Rec parameters:

- `trainLabelFile`: optional explicit training label file.
- `valLabelFile`: optional explicit validation label file.
- `dictionaryFile`: optional explicit recognition dictionary.
- `officialConfig`: optional official config source path.
- `pretrainedModel`: optional pretrained/export input checkpoint.
- `resumeCheckpoint`: optional official resume checkpoint.
- `exportOnly`: skip train and run export from an existing checkpoint.
- `runInferenceAfterExport`: run official recognition inference after export.
- `inferenceImage`: sample image for official recognition inference.
- `recImageShape`: generated recognition image shape, for example `3,48,320`.

Generate minimal smoke datasets and request JSON files with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```
