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

Generate minimal smoke datasets and request JSON files with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```
