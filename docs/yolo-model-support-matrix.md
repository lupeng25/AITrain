# YOLO Model Support Matrix

This document is the source of truth for Ultralytics YOLO detection and instance-segmentation model-family productization in AITrain Studio.

AITrain owns Worker routing, dataset normalization, artifact recording, official export checks, C++ ONNX Runtime / NCNN single-image inference, benchmark, deployment validation, TensorRT export/deployment status, and smoke regression. Training, first official export, and detection/segmentation evaluation still run through the installed official `ultralytics` Python package, so supported model resolution depends on that package version and its license terms.

## Productized Families

P1 expands the GUI and acceptance matrix from nano-only entries to full YOLOv8 / YOLO11 / YOLO12 detection and instance-segmentation presets, plus Ultralytics YOLOv5u standard P5 detection presets:

| Family | Task | Source types | Scales | AITrain backend | Status |
|---|---|---|---|---|---|
| YOLOv5u | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLOv8 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO11 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO12 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLOv8 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 full matrix required |
| YOLO11 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 full matrix required |
| YOLO12 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 full matrix required |
| YOLOv8 P2/P6 | Detection architecture only | `.yaml` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |

The GUI remains editable, so operators may type an official model name that is not listed here. Such runs are accepted only when the installed Ultralytics package resolves the model and the backend/task pairing is valid. They are not counted as P1 matrix evidence unless added to the matrix script.

YOLOv5 support follows the Ultralytics 8 YOLOv5u detection route. The P1 matrix uses `yolov5n.yaml` / `yolov5s.yaml` / `yolov5m.yaml` / `yolov5l.yaml` / `yolov5x.yaml` architecture entries and `yolov5nu.pt` / `yolov5su.pt` / `yolov5mu.pt` / `yolov5lu.pt` / `yolov5xu.pt` pretrained entries. Original `ultralytics/yolov5` repository weights are not an AITrain compatibility promise.

## Export Parameters

YOLO official export arguments are recorded under `ultralyticsExportArgs`:

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

- Training still creates an ONNX artifact by default.
- `dynamic` and `half` are passed to official ONNX export.
- `int8=true` is treated as TensorRT engine export and requires calibration data from the normalized YOLO `data.yaml`.
- Model export requests may include optional `data` for `.pt -> TensorRT INT8` calibration.
- Existing `.onnx` inputs continue through AITrain C++ copy / NCNN conversion and TensorRT export/deployment validation paths.
- `.pt -> onnx` and `.pt -> tensorrt` run official Ultralytics Python export through Worker.
- `.pt -> ncnn` first runs a static FP32 official ONNX export, then uses the existing `onnx2ncnn` path.
- NCNN rejects `dynamic`, `half`, and `int8`; ONNX rejects `int8`.

## Acceptance Command

Run the full P1 matrix:

```powershell
.\tools\phase-p1-yolo-full-matrix-smoke.ps1
```

The script writes `p1_yolo_full_matrix_summary.json` under `.deps\phase-p1-yolo-full-matrix` by default. A pass requires all 80 rows to produce:

- `best.pt`
- `best.onnx` or the official ONNX export path
- `ultralytics_training_report.json`
- report `model`, `backend`, `metrics`, and `ultralyticsExportArgs`
- `.pt` rows preserving the `.pt` model name in the report

The previous Phase 45 smoke remains as a faster historical YOLO11/YOLO12 nano wiring check:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

## Boundaries

- This matrix covers detection and instance segmentation only.
- YOLOv5u is detection-only in this matrix; YOLOv5 segmentation and YOLOv5 P6 variants remain outside P1.
- YOLO26, semantic segmentation, tracking, YOLOE, YOLO-World, classification, pose, OBB, and anomaly remain outside P1.
- Full matrix acceptance is a wiring/artifact/productization gate, not an accuracy benchmark.
- `.pt` weights are not bundled with AITrain Studio; official Ultralytics may download them into the user environment.
- Ultralytics licensing must be reviewed before redistribution of official backend dependencies or weights.
