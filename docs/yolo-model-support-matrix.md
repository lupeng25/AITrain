# YOLO Model Support Matrix

> 2026-07-16 破坏性重构说明：本文下方的旧 YOLO26/OBB 独立 smoke 命令仅保留为历史证据，脚本已删除。当前 YOLO 训练、评估、导出和交付分别由 `TrainingWorkflowProfileV2`、官方 Python 适配器和 `runRuntimeDeliveryWorkflowV2` 负责。

本文是 AITrain Studio Ultralytics YOLO 检测与实例分割模型族的历史支持矩阵。OBB 当前只通过 V2 Profile 与官方适配器描述，不再提供独立 smoke/matrix 脚本。

AITrain owns Worker routing, dataset normalization, artifact recording, official export checks, C++ ONNX Runtime / NCNN single-image inference, benchmark, deployment validation, TensorRT export/deployment status, and smoke regression. Training, first official export, and detection/segmentation/OBB evaluation still run through the installed official `ultralytics` Python package, so supported model resolution depends on that package version and its license terms.

## Productized Families

P1 expands the GUI and acceptance matrix from nano-only entries to full YOLOv8 / YOLO11 / YOLO12 detection and instance-segmentation presets, plus Ultralytics YOLOv5u standard P5 detection presets. YOLO26 is tracked as a separate compatibility phase and is not mixed into the P1 matrix:

| Family | Task | Source types | Scales | AITrain backend | Status |
|---|---|---|---|---|---|
| YOLOv5u | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLOv8 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO11 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO12 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO26 | Detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | Separate targeted matrix passed training/ONNX/TensorRT; NCNN is not a supported target |
| YOLOv8 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 full matrix required |
| YOLO11 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 full matrix required |
| YOLO12 | Segmentation | `.yaml`; `.pt` currently blocked unless official `yolo12*-seg.pt` resolves | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | P1 `.yaml` matrix required; `.pt` rows require upstream official weights |
| YOLO26 | Segmentation | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_segment` | Separate targeted matrix passed training/ONNX/TensorRT; NCNN is not a supported target |
| YOLOv8 P2/P6 | Detection architecture only | `.yaml` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_detect` | P1 full matrix required |
| YOLO11 OBB | OBB rotated-box detection | `.yaml`, `.pt` | `n`, `s`, `m`, `l`, `x` | `ultralytics_yolo_obb` | Separate OBB v1 route; ONNX Runtime deployment only; NCNN is not a supported OBB v1 target |

The GUI remains editable, so operators may type an official model name that is not listed here. Such runs are accepted only when the installed Ultralytics package resolves the model and the backend/task pairing is valid. They are not counted as P1 matrix evidence unless added to the matrix script.

YOLOv5 support follows the Ultralytics 8 YOLOv5u detection route. The P1 matrix uses `yolov5n.yaml` / `yolov5s.yaml` / `yolov5m.yaml` / `yolov5l.yaml` / `yolov5x.yaml` architecture entries and `yolov5nu.pt` / `yolov5su.pt` / `yolov5mu.pt` / `yolov5lu.pt` / `yolov5xu.pt` pretrained entries. Original `ultralytics/yolov5` repository weights are not an AITrain compatibility promise.

2026-06-14 full lifecycle finding: in the current validation environment with Ultralytics 8.3.171, `yolo12n-seg.pt` failed before training because the official package could not resolve `yolo12n-seg.pt`. The package asset stems expose YOLO12 detection weights but not `yolo12*-seg` weights. Treat YOLO12 segmentation `.pt` rows as `blocked_missing_official_weight` until official Ultralytics resolves those weights. YOLO12 segmentation `.yaml` architecture rows and YOLO12 detection `.pt` rows are separate routes and may still pass.

YOLO26 support is intended to follow official Ultralytics YOLO26 detection and instance-segmentation routes when the installed package supports them. The compatibility matrix uses `yolo26n/s/m/l/x.yaml`, `yolo26n/s/m/l/x.pt`, `yolo26n/s/m/l/x-seg.yaml`, and `yolo26n/s/m/l/x-seg.pt`. It does not include semantic segmentation, classification, pose, OBB, tracking, YOLOE-26, or other YOLO26 task variants.

OBB v1 uses `taskType=obb_detection`, `datasetFormat=yolo_obb`, `trainingBackend=ultralytics_yolo_obb`, and `modelFamily=yolo_obb`. Default GUI preset is `yolo11n-obb.pt`; YOLO11 OBB `n/s/m/l/x` `.pt` and `.yaml` are selectable. YOLO26 OBB is not part of this matrix or the OBB v1 default preset list; users may manually type an official Ultralytics model name, and unresolved names must be recorded as blocked.

2026-06-15 full lifecycle finding: all 20 YOLO26 rows failed in the shared validation environment with Ultralytics 8.3.171. `GITHUB_ASSETS_STEMS` contains no `yolo26n/s/m/l/x` or `yolo26*-seg` stems. The `.yaml` rows fail because files such as `yolo26n.yaml` and `yolo26x-seg.yaml` do not exist; most `.pt` rows fail because weights such as `yolo26x.pt` or `yolo26x-seg.pt` cannot be resolved; nano `.pt` rows expose package/code incompatibility (`SPPF.__init__()` argument mismatch and missing `Segment26`).

The isolated YOLO26 targeted environment then passed `tools\phase-yolo26-model-matrix-smoke.ps1 -Full -Epochs 100 -Device 0` on 2026-06-15 with 20/20 required rows passing training, official ONNX export, AITrain C++ ONNX inference, and TensorRT deployment validation. The historical YOLO26 NCNN attempt failed 20/20, so AITrain no longer offers or runs YOLO26 NCNN export/conversion; use ONNX or TensorRT for YOLO26 deployment.

## Export Parameters

YOLO official export arguments are recorded under `ultralyticsExportArgs`:

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

- Training still creates an ONNX artifact by default.
- `dynamic` and `half` are passed to official ONNX export.
- `int8=true` is treated as TensorRT engine export and requires calibration data from the normalized YOLO `data.yaml`.
- AITrain metadata accepts `end2end` as `auto`, `true`, or `false`; normalized reports store the final boolean value. `auto` now uses the loaded Ultralytics model config when it exposes an `end2end` default, then falls back to the historical rule of YOLO26 detection=true and other models=false.
- Pass `end2end` to official Ultralytics export only when the installed package/model exposes support or the user explicitly requests it. The 2026-06-14 full lifecycle run showed Ultralytics 8.3.171 rejects unsupported `end2end` arguments on generic ONNX export, so unsupported combinations must be recorded as failed or blocked.
- Model export requests may include optional `data` for `.pt -> TensorRT INT8` calibration.
- Existing `.onnx` inputs continue through AITrain C++ copy / NCNN conversion and TensorRT export/deployment validation paths.
- `.pt -> onnx` and `.pt -> tensorrt` run official Ultralytics Python export through Worker.
- `.pt -> ncnn` first runs a static FP32 traditional official ONNX export, then uses the existing `onnx2ncnn` path. This excludes YOLO26, where NCNN export/conversion is rejected as unsupported.
- OBB rejects `format=ncnn` in AITrain v1; use ONNX Runtime for OBB deployment validation. TensorRT engine export is optional status evidence only.
- NCNN rejects `dynamic`, `half`, `int8`, and `end2end=true`; ONNX rejects `int8`.

## Acceptance Command

Run the full P1 matrix:

```powershell
.\tools\phase-p1-yolo-full-matrix-smoke.ps1
```

The script writes `p1_yolo_full_matrix_summary.json` under `.deps\phase-p1-yolo-full-matrix` by default. In environments where every listed official asset resolves, a pass requires all required rows to produce the artifacts below. If an official upstream asset cannot be resolved, the row must be reported as `blocked_missing_official_weight` or an equivalent explicit blocker and must not be counted as passed; in the recorded Ultralytics 8.3.171 environment this applies to YOLO12 segmentation `.pt` rows.

- `best.pt`
- `best.onnx` or the official ONNX export path
- `ultralytics_training_report.json`
- report `model`, `backend`, `metrics`, and `ultralyticsExportArgs`
- `.pt` rows preserving the `.pt` model name in the report

The previous Phase 45 smoke remains as a faster historical YOLO11/YOLO12 nano wiring check:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

Run the separate YOLO26 compatibility matrix:

```powershell
```

The YOLO26 script writes `yolo26_model_matrix_summary.json` and `yolo26_environment_self_check.json` under `.deps\phase-yolo26-model-matrix` by default. Probe mode validates the isolated Python environment, CUDA Torch when `-Device 0` is requested, `cfg/models/26`, and nano `.yaml` / `.pt` model loading before any training. Full mode has 20 rows covering detection and instance segmentation `n/s/m/l/x` `.yaml` and `.pt` presets. Focused mode covers the four nano lifecycle rows: `yolo26n.yaml`, `yolo26n.pt`, `yolo26n-seg.yaml`, and `yolo26n-seg.pt`. In an environment where official YOLO26 assets resolve, each passed row must produce `best.pt`, ONNX, `ultralytics_training_report.json`, AITrain inference JSON, overlay output, and deployment statuses for ONNX and TensorRT. `ncnn` is not an accepted YOLO26 deployment target.

Run the separate OBB v1 smoke/matrix:

```powershell
```

Each passed OBB row must produce `best.pt`, `best.onnx`, `ultralytics_training_report.json`, official `evaluation_report.json`, AITrain prediction JSON with `xywhr` and four `points`, overlay output, benchmark report, and ONNX deployment validation report.

## Boundaries

- This matrix covers detection and instance segmentation only.
- YOLOv5u is detection-only in this matrix; YOLOv5 segmentation and YOLOv5 P6 variants remain outside P1.
- YOLO26 remains outside P1. The 2026-06-15 targeted full summary accepts YOLO26 training, official ONNX, AITrain C++ ONNX inference, and TensorRT evidence for the 20 required detection/instance-segmentation rows; YOLO26 NCNN is not a supported export or deployment target.
- Semantic segmentation, tracking, YOLOE-26, YOLO-World, classification, pose, and anomaly remain outside this detection/instance-segmentation matrix. OBB is supported only through the separate OBB v1 route and is not part of P1 or YOLO26 compatibility acceptance.
- Full matrix acceptance is a wiring/artifact/productization gate, not an accuracy benchmark.
- `.pt` weights are not bundled with AITrain Studio; official Ultralytics may download them into the user environment.
- YOLO12 segmentation `.pt` rows currently depend on official `yolo12*-seg.pt` weights that are not resolvable in the recorded Ultralytics 8.3.171 validation environment; record these as blocked instead of productized passes.
- Shared Ultralytics 8.3.171 environments must still record YOLO26 as `blocked_model_unavailable` / `blocked_ultralytics_incompatible`; customer preflight may only rely on the isolated targeted summary for training/ONNX/TensorRT and must not offer YOLO26 NCNN.
- Ultralytics licensing must be reviewed before redistribution of official backend dependencies or weights.
