# YOLO Model Support Matrix

This document is the Phase 45 source of truth for Ultralytics YOLO model-family productization in AITrain Studio.

AITrain Studio currently owns the orchestration layer: Worker request routing, dataset normalization, artifact recording, ONNX export checks, and C++ ONNX Runtime smoke regression. The actual YOLO training still runs through the installed official `ultralytics` Python package, so supported model names depend on that package version and its license terms.

## Status Levels

| Status | Meaning |
|---|---|
| Product default | Exposed as the stable default path and covered by earlier local CPU smoke evidence. |
| Phase 45 validated | Covered by the Phase 45 model matrix smoke on this repository. |
| Candidate | The local Ultralytics package resolves the model name, but this repository has not recorded a required Phase 45 pass for it yet. |
| Out of scope | Not productized by the current detection/segmentation acceptance path. |

## Current Matrix

| Family / model | Task | AITrain backend | Status | Notes |
|---|---|---|---|---|
| `yolov8n.yaml` | Detection | `ultralytics_yolo_detect` | Product default | Earlier local CPU smoke produced `best.pt`, `best.onnx`, and `ultralytics_training_report.json`. |
| `yolov8n-seg.yaml` | Segmentation | `ultralytics_yolo_segment` | Product default | Earlier local CPU smoke produced `best.pt`, `best.onnx`, report metrics, and C++ ONNX Runtime segmentation smoke coverage. |
| `yolo11n.yaml` | Detection | `ultralytics_yolo_detect` | Phase 45 validated after smoke pass | Required Phase 45 model-matrix target. |
| `yolo11n-seg.yaml` | Segmentation | `ultralytics_yolo_segment` | Phase 45 validated after smoke pass | Required Phase 45 model-matrix target. |
| `yolo12n.yaml` | Detection | `ultralytics_yolo_detect` | Candidate | Optional Phase 45 target via `-IncludeYolo12`; do not call product default until a pass is recorded. |
| `yolo12n-seg.yaml` | Segmentation | `ultralytics_yolo_segment` | Candidate | Optional Phase 45 target via `-IncludeYolo12`; do not call product default until a pass is recorded. |

The local development baseline used for Phase 45 investigation has `ultralytics` 8.4.45 installed, and that package resolves the YOLO11 and YOLO12 nano detection/segmentation YAML names above.

## Acceptance Command

Run the required YOLO11 matrix:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

Optionally include YOLO12 candidates:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

The script writes `yolo_model_matrix_summary.json` under `.deps\phase45-yolo-model-matrix` by default. A required pass means each required model has:

- `ultralytics_training_report.json`
- `checkpointPath` resolving to a real `.pt` file
- `onnxPath` resolving to a real `.onnx` file
- report fields for `backend`, `model`, and `metrics`

When CTest is available, the script also points `AITRAIN_ACCEPTANCE_SMOKE_ROOT` at the Phase 45 work directory and runs the existing C++ ONNX Runtime regression tests against the generated artifacts.

## Boundaries

- Phase 45 productizes detection and segmentation model-family acceptance only.
- Classification, pose, OBB, anomaly, YOLO-World, and YOLOE are not productized by this matrix.
- TensorRT engine building remains external RTX / SM 75+ acceptance and is not changed by this phase.
- C++ PaddleOCR Det DB ONNX postprocess is unrelated to this YOLO matrix and remains tracked separately.
- Ultralytics licensing must be reviewed before redistribution of official backend dependencies or weights.
