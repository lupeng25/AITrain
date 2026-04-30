# AITrain Studio Training Backends

AITrain Studio keeps training out of the GUI process. Real training is launched by `aitrain_worker` as a Python subprocess and reports newline-delimited JSON events back to the existing Worker protocol.

## Backend Summary

| Backend | Task | Status | Notes |
|---|---|---|---|
| `tiny_linear_detector` | Detection | Scaffold | C++ baseline for protocol, tests, and demos. Not real YOLO. |
| `ultralytics_yolo_detect` | Detection | Real Python backend | Uses Ultralytics YOLO detection training and ONNX export. Review AGPL-3.0 / Enterprise license before redistribution. |
| `ultralytics_yolo_segment` | Segmentation | Real Python backend | Uses Ultralytics YOLO segmentation training and ONNX export. C++ mask ONNX postprocess is still future work. |
| `paddleocr_rec` | OCR recognition | Real PaddlePaddle CTC smoke backend | Trains a small PaddlePaddle CTC recognizer on PaddleOCR-style Rec data. Not a full PP-OCRv4 official config/export pipeline yet. |
| `python_mock` | Any | Protocol fixture | Used only to verify Worker subprocess handling. Not real training. |

## Environment Setup

Use an isolated Python environment. The local development machine used Python 3.13 embeddable under `.deps`, but a regular venv is preferred for users.

Detection and segmentation:

```powershell
python -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install -r python_trainers\requirements-yolo.txt
```

OCR recognition:

```powershell
python -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install -r python_trainers\requirements-ocr.txt
```

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

## Known Boundaries

- TensorRT engine building is still pending external RTX / SM 75+ acceptance.
- GTX 1060 / SM 61 can run CPU training smoke and ONNX Runtime checks, but it cannot validate TensorRT 10 engine building.
- C++ segmentation mask ONNX postprocess and OCR ONNX CTC decode remain future work.
- Official third-party backend licensing must be reviewed before commercial redistribution.
