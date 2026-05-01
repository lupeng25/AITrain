# AITrain Studio Training Backends

AITrain Studio keeps training out of the GUI process. Real training is launched by `aitrain_worker` as a Python subprocess and reports newline-delimited JSON events back to the existing Worker protocol.

## Backend Summary

| Backend | Task | Status | Notes |
|---|---|---|---|
| `tiny_linear_detector` | Detection | Scaffold | C++ baseline for protocol, tests, and demos. Not real YOLO. |
| `ultralytics_yolo_detect` | Detection | Real Python backend | Uses Ultralytics YOLO detection training and ONNX export. Review AGPL-3.0 / Enterprise license before redistribution. |
| `ultralytics_yolo_segment` | Segmentation | Real Python backend | Uses Ultralytics YOLO segmentation training and ONNX export. C++ ONNX Runtime can decode boxes, mask coefficients, prototypes, and render mask overlays. |
| `paddleocr_rec` | OCR recognition | Real PaddlePaddle CTC smoke backend | Trains a small PaddlePaddle CTC recognizer on PaddleOCR-style Rec data, exports ONNX, and supports C++ ONNX Runtime CTC greedy decode. Not a full PP-OCRv4 official config/export pipeline yet. |
| `paddleocr_rec_official` / `paddleocr_ppocrv4_rec` | OCR recognition | Official PaddleOCR adapter | Generates a PP-OCRv4 recognition config from AITrain PaddleOCR-style Rec data and can run official PaddleOCR `tools/train.py`, `tools/export_model.py`, and optional `tools/infer/predict_rec.py` when `runOfficial=true` and `paddleOcrRepoPath` or `AITRAIN_PADDLEOCR_REPO` points to a checkout. `prepareOnly=true` validates config generation only. |
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

Optional official PaddleOCR Rec training requires a PaddleOCR source checkout because the installed `paddleocr` package exposes inference pipelines, not the legacy `tools/train.py` training scripts:

```powershell
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git .deps\PaddleOCR
$env:AITRAIN_PADDLEOCR_REPO = (Resolve-Path .deps\PaddleOCR).Path
```

The reproducible local smoke command is:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

That script uses an isolated OCR Python embeddable environment under `.deps\python-3.13.13-ocr-amd64`, checks out a pinned PaddleOCR source ref, installs pinned OCR smoke constraints unless disabled, runs official PP-OCRv4 Rec training for 1 epoch on CPU, exports the official inference model, runs official recognition inference on one generated sample image, and checks the checkpoint, inference config, prediction report, resolved source ref, and metrics report.

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

Dataset validation and split are Worker-backed flows. The GUI can now auto-detect YOLO detection, YOLO segmentation, and PaddleOCR Rec layouts; split outputs are recorded as SQLite dataset versions and task artifacts. PaddleOCR Rec split writes `rec_gt_train.txt`, `rec_gt_val.txt`, `rec_gt_test.txt`, and keeps a compatible `rec_gt.txt`.

## Model Export Formats

The GUI model export page routes conversion through `aitrain_worker`; conversion logic must not run in `MainWindow`.

- `onnx`: copies or creates an ONNX model and writes an AITrain sidecar report.
- `ncnn`: converts an ONNX source or generated tiny-detector ONNX into NCNN `.param` and `.bin` files through the external `onnx2ncnn` tool. Configure it with `AITRAIN_NCNN_ONNX2NCNN` or an NCNN install root in `AITRAIN_NCNN_ROOT`.
- `tensorrt`: remains RTX / SM 75+ external acceptance only on this hardware baseline.

NCNN export creates deployment artifacts only. AITrain Studio does not yet run NCNN inference in the C++ inference page.

## Acceptance Smoke

The unified Phase 17-21 acceptance entry point is:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

This mode checks required Python modules, generates tiny local datasets under `.deps\acceptance-smoke` by default, tries to materialize Ultralytics COCO8 / COCO8-seg through the installed official package, then runs 1-epoch smoke training for YOLO detection, YOLO segmentation, PaddlePaddle OCR Rec CTC, and the isolated official PaddleOCR Rec path when available. During the CTest step, it sets `AITRAIN_ACCEPTANCE_SMOKE_ROOT` so ONNX inference tests consume the artifacts generated in the current WorkDir rather than relying on older local smoke outputs.

Public dataset materialization is handled by `tools\materialize-ultralytics-dataset.py`. It reads the installed Ultralytics dataset yaml, resolves the official download URL, downloads into `.deps\datasets\downloads`, extracts into `.deps\datasets\materialized`, rewrites a local absolute-path `data.yaml`, and writes a machine-readable materialization report. Use `-RequirePublicDatasets` to fail if COCO8 / COCO8-seg cannot be materialized:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets -SkipOfficialOcr
```

Every `acceptance-smoke.ps1` run writes `acceptance_summary.json` into its work directory with modes, status, timing, failure reason, and hardware-blocked reason when applicable.

## Official PaddleOCR Adapter Parameters

The official adapter accepts these extra parameters in addition to the common Python trainer request fields:

- `trainLabelFile`, `valLabelFile`, and `dictionaryFile` to use explicit PaddleOCR Rec materials.
- `officialConfig` to start from a specific PaddleOCR recognition config.
- `pretrainedModel` and `resumeCheckpoint` for official train/export inputs.
- `exportOnly=true` to skip training and export an existing checkpoint.
- `runInferenceAfterExport=true` plus `inferenceImage` to run official `predict_rec.py` after export and write `official_prediction.json`.
- `recImageShape` to override the generated recognition image shape, for example `3,48,320`.

The final `paddleocr_official_rec_report.json` records PaddleOCR requested/resolved refs, Python/Paddle/PaddleOCR versions, train/export/predict commands, config and label paths, dictionary path, checkpoint and inference model paths, parsed metrics, exit codes, and failure log paths.

If public dataset materialization fails or requires external interaction, the generated minimal datasets remain the required smoke baseline. The failure reason should be recorded as an external data acquisition blocker, not hidden as a successful public dataset run.

## Known Boundaries

- TensorRT engine building is still pending external RTX / SM 75+ acceptance.
- GTX 1060 / SM 61 can run CPU training smoke and ONNX Runtime checks, but it cannot validate TensorRT 10 engine building.
- C++ segmentation mask ONNX postprocess and OCR ONNX CTC greedy decode are available for the current smoke models.
- Official third-party backend licensing must be reviewed before commercial redistribution.
- The official PaddleOCR adapter should be run in an isolated OCR Python environment. Mixing PaddlePaddle and PyTorch in one Windows Python process can trigger DLL conflicts through newer `albumentations` builds.
- The official PP-OCRv4 smoke uses a tiny generated dataset; it validates train/export/inference wiring and artifacts, not useful OCR accuracy.
