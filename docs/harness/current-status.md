# Current Project Status

Last updated: 2026-05-01

This file is the source of truth for phase status in new AI coding conversations. Read it before using `AITrainStudio_后续实施方案.md`, because that document is the long-range roadmap and may contain historical phase descriptions.

## Phase Status

| Phase | Status | Current interpretation |
|---|---|---|
| Phase 1: Platform stabilization | Done as platform scaffold | Task state transitions, Worker commands, SQLite metadata, artifacts/exports/environment records, and harness checks are in place. |
| Phase 2: Dataset system | Done as initial system | YOLO detection, YOLO segmentation, and PaddleOCR Rec validation exist. Dataset split currently covers YOLO detection. |
| Phase 3: YOLO detection training | Scaffold done | Tiny linear detector can train on small data, report loss/mAP50-style metrics, save checkpoint, resume, and generate preview. This is not real LibTorch/CUDA YOLO. |
| Phase 4: ONNX export and inference | Tiny detector path done | Tiny detector ONNX export, ONNX Runtime inference, checkpoint/ONNX consistency, Worker inference, prediction JSON, and overlay output are covered by tests. Full YOLO/OCR postprocess and TensorRT are not done. |
| Phase 5: YOLO segmentation training | Scaffold done | `SegmentationDataset`, `SegmentationDataLoader`, polygon-to-mask, letterbox-aligned masks, multi-polygon/multi-class masks, overlay preview, mask preview artifact, scaffold `maskIoU`/`segmentationMap50`, Worker `taskType=segmentation`, checkpoint, and tests are in place. Real mask head, real mask loss, and CUDA training are not done. |
| Phase 6: OCR recognition training | Scaffold done | `OcrRecDataset`, dictionary loading, label encode/decode, resize/pad batching, `OcrRecTrainer` scaffold, Worker `taskType=ocr_recognition`, `ctcLoss`/`accuracy`/`editDistance`, checkpoint, preview artifact, and tests are in place. Real CRNN/CTC loss and CUDA training are not done. |
| Phase 7: TensorRT and packaging | Code complete; local TensorRT smoke hardware-blocked | Install layout, package smoke, ZIP packaging target, packaged Worker/plugin smoke, runtime DLL discovery, CUDA runtime initialization self-check, downloaded local CUDA/cuDNN/TensorRT redistributables, and SDK-backed TensorRT ONNX->engine export/inference code are in place. After updating the local NVIDIA driver to 582.28, CUDA 13 runtime self-check passes, but TensorRT 10 engine build is still blocked because the local GTX 1060 is SM 61 and this TensorRT release requires a newer target GPU. |
| Phase 8: Python Trainer Adapter and environment management | Done | Worker can launch a Python trainer subprocess, pass a JSON request, read JSON Lines log/progress/metric/artifact/completed/failed messages, propagate failures, and report Python/Ultralytics/PaddleOCR/PaddlePaddle availability in environment checks. The shipped `python_mock` trainer is a scaffold protocol fixture, not real model training. |
| Phase 9: Official YOLO detection training integration | Done on local CPU smoke | Worker routes `trainingBackend=ultralytics_yolo_detect` / `ultralytics_yolo` to `python_trainers/detection/ultralytics_trainer.py`. The adapter normalizes YOLO data yaml, calls official `ultralytics.YOLO(...).train()`, exports ONNX, forwards metrics/artifacts, and has deterministic Worker coverage with a fake official API package. Local `.deps` Python has Ultralytics 8.4.45, Torch 2.11.0 CPU, ONNX 1.21.0, and ONNX Runtime 1.25.1; a 1-epoch CPU smoke with `yolov8n.yaml` produced `best.pt`, `best.onnx`, and `ultralytics_training_report.json`. |
| Phase 10: Real detection ONNX inference and conversion | Done for ONNX Runtime; TensorRT external GPU pending | C++ ONNX Runtime can now run Ultralytics YOLO detection ONNX models with letterbox preprocessing, YOLOv8-style output decode, confidence filtering, NMS, class mapping from the Phase 9 data yaml/report, prediction JSON, and overlay rendering. `exportDetectionCheckpoint(..., format=onnx)` can also copy a real YOLO ONNX and write an AITrain sidecar; TensorRT conversion accepts ONNX sources but remains hardware-blocked on this GTX 1060 / SM 61 machine. |
| Phase 11: Official YOLO segmentation training integration | Done on local CPU smoke | Worker routes `trainingBackend=ultralytics_yolo_segment` to `python_trainers/segmentation/ultralytics_trainer.py`, which reuses the official Ultralytics adapter with segmentation defaults. A 1-epoch CPU smoke with a minimal polygon dataset and `yolov8n-seg.yaml` produced `best.pt`, `best.onnx`, `ultralytics_training_report.json`, and mask metrics (`maskPrecision`, `maskRecall`, `maskMap50`, `maskMap50_95`). C++ segmentation ONNX mask postprocess is still a future enhancement; current C++ mask training remains scaffold. |
| Phase 12: PaddlePaddle OCR Rec training integration | Done on local CPU smoke | Worker routes `trainingBackend=paddleocr_rec` to `python_trainers/ocr_rec/paddleocr_trainer.py`. Local `.deps` Python has PaddlePaddle 3.3.1 and PaddleOCR 3.5.0 installed. A minimal PaddleOCR-style Rec dataset trained for 1 epoch on CPU and produced `paddleocr_rec_ctc.pdparams`, `dict.txt`, `paddleocr_rec_training_report.json`, and real `loss`/`accuracy`/`editDistance` metrics. This is a small PaddlePaddle CTC Rec trainer compatible with PaddleOCR-style data, not a full PP-OCRv4 official config/export pipeline yet. |
| Phase 13: Productization and acceptance | Local package/docs/examples complete; external acceptance pending | Added Python requirements, training backend docs, hardware compatibility docs, minimal dataset generator, package layout checks for docs/examples/requirements, and README/protocol updates. Generated Phase 13 examples ran direct CPU trainer smoke for YOLO detection, YOLO segmentation, and PaddlePaddle OCR Rec. `harness-check.ps1` and `package-smoke.ps1 -SkipBuild` pass locally. Clean Windows machine validation and RTX / SM 75+ TensorRT acceptance remain external tasks. |

## Local Hardware Note

Recorded on 2026-04-30:

- Machine: Lenovo Legion Y7000P-1060 / model 81LE.
- CPU: Intel Core i5-8300H, 4 cores / 8 threads.
- Memory: 16 GB DDR4-2666 installed as 2x8 GB; firmware reports 2 slots and 32 GB maximum.
- Storage: 512 GB NVMe SSD.
- GPU: NVIDIA GeForce GTX 1060 6 GB. Driver was upgraded from 398.27 to 582.28 on 2026-04-30.
- Phase 7 impact: this machine should not be treated as the final TensorRT acceptance machine. The driver update makes CUDA 13 runtime initialization pass, but TensorRT 10 smoke fails with `Target GPU SM 61 is not supported by this TensorRT release`.
- Practical upgrade path: optionally upgrade memory to 32 GB and storage to 1-2 TB. Do not plan around replacing the laptop GPU. Use a CUDA 13/TensorRT-compatible RTX machine or cloud GPU for real Phase 7 TensorRT verification.
- Python note: Python 3.13.13 embeddable x64 was downloaded under `.deps/python-3.13.13-embed-amd64` for Phase 8 adapter verification. `.deps/` is ignored and should not be committed.

## Current Next Task

Next external acceptance task: validate the packaged build on a clean Windows machine and run TensorRT smoke on an RTX / SM 75+ GPU:

- keep Phase 7 marked as code complete but hardware-blocked on this GTX 1060 / SM 61 machine
- Phase 8 is complete as an adapter/protocol layer; `python_mock` remains a scaffold protocol fixture
- Phase 9 is complete on this machine for a CPU smoke using the installed official Ultralytics package; keep license constraints visible before redistribution
- Phase 10 ONNX Runtime inference is complete for real YOLO detection ONNX; TensorRT engine acceptance still needs RTX / SM 75+
- Phase 11 official YOLO segmentation training is complete on local CPU smoke
- Phase 12 PaddlePaddle OCR Rec training is complete on local CPU smoke; full PP-OCRv4 config/export is still future work
- Phase 13 local productization is complete: docs, generated sample datasets, Python dependency files, hardware compatibility notes, package smoke verification
- keep training logic inside core/plugin/Worker boundaries, not in `MainWindow`
- keep C++ tiny detector as a scaffold/demo/test backend until the Python path is stable
- preserve the external TensorRT acceptance checklist for a future RTX / SM 75+ machine

After each small step, run:

```powershell
.\tools\harness-check.ps1
```

For packaging smoke checks, run:

```powershell
.\tools\package-smoke.ps1
```

For ZIP package generation, run:

```powershell
cmake --build build-vscode --target package
```

Minimal example datasets can be generated with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

When an RTX / SM 75+ machine is available, resume Phase 7 acceptance with:

```powershell
.\aitrain_worker.exe --self-check
.\aitrain_worker.exe --tensorrt-smoke <work-dir>
```

## Non-Negotiable Notes

- Do not describe the C++ tiny detector, segmentation baseline, or OCR baseline as real YOLO/OCR training.
- The `ultralytics_yolo_detect` backend is the first official YOLO detection training integration, but it requires an installed official Python package and license review before redistribution.
- Current detection and segmentation training are scaffold/baseline workflows.
- Real training should now be implemented through Worker-managed Python trainer subprocesses; do not embed Python inside the GUI process.
- Phase 5 is complete only as a segmentation scaffold/baseline loop; it is not real YOLO segmentation training.
- Phase 6 is complete only as an OCR recognition scaffold/baseline loop; it is not real CRNN/CTC OCR training.
- Phase 11 adds official YOLO segmentation training/export, but C++ segmentation ONNX mask postprocess is not complete yet.
- Phase 12 adds a real PaddlePaddle CTC OCR Rec trainer for PaddleOCR-style data, but it is not a full PaddleOCR PP-OCRv4 official training/export pipeline yet.
- Phase 13 local productization is complete, but clean-machine and RTX TensorRT acceptance still require external hardware/environment.
- Keep long-running execution in `aitrain_worker`.
- Keep model-specific behavior behind core/plugin/Worker boundaries, not in `MainWindow`.
