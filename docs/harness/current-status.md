# Current Project Status

Last updated: 2026-04-30

This file is the source of truth for phase status in new AI coding conversations. Read it before using `AITrainStudio_后续实施方案.md`, because that document is the long-range roadmap and may contain historical phase descriptions.

## Phase Status

| Phase | Status | Current interpretation |
|---|---|---|
| Phase 1: Platform stabilization | Done as platform scaffold | Task state transitions, Worker commands, SQLite metadata, artifacts/exports/environment records, and harness checks are in place. |
| Phase 2: Dataset system | Done as initial system | YOLO detection, YOLO segmentation, and PaddleOCR Rec validation exist. Dataset split currently covers YOLO detection. |
| Phase 3: YOLO detection training | Scaffold done | Tiny linear detector can train on small data, report loss/mAP50-style metrics, save checkpoint, resume, and generate preview. This is not real LibTorch/CUDA YOLO. |
| Phase 4: ONNX export and inference | Tiny detector path done | Tiny detector ONNX export, ONNX Runtime inference, checkpoint/ONNX consistency, Worker inference, prediction JSON, and overlay output are covered by tests. Full YOLO/OCR postprocess and TensorRT are not done. |
| Phase 5: YOLO segmentation training | Admission scaffold started | `SegmentationDataset`, polygon-to-mask, overlay preview, `SegmentationTrainer` scaffold, Worker `taskType=segmentation`, `maskLoss`/`maskCoverage`, checkpoint, preview artifact, and tests are in place. Real mask head, real mask loss, segmentation mAP, and CUDA training are not done. |
| Phase 6: OCR recognition training | Not started | PaddleOCR Rec validation exists, but CRNN/CTC training is not implemented. |
| Phase 7: TensorRT and packaging | Not started | TensorRT export/inference and Windows packaging remain future work. |

## Current Next Task

Continue Phase 5 by implementing `SegmentationDataLoader`:

- batch images from `SegmentationDataset`
- batch masks aligned with the configured `imageSize`
- multi-polygon and multi-class mask handling
- letterbox/resize mapping for polygons and masks
- QtTest coverage for mask alignment, batch size greater than 1, and invalid polygons

After each small step, run:

```powershell
.\tools\harness-check.ps1
```

## Non-Negotiable Notes

- Do not describe current training as real YOLO/OCR training.
- Current detection and segmentation training are scaffold/baseline workflows.
- Keep long-running execution in `aitrain_worker`.
- Keep model-specific behavior behind core/plugin/Worker boundaries, not in `MainWindow`.
