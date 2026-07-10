# Hardware Compatibility Matrix

This matrix records what can be verified on current and historical AITrain Studio validation machines.

| Environment | Supported / Verified | Not Supported / Notes |
|---|---|---|
| CPU-only Windows | Qt GUI, Worker, SQLite, built-in capabilities, Python YOLO CPU smoke through official Ultralytics adapters, PaddlePaddle OCR CPU smoke through official PaddleOCR adapters, C++ ONNX Runtime detection inference | GPU acceleration and TensorRT engine validation |
| RTX 4090 D Windows validation machine | TensorRT 10 engine build/deployment validation smoke, Worker CUDA/cuDNN/TensorRT self-check, ONNX Runtime, SMP GPU realtest, OBB public DOTA matrix, anomaly public MVTec matrix, package and RC validation | Generated `.deps` evidence is local validation output and must not be committed |
| Lenovo Legion Y7000P GTX 1060 / SM 61 | CUDA runtime self-check after driver 582.28, package smoke, ONNX Runtime, CPU training smoke | TensorRT 10 engine build; TensorRT reports SM 61 unsupported and should remain `hardware-blocked` on that hardware |
| Cloud GPU with RTX / SM 75+ | Optional repeat target for TensorRT acceptance if the RTX 4090 D evidence needs independent reproduction | Requires matching CUDA/TensorRT runtime setup |
| CPU with NCNN SDK/runtime | NCNN CPU deployment validation for YOLO detection/segmentation `.param/.bin` artifacts when a sample image and sidecar/config are supplied | Vulkan is a configuration option only, not the default acceptance path |

## NCNN Runtime Acceptance

NCNN deployment validation is CPU-first. Configure `AITRAIN_NCNN_ROOT` at CMake time so `net.h`, `ncnn.lib`/`libncnn.a`, optional `ncnn.dll`, and `onnx2ncnn` can be discovered. Without that SDK/runtime, NCNN validation reports unavailable instead of passing artifact-only.

Current local NCNN SDK/runtime is under `.deps\sdks\ncnn`; older worktrees may still expose the same SDK through legacy `.deps\ncnn` via `.\tools\sync-deps-layout.ps1`. Hyuto YOLOv8 detection ONNX converted through `onnx2ncnn` passed runtime deployment validation, and nihui `ncnn-android-yolov8` preconverted YOLOv8n-seg pnnx/DFL NCNN passed segmentation runtime validation with an explicit sidecar. Hyuto and X-AnyLabeling YOLOv8-seg ONNX conversion attempts still produced unsupported `Shape` layers; those are failed conversion compatibility reports, not hardware failures.

## Current TensorRT Acceptance

RTX 4090 D TensorRT acceptance has passed for the current validation lane.

Evidence archive:

```text
docs/validation/rtx4090-validation-evidence-20260615.json
docs/validation/rtx4090-validation-evidence-20260615.md
```

The OCR GPU rerun environment is intentionally retained and exposed through the canonical `.deps/envs/ocr-gpu` path. On this machine it may be a junction to the preserved historical `.deps/rtx4090-validation/python-ocr-gpu` directory; do not delete that retained environment without first provisioning a replacement.

The passing run requires:

- Worker self-check reports CUDA/TensorRT runtime availability.
- TensorRT smoke builds an engine from ONNX.
- Engine export/deployment validation completes without unsupported-SM errors.
- Result is recorded in `docs/harness/current-status.md`.

## Historical Unsupported Hardware Note

GTX 1060 / SM 61 remains a historical unsupported TensorRT 10 engine-build case and should report `hardware-blocked` on that hardware. That result must not override the RTX 4090 D passing evidence.
