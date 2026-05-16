# Hardware Compatibility Matrix

This matrix records what can be verified on current and historical AITrain Studio validation machines.

| Environment | Supported / Verified | Not Supported / Notes |
|---|---|---|
| CPU-only Windows | Qt GUI, Worker, SQLite, plugins, C++ scaffold training, Python YOLO CPU smoke, PaddlePaddle OCR CPU smoke, ONNX Runtime detection inference | GPU acceleration and TensorRT engine validation |
| RTX 4090 D Windows validation machine | TensorRT 10 engine build and inference smoke, Worker CUDA/cuDNN/TensorRT self-check, ONNX Runtime, package and RC validation | Generated `.deps` evidence is local validation output and must not be committed |
| Lenovo Legion Y7000P GTX 1060 / SM 61 | CUDA runtime self-check after driver 582.28, package smoke, ONNX Runtime, CPU training smoke | TensorRT 10 engine build; TensorRT reports SM 61 unsupported and should remain `hardware-blocked` on that hardware |
| Cloud GPU with RTX / SM 75+ | Optional repeat target for TensorRT acceptance if the RTX 4090 D evidence needs independent reproduction | Requires matching CUDA/TensorRT runtime setup |
| CPU with NCNN SDK/runtime | NCNN CPU deployment validation for YOLO detection/segmentation `.param/.bin` artifacts when a sample image and sidecar/config are supplied | Vulkan is a configuration option only, not the default acceptance path |

## NCNN Runtime Acceptance

NCNN deployment validation is CPU-first. Configure `AITRAIN_NCNN_ROOT` at CMake time so `net.h`, `ncnn.lib`/`libncnn.a`, optional `ncnn.dll`, and `onnx2ncnn` can be discovered. Without that SDK/runtime, NCNN validation reports unavailable instead of passing artifact-only.

Local refresh on 2026-05-16 used `.deps\ncnn` as the NCNN SDK/runtime root. Hyuto YOLOv8 detection ONNX converted through `onnx2ncnn` passed runtime deployment validation, and nihui `ncnn-android-yolov8` preconverted YOLOv8n-seg pnnx/DFL NCNN passed segmentation runtime validation with an explicit sidecar. Hyuto and X-AnyLabeling YOLOv8-seg ONNX conversion attempts still produced unsupported `Shape` layers; those are failed conversion compatibility reports, not hardware failures.

## Current TensorRT Acceptance

RTX 4090 D TensorRT acceptance has passed for the current validation lane.

Evidence:

```text
.deps/rtx4090-validation/acceptance-tensorrt/acceptance_summary.json
.deps/rtx4090-validation/validation-index.md
```

The passing run requires:

- Worker self-check reports CUDA/TensorRT runtime availability.
- TensorRT smoke builds an engine from ONNX.
- Engine inference runs without unsupported-SM errors.
- Result is recorded in `docs/harness/current-status.md`.

## Historical Unsupported Hardware Note

GTX 1060 / SM 61 remains a historical unsupported TensorRT 10 engine-build case and should report `hardware-blocked` on that hardware. That result must not override the RTX 4090 D passing evidence.
