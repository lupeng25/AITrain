# Hardware Compatibility Matrix

This matrix records what can be verified on the current development laptop and what still needs an external machine.

| Environment | Supported Today | Not Supported / Pending |
|---|---|---|
| CPU-only Windows | Qt GUI, Worker, SQLite, plugins, C++ scaffold training, Python YOLO CPU smoke, PaddlePaddle OCR CPU smoke, ONNX Runtime detection inference | GPU acceleration and TensorRT engine validation |
| Lenovo Legion Y7000P GTX 1060 / SM 61 | CUDA runtime self-check after driver 582.28, package smoke, ONNX Runtime, CPU training smoke | TensorRT 10 engine build; TensorRT reports SM 61 unsupported |
| RTX / SM 75+ Windows machine | Expected target for TensorRT 10 engine build and runtime smoke | Must still be verified externally |
| Cloud GPU with RTX / SM 75+ | Expected target for TensorRT acceptance if local hardware is unavailable | Requires matching CUDA/TensorRT runtime setup |

## Required External TensorRT Acceptance

Run these commands on an RTX / SM 75+ machine from a packaged build:

```powershell
.\aitrain_worker.exe --self-check
.\aitrain_worker.exe --tensorrt-smoke <work-dir>
.\tools\package-smoke.ps1
```

Acceptance requires:

- Worker self-check reports CUDA/TensorRT runtime availability.
- TensorRT smoke builds an engine from ONNX.
- Engine inference runs without unsupported-SM errors.
- Result is recorded back in `docs/harness/current-status.md`.

## Current Local Hardware Note

The current machine can continue development for Python trainers, ONNX Runtime inference, packaging, and Worker protocol validation. It should not be treated as the final TensorRT acceptance machine.
