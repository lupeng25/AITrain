# `.deps` Layout

`.deps` is the local, untracked dependency and evidence root. Reusable environments should use the canonical folders below; validation outputs and historical evidence should stay in their own run folders.

## Canonical Reusable Dependencies

| Purpose | Canonical path |
|---|---|
| Python environments | `.deps\envs` |
| General embedded Python | `.deps\envs\python-embed-3.13.13` |
| YOLO CUDA Python | `.deps\envs\yolo-cuda` |
| YOLO26 isolated Python | `.deps\envs\yolo26` |
| PaddleOCR CPU/portable Python | `.deps\envs\ocr-cpu` |
| PaddleOCR GPU Python | `.deps\envs\ocr-gpu` |
| Paddle2ONNX compatibility Python | `.deps\envs\paddle2onnx` |
| Source checkouts | `.deps\repos` |
| PaddleOCR source checkout | `.deps\repos\PaddleOCR` |
| SDK/runtime dependencies | `.deps\sdks` |
| ONNX Runtime SDK/runtime | `.deps\sdks\onnxruntime` |
| NCNN SDK/runtime | `.deps\sdks\ncnn` |
| TensorRT headers/source support | `.deps\sdks\tensorrt-oss` |
| TensorRT runtime DLLs | `.deps\sdks\tensorrt-runtime` |
| Download archives/bootstrap files | `.deps\archives` |
| External tools | `.deps\tools` |
| X-AnyLabeling | `.deps\tools\annotation-tools\X-AnyLabeling` |
| UI walkthrough evidence root | `.deps\UI-Walkthrough` |

## Run And Evidence Outputs

Use separate folders for generated outputs, for example:

- `.deps\acceptance-smoke`
- `.deps\full-model-lifecycle`
- `.deps\phase-yolo26-model-matrix`
- `.deps\production-ocr-data`
- `.deps\production-ocr-official-chain`
- `.deps\rtx4090-validation`
- `.deps\UI-Walkthrough\rc`

These folders are evidence/run outputs, not the canonical home for reusable Python environments or SDKs.

## Compatibility

Older local machines may still have dependencies under paths such as `.deps\PaddleOCR`, `.deps\ncnn`, `.deps\python-3.13.13-ocr-amd64`, or `.deps\rtx4090-validation\python-ocr-gpu`. Scripts now prefer the canonical paths and keep the old paths only as fallback.

Run this after cleanup or after importing an older `.deps` tree:

```powershell
.\tools\sync-deps-layout.ps1
```

The script creates the canonical top-level directories and, where safe, junctions from canonical paths to existing legacy dependencies. It does not delete historical evidence or move the retained OCR GPU environment.
