# AITrain Studio

AITrain Studio is a C++/Qt desktop foundation for managing local computer-vision training workflows on Windows + NVIDIA GPU workstations.

This repository implements the first usable platform layer from the requested plan:

- Qt Widgets GUI organized as a local training workbench with project dashboard, dataset library, training experiments, task/artifact history, model export, inference validation, plugin, and environment views.
- Isolated `aitrain_worker` process using JSON Lines over `QLocalSocket`.
- SQLite-backed project/task/artifact metadata with task detail queries for artifacts, metrics, exports, and dataset versions.
- Qt plugin interfaces for model, dataset, training, validation, export, and inference extensions.
- Built-in plugin manifests for YOLO-style detection/segmentation, PaddleOCR-style recognition, and dataset interop.
- Dataset validation and split helpers for YOLO detection, YOLO segmentation, and PaddleOCR recognition label files.
- Segmentation admission scaffold with dataset loading, polygon-to-mask conversion, overlay preview, Worker metrics, and scaffold checkpoints.
- Worker-managed Python trainer adapters for Ultralytics YOLO detection, Ultralytics YOLO segmentation, PaddlePaddle OCR Rec CPU smoke training, and official PaddleOCR PP-OCRv4 Rec train/export/inference orchestration.
- QtTest coverage for JSONL protocol, project repository behavior, detection workflow, and segmentation admission behavior.

The native C++ training implementation remains an executable workflow scaffold: the worker can run a tiny detector placeholder, produce checkpoints, export a tiny detector ONNX model, validate it through ONNX Runtime, and run tiny segmentation/OCR admission scaffolds. Real model training is now routed through Worker-managed Python trainer subprocesses. Ultralytics YOLO detection and segmentation have CPU smoke coverage, and PaddlePaddle OCR Rec has a small CTC smoke trainer. C++ ONNX Runtime now supports YOLO segmentation mask postprocess and OCR CTC greedy decode for those smoke models. The official PaddleOCR adapter can prepare PP-OCRv4 Rec configs and command files, and can run official PaddleOCR training/export in an isolated OCR Python environment. External TensorRT acceptance remains future work.

## Build

Requirements:

- CMake 3.21+
- C++20-capable MSVC
- Qt 6 preferred, Qt 5.12+ supported for the current platform layer

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=C:\Qt\6.6.3\msvc2019_64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

If only Qt 5 is installed, point `CMAKE_PREFIX_PATH` to the Qt 5 MSVC kit.

## Run

```powershell
.\build\bin\Release\AITrainStudio.exe
```

For single-config generators such as NMake, use:

```powershell
.\build\bin\AITrainStudio.exe
```

Plugins are built under `build\plugins\models`. The GUI scans that directory and the application-local `plugins\models` directory.

## Project Layout

- `src/core`: shared interfaces, plugin contracts, protocol, SQLite repository.
- `src/app`: Qt Widgets desktop app.
- `src/worker`: isolated task runner.
- `src/plugins`: built-in plugin DLLs.
- `tests`: QtTest tests.

## Harness

This repository includes a lightweight engineering harness for AI-assisted development:

- `HARNESS.md`: operating contract and definition of done.
- `docs/harness/project-context.md`: current project facts and source map.
- `docs/harness/implementation-checklist.md`: task workflow.
- `docs/harness/quality-gates.md`: verification gates by subsystem.
- `docs/harness/ui-guidelines.md`: UI rules for the current workbench design.
- `tools/harness-check.ps1`: configure, build, and test in one command.

Run the full check:

```powershell
.\tools\harness-check.ps1
```

Print the project context:

```powershell
.\tools\harness-context.ps1
```

## Acceptance

Phase 17-26 delivery acceptance is documented in `docs/acceptance-runbook.md`.

Run the local baseline and packaged layout smoke checks with:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

Small generated-data training smoke can be run with:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

To require real Ultralytics COCO8 / COCO8-seg materialization instead of generated fallback:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets -SkipOfficialOcr
```

TensorRT acceptance must be run on an RTX / SM 75+ machine:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT
```

## Python Training Backends

Environment and backend notes are documented in `docs/training-backends.md`.

Minimal sample datasets can be generated with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

The generated requests can be passed directly to the Python trainers for smoke validation. These datasets are intentionally tiny and are only meant to verify trainer wiring and artifact creation. The GUI can import, validate, split, and preview the generated detection, segmentation, and OCR Rec layouts.

Hardware support and TensorRT acceptance requirements are documented in `docs/hardware-compatibility.md`.

Official PaddleOCR Rec train/export/inference smoke can be run in an isolated OCR Python environment with:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```
