# AITrain Studio

AITrain Studio is a C++/Qt desktop foundation for managing local computer-vision training workflows on Windows + NVIDIA GPU workstations.

This repository implements the first usable platform layer from the requested plan:

- Qt Widgets GUI with project, dataset, training, conversion, inference, and plugin views.
- Isolated `aitrain_worker` process using JSON Lines over `QLocalSocket`.
- SQLite-backed project/task/artifact metadata.
- Qt plugin interfaces for model, dataset, training, validation, export, and inference extensions.
- Built-in plugin manifests for YOLO-style detection/segmentation, PaddleOCR-style recognition, and dataset interop.
- Dataset validation helpers for YOLO txt and PaddleOCR recognition label files.
- QtTest coverage for JSONL protocol and project repository behavior.

The current training implementation is an executable workflow scaffold: the worker simulates training metrics, checkpoints, and export artifacts so the GUI, protocol, task lifecycle, and plugin architecture can be exercised end to end. Full LibTorch/CUDA YOLO and OCR training kernels are intentionally left behind the plugin interfaces instead of being represented as finished model training code.

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
