# AITrain Studio

AITrain Studio is a C++/Qt desktop foundation for managing local computer-vision training workflows on Windows + NVIDIA GPU workstations.

This repository implements the first usable platform layer from the requested plan:

- Qt Widgets GUI organized as a local training workbench with project dashboard, dataset library, training experiments, task/artifact history, model export, inference validation, plugin, and environment views.
- Chinese/English GUI language switching through Qt translation resources, with language settings persisted in `QSettings` and applied after restart.
- Offline machine-bound license verification before the main window opens, plus a separate Qt license generator tool for issuing signed license codes.
- Isolated `aitrain_worker` process using JSON Lines over `QLocalSocket`.
- SQLite-backed project/task/artifact metadata with task detail queries for artifacts, metrics, exports, and dataset versions.
- Qt plugin interfaces for model, dataset, training, validation, export, and inference extensions.
- Built-in plugin manifests for YOLO-style detection/segmentation, PaddleOCR-style recognition, dataset interop, and the local/offline-first plugin marketplace.
- Dataset validation and split helpers for YOLO detection, YOLO segmentation, PaddleOCR Det, and PaddleOCR Rec label files.
- Worker-backed dataset conversion GUI for the implemented COCO / Pascal VOC / YOLO detection / YOLO segmentation conversion matrix.
- External annotation workflow entrypoint for X-AnyLabeling, with local tool detection and a post-labeling refresh/revalidation path.
- Segmentation dataset admission with dataset loading, polygon-to-mask conversion, overlay preview, and Worker metrics; model training is routed through the official Ultralytics segmentation backend.
- Worker-managed Python trainer adapters for official Ultralytics YOLO detection, official Ultralytics YOLO segmentation, and official PaddleOCR Det/Rec train/export/inference orchestration.
- Delivery-closeout workbench surfaces for sample review, delivery acceptance, customer OCR validation, diagnostics, deployment validation, and model-card/report generation.
- QtTest coverage for JSONL protocol, project repository behavior, detection workflow, and segmentation admission behavior.

Production training is routed through Worker-managed official Python trainer subprocesses: Ultralytics for YOLO detection/segmentation and PaddleOCR official adapters for Det/Rec. The legacy tiny detector, small PaddleOCR CTC trainer, C++ segmentation/OCR training scaffolds, and shipped `python_mock` trainer have been physically removed from the product path. RTX 4090 D TensorRT acceptance has passing evidence for the current validation lane; clean Windows package acceptance and any package-root TensorRT rerun still require returned external evidence before they can be marked passed.

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

## User Guide

End-user operation is documented in `docs/user-guide.md`. It covers registration, projects, dataset preparation, dataset conversion, validation and split, sample review, training, artifacts, evaluation, model export, deployment validation, inference validation, plugin marketplace use, delivery acceptance, and diagnostic bundles through the GUI.

Additional operational and delivery references:

- `docs/dataset-conversion.md`: implemented conversion matrix, GUI workflow, outputs, and limitations.
- `docs/delivery-evidence-index.md`: evidence map for local RC, RTX validation, package, OCR, diagnostics, and deferred external lanes.
- `docs/operations-runbook.md`: installer/operator notes for package layout, dependencies, acceptance commands, and redistribution boundaries.
- `docs/developer-architecture.md`: architecture map and extension rules for future maintainers.

## Localization and Offline Licensing

The main GUI and registration dialog support Chinese and English. The selected language is stored in `QSettings` and takes effect after restarting AITrain Studio. Translation resources are built from `src/app/translations/*.ts` into `.qm` files and copied beside the application under `translations`.

AITrain Studio performs offline license validation before showing the main window. Licenses are signed tokens bound to the local machine code; the app stores accepted license data in `QSettings`, not in the project SQLite database.

Build-time licensing knobs:

- `AITRAIN_LICENSE_PUBLIC_KEY`: base64 public key compiled into the main application.
- `AITRAIN_BUILD_LICENSE_GENERATOR`: builds `AITrainLicenseGenerator.exe` when enabled.
- `AITRAIN_INSTALL_LICENSE_GENERATOR`: installs the generator only when explicitly enabled; it defaults off so customer packages do not accidentally include it.

The generator uses a private key file to issue customer license codes. Keep private keys local and out of customer packages and source control. The tracked repository must contain only `tools/aitrain-license-private-key.example.json`; any real `aitrain-license-private-key.json` must live in a secured operator path outside the repo. If a private key has ever been committed, treat it as leaked: rotate to a newly generated key pair, rebuild with the new `AITRAIN_LICENSE_PUBLIC_KEY`, and handle history purging through a separate security procedure if the repository was pushed or distributed.

## Project Layout

- `src/core`: shared interfaces, plugin contracts, protocol, SQLite repository.
- `src/app`: Qt Widgets desktop app.
- `src/license_generator`: internal Qt license generator tool.
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

Phase 17-49 delivery acceptance is documented in `docs/acceptance-runbook.md`. The current source of truth for phase status and evidence paths is `docs/harness/current-status.md`; the evidence index in `docs/delivery-evidence-index.md` summarizes what is local, what has RTX 4090 D evidence, and what still needs returned external evidence.

Run the local baseline and packaged layout smoke checks with:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

Small generated-data training smoke can be run with:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

For a longer local CPU smoke on deterministic small/medium generated data:

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
```

To require real Ultralytics COCO8 / COCO8-seg materialization instead of generated fallback:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets
```

TensorRT acceptance must be run on an RTX / SM 75+ machine. The RTX 4090 D validation lane already has passing evidence; rerun this only when validating a new package root, machine, driver/runtime set, or reopened external acceptance lane:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT
```

## Python Training Backends

Environment and backend notes are documented in `docs/training-backends.md`.

Production training entry points expose only official backends: Ultralytics for YOLO detection/segmentation and PaddleOCR official adapters for Det/Rec. Legacy diagnostic training implementations have been removed instead of hidden behind user-facing switches. `paddleocr_rec` remains only a dataset format; production OCR Rec training uses `paddleocr_rec_official` or `paddleocr_ppocrv4_rec`.

Minimal sample datasets can be generated with:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

The local CPU smoke profile generates larger deterministic detection, segmentation, and OCR Rec samples:

```powershell
python examples\create-minimal-datasets.py --output .deps\cpu-smoke-data --profile cpu-smoke
```

The generated requests can be passed directly to the Python trainers for smoke validation. These datasets are intentionally tiny and are only meant to verify trainer wiring and artifact creation. The GUI can import, validate, split, and preview the generated detection, segmentation, and OCR Rec layouts.

Hardware support and TensorRT acceptance requirements are documented in `docs/hardware-compatibility.md`.

Official PaddleOCR Rec train/export/inference smoke can be run in an isolated OCR Python environment with:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

## Annotation Tool

The dataset page uses X-AnyLabeling as the default external annotation tool. The GUI detects it from `AITRAIN_XANYLABELING_EXE`, the app directory, `tools\x-anylabeling`, `.deps\annotation-tools\X-AnyLabeling`, or `PATH`.

Recommended exports:

- YOLO detection: YOLO bbox labels.
- YOLO segmentation: YOLO polygon labels.
- COCO / Pascal VOC / YOLO interop: use the dataset conversion GUI or X-AnyLabeling export/conversion, then import or select the converted output explicitly. Conversion outputs are not auto-registered as datasets.
- OCR Rec: AITrain currently trains from `rec_gt.txt` / `rec_gt_train.txt` plus `dict.txt`.

Downloaded annotation binaries should stay under `.deps\` or another local dependency directory unless a separate redistribution review is completed.
