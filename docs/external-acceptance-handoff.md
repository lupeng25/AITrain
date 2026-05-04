# AITrain Studio External Acceptance Handoff

This handoff package is for Phase 43 Lite external acceptance. It prepares the clean Windows package check and the RTX / SM 75+ TensorRT check without changing training, inference, Worker, or GUI behavior.

## Scope

External acceptance covers:

- Clean Windows package layout validation from an installed package root.
- Worker self-check and plugin loading from the package.
- Optional source-tree local RC repeatability check before handoff.
- TensorRT engine build and inference smoke on supported RTX / SM 75+ hardware.
- Return of machine-readable summaries and enough environment evidence to update `docs\harness\current-status.md`.

Out of scope:

- Accuracy benchmarking for production use.
- New training backends.
- Redistribution review for downloaded third-party packages, datasets, annotation tools, CUDA, cuDNN, TensorRT, or model weights.
- Treating GTX 1060 / SM 61 TensorRT results as passed.

## Required Preconditions

Clean Windows package acceptance:

- Windows x64 machine separate from the primary development tree.
- Unpacked AITrain Studio package root containing `AITrainStudio.exe`, `aitrain_worker.exe`, `plugins`, `docs`, `examples`, `python_trainers`, and `tools`.
- PowerShell available.
- MSVC/Qt build tools are not required when testing from an already-installed package root.
- Optional Python runtimes may be missing; package acceptance should report optional runtime gaps clearly rather than fail for every optional backend.

TensorRT acceptance:

- Windows x64 machine with NVIDIA RTX or other TensorRT 10 supported GPU.
- GPU compute capability must be SM 75 or newer unless the TensorRT release explicitly supports a different target.
- Current NVIDIA driver compatible with the packaged CUDA/TensorRT runtime.
- CUDA runtime, cuDNN, TensorRT, TensorRT plugin, TensorRT ONNX parser, and ONNX Runtime discoverable by the Worker self-check.
- Package root or source tree with `tools\acceptance-smoke.ps1`.
- Work directory with enough free space for generated ONNX/engine/input/output smoke artifacts.

Do not use the local GTX 1060 / SM 61 laptop as a passing TensorRT machine. Its expected TensorRT result is `hardware-blocked`.

## Source-Side Pre-Handoff Check

From the repository root, run the repeatable local RC gate before sending a package for external acceptance:

```powershell
.\tools\local-rc-closeout.ps1
```

For a traceable release-freeze package identity, run:

```powershell
.\tools\release-freeze-handoff.ps1
```

This generates the CPack ZIP, SHA256 hashes, and `build-vscode\release-freeze-handoff\release_handoff_manifest.json`.

For a fuller local preflight that also exercises the local baseline and package layout without rebuilding:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
```

For the local CPU wiring smoke:

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke -SkipOfficialOcr
```

These checks do not replace clean Windows or RTX TensorRT acceptance.

## Clean Windows Package Acceptance

From the unpacked package root:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

Expected result:

- Package layout exists.
- Worker self-check emits valid JSON.
- Built-in plugins load through Worker plugin smoke.
- Missing optional runtimes are reported as environment gaps, not as silent success.
- `acceptance_summary.json` is written under the selected work directory.

Use `docs\acceptance-templates\clean-windows-acceptance-result.md` to record the result.

## TensorRT Acceptance

From a supported RTX / SM 75+ package root or source tree:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

Acceptance requires:

- Worker self-check resolves the CUDA/TensorRT runtime components needed for TensorRT inference.
- GPU compute capability is accepted by the TensorRT release.
- `aitrain_worker.exe --tensorrt-smoke <work-dir>` builds an engine and runs inference.
- The command finishes with `passed`, not `hardware-blocked`.

Use `docs\acceptance-templates\tensorrt-acceptance-result.md` to record the result.

## Artifacts To Return

For every external acceptance run, return:

- Filled result template.
- `acceptance_summary.json`.
- Full console output.
- Package root file layout summary.
- Worker self-check JSON.
- Worker plugin smoke JSON when package mode is run.
- GPU and driver information for TensorRT mode.
- TensorRT smoke output or exact failure / hardware-blocked message.
- ZIP/package name, hash, and source commit when available.

Suggested commands for environment evidence:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsHardwareAbstractionLayer, CsProcessors, CsTotalPhysicalMemory
nvidia-smi
Get-ChildItem -Name
```

## Artifact Boundaries

The following are generated or external dependency artifacts and must not be treated as source-controlled or default packaged source artifacts:

- `.deps\`
- Generated datasets.
- Downloaded public datasets.
- Downloaded Python embeddable runtimes.
- Downloaded CUDA, cuDNN, TensorRT, or ONNX Runtime SDK folders.
- X-AnyLabeling or other annotation tool downloads.
- Model weights, checkpoints, TensorRT engines, ONNX outputs from smoke runs, and trainer output directories.

Only update `docs\harness\current-status.md` after external evidence is returned. Record clean Windows package acceptance and TensorRT acceptance separately.
