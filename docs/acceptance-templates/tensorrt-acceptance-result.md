# TensorRT Acceptance Result

## Run Identity

- Date/time:
- Tester:
- Organization/team:
- Package or source path:
- Package hash:
- Source commit:
- Work directory:

## Machine And GPU

- Windows edition/version:
- CPU:
- RAM:
- GPU model:
- GPU compute capability:
- SM 75+ accepted: yes / no
- NVIDIA driver:
- CUDA runtime version:
- cuDNN version:
- TensorRT version:
- ONNX Runtime version:
- PowerShell version:

## Command

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

## Result

- Status: pass / fail / hardware-blocked
- Start time:
- End time:
- Exit code:
- `acceptance_summary.json` path:
- Summary status:
- Hardware-blocked reason, if any:
- Failure reason, if any:

## Evidence Attached

- Full console output:
- `acceptance_summary.json`:
- Worker self-check JSON:
- `nvidia-smi` output:
- TensorRT smoke output:
- Generated engine path, if any:
- Generated prediction/output path, if any:

## Acceptance Decision

- TensorRT engine build passed:
- TensorRT inference passed:
- Result can update Phase 7 / Phase 10 status:
- Status wording to add to `docs\harness\current-status.md`:

## Notes And Follow-Ups

- Driver/runtime mismatch:
- Missing DLL/runtime:
- Unsupported GPU or compute capability:
- Follow-up owner:
- Follow-up due date:
