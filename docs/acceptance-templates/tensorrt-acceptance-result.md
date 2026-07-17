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

## Runtime Delivery workflow

填写本次 Runtime Delivery workflow 的 ModelPackageId、TaskId、EvidenceId，以及返回的能力分类报告。当前没有独立 TensorRT smoke CLI；裸 ONNX 路径不能作为产品推理通过证据。

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
- Official ONNX path:
- Generated engine path, if any:
- Generated export report path, if any:

## Acceptance Decision

- TensorRT engine build passed:
- TensorRT engine export/deployment validation passed:
- Result can update Phase 7 / Phase 10 status:
- Status wording to add to `docs\harness\current-status.md`:

## Notes And Follow-Ups

- Driver/runtime mismatch:
- Missing DLL/runtime:
- Unsupported GPU or compute capability:
- Follow-up owner:
- Follow-up due date:
