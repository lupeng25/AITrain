# AITrain Studio Local RC Closeout

This checklist is the local, non-external release-candidate closeout path after Phase 39B, Phase 39C, Phase 41 Lite, Phase 49 Lite, and the current industrial-vision expansion lane. It does not add new model backends by itself. It exists to make the current detection, segmentation, OBB, semantic segmentation, anomaly, OCR, pipeline, benchmark, model registry, delivery report, environment profile, sample review, deployment validation, diagnostics, and environment-page delivery evidence loop repeatable on the development machine.

After this local gate passes, use `docs\external-acceptance-handoff.md` and `docs\acceptance-templates\` for the clean Windows package handoff and for any explicitly reopened package-root RTX / SM 75+ TensorRT rerun. The RTX 4090 D source-side TensorRT validation lane already has passing evidence archived in `docs\validation\rtx4090-validation-evidence-20260615.json`; do not weaken or overwrite that lane with package-root or clean-machine status.

## Scope

Local closeout covers:

- Source build and CTest through the harness.
- Packaged layout smoke from the source tree.
- Worker self-check, built-in capability check, and package documentation/script presence.
- Optional local baseline acceptance and CPU training smoke.
- Automated 1280x820 GUI walkthrough for the current workbench; use `-SkipGuiWalkthrough` only in intentionally headless environments.
- Documentation language check for scaffold, official backend, TensorRT hardware-blocked, NCNN runtime/SDK requirements, and customer-domain OCR boundaries.

Out of scope:

- Clean Windows machine acceptance.
- Clean Windows package-root TensorRT rerun or independent external TensorRT refresh. The existing RTX 4090 D validation evidence remains the recorded passing TensorRT lane unless a new package-root rerun is explicitly requested and returned.
- New classification, pose, YOLO-World, YOLOE, tracking, video/time-series, 3D/RGB-D, cloud, or multi-user training/product directions.
- Embedding Python training or annotation tools into the GUI process.

## Command Plan

Fast local RC closeout:

```powershell
.\tools\local-rc-closeout.ps1
```

This runs:

```powershell
git diff --check
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
.\tools\ui-workbench-walkthrough.ps1
```

Fuller local closeout, still without external hardware:

```powershell
.\tools\local-rc-closeout.ps1 -RunLocalBaseline -RunCpuTrainingSmoke
```

This additionally runs:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
```

The CPU training smoke is intentionally heavier. It validates integration and artifacts through official production backends, not production model accuracy.

## GUI Walkthrough

The default RC command runs the fixed 1280x820 walkthrough wrapper:

```powershell
.\tools\ui-workbench-walkthrough.ps1
```

该检查覆盖 `总览`、`项目`、`数据集`、`训练实验`、`任务与产物`、`模型库`、`部署验证`、`环境` 与 `系统设置`，并写入 `.deps\UI-Walkthrough\rc\ui_walkthrough_rc_summary.json`。`数据集 > 质量与复核`、`模型库 > 评估报告`、`部署验证 > 部署验证 / 推理验证`、`系统设置 > 内置能力 / 应用设置` 与 `环境 > 交付证据` 等页签由 QtTest 覆盖。

If the app opens the offline registration dialog before the workbench, the wrapper writes a blocked summary with `errorCode=license_required`. If the optional `qt-gui-walkthrough` dependency is not installed, it writes `errorCode=walkthrough_script_missing`. Treat either result as environment/setup blocked evidence: install the walkthrough dependency and configure a valid offline license token plus build-time `AITRAIN_LICENSE_PUBLIC_KEY`, then rerun the walkthrough instead of marking the GUI gate passed.

For manual exploration beyond the automated gate, walk through these screens:

| Area | Check |
|---|---|
| Project | Create or open a project; dashboard should show project, task, dataset, model, built-in capability, and environment summaries. |
| Environment | Run environment check; YOLO, OCR, and TensorRT profile rows should appear with repair hints. GTX 1060 / SM 61 TensorRT must read as `hardware-blocked` / hardware limited, not passed. |
| Dataset | Import generated YOLO detection, YOLO segmentation, PaddleOCR Rec, and PaddleOCR Det datasets; auto-detection and validation should be visible. |
| Dataset > Quality Review | Load problem/error/rework sample JSON when available; filters and X-AnyLabeling review-list export should be visible. |
| Annotation | X-AnyLabeling remains an external tool; launch/detect actions should not block the GUI or imply embedded annotation. |
| Training | Official/upstream YOLO/OBB, SMP, Anomalib, and OCR backends should be the only product training choices; removed diagnostic/scaffold backends must not reappear in the GUI. |
| Task Artifacts | Select recent tasks and preview JSON/TXT/CSV/image/ONNX/model artifacts; unsupported artifacts should show a clear message. |
| Model Library | Registered model versions, evaluation reports, comparison rows, pipeline records, lineage, benchmarks, artifacts, and limitation summaries should be visible. |
| 部署验证 | `部署验证` 与 `推理验证` 页签都应只允许选择已验证 V2 模型包，不显示裸路径模型导出入口。 |
| System Settings | Built-in capability matrix and application settings should be visible under `内置能力` and `应用设置`. |
| Delivery Report | Generate a delivery report and confirm HTML, model card, and artifact inventory are present and previewable. |
| Delivery Evidence | Open `环境 > 交付证据`; local RC, clean Windows, TensorRT, customer OCR, package integrity, diagnostics, and deployment validation states should render as `passed`, `blocked`, `failed`, `hardware-blocked`, or `not-run` without horizontal overflow. |

## Boundary Wording Checklist

Before marking the local RC closeout done, check docs and UI text for:

- Tiny detector, segmentation baseline, OCR baseline, small OCR CTC, and shipped `python_mock` trainer implementations are removed from the product training path.
- Ultralytics YOLO official backends require installed official Python packages and license review before redistribution.
- `paddleocr_rec` is a dataset format only; PaddleOCR Rec training must use the official PaddleOCR adapter.
- PaddleOCR System is official `predict_system.py` tool orchestration, not C++ DB ONNX postprocess.
- TensorRT on GTX 1060 / SM 61 is `hardware-blocked`; RTX / SM 75+ is still required for real TensorRT acceptance. The RTX 4090 D validation lane already passed, while clean Windows package-root reruns remain separate evidence.
- Customer-domain OCR production readiness requires customer/target-domain data; Total-Text, generated smoke, and `.deps` examples are workflow smoke only.
- NCNN deployment validation runs runtime inference for supported YOLO detection/segmentation artifacts when NCNN SDK/runtime and a sample image are available; otherwise it reports failed/blocked instead of artifact-only passed. Current local evidence covers Hyuto YOLOv8 detection ONNX -> NCNN and nihui preconverted YOLOv8n-seg pnnx/DFL NCNN; YOLOv8-seg ONNX conversion with unsupported `Shape` layers is failed conversion evidence.
- Classification, pose, YOLO-World, YOLOE, tracking, video/time-series, 3D/RGB-D, cloud, and multi-user product directions remain out of scope until priorities are reset. OBB v1, SMP semantic segmentation, and Anomalib PatchCore/EfficientAD anomaly v1 are current local industrial-vision capabilities, with customer-domain readiness still requiring target-domain evidence.

## Completion Record

After closeout passes locally, update `docs/harness/current-status.md` with:

- Phase 42 Lite local RC closeout status.
- Commands run and whether CPU training smoke was included.
- Any skipped heavy smoke and why.
- External clean Windows package acceptance remaining pending, and any package-root TensorRT rerun marked separately from the already-passed RTX 4090 D validation lane.

The next handoff step, when explicitly reopened, is to send the package plus `docs\external-acceptance-handoff.md` and collect filled templates from the external clean Windows machine and any requested package-root TensorRT machine.
