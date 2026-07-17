# AITrain Studio Acceptance Runbook

本手册中的 Phase 记录保留历史验收证据。2026-07-16 破坏性重构后，旧 YOLO26/SMP/OBB/NCNN smoke 脚本、X-AnyLabeling 独立脚本以及 Worker 的裸路径和 `--*-smoke` CLI 均已删除，不得按历史命令执行。当前验收入口是 `tools\\harness-check.ps1`、`tools\\acceptance-smoke.ps1` 的 LocalBaseline/Package/PublicDatasets/CpuTrainingSmoke 模式，以及各  Workflow 的 QtTest；TensorRT、NCNN 和 OCR 只通过其 /官方适配器边界报告。

## Acceptance Modes

Run the unified smoke script from the repository root:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
.\tools\acceptance-smoke.ps1 -PublicDatasets
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\acceptance-smoke.ps1
.\tools\phase-ppocrv6-model-matrix-smoke.ps1
.\tools\phase-smp-semantic-segmentation-smoke.ps1
.\tools\phase-anomaly-anomalib-smoke.ps1
.\tools\customer-ocr-validation.ps1
```

The same script is also installed into packaged builds under `tools\acceptance-smoke.ps1`. From a package root, run:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

All generated datasets, official downloads, trainer outputs, and smoke artifacts must stay under `.deps\` or another explicitly supplied work directory. Do not add them to source control.

Every `acceptance-smoke.ps1` run writes `acceptance_summary.json` to its work directory. The summary records requested modes, status, work directory, start/end timestamps, failure reason, and hardware-blocked reason when applicable.

For the current local release-candidate closeout path, use `docs\local-rc-closeout.md`. It is the Phase 42 Lite entry point for source build, package smoke, optional local baseline, optional CPU training smoke, Phase 41 environment profile GUI walkthrough, and boundary wording checks.

For external handoff, use `docs\external-acceptance-handoff.md` and the result templates under `docs\acceptance-templates`. These files define the package-root commands, TensorRT commands, required returned evidence, and the clean separation between local RC closeout and external acceptance.

For release-freeze package identity, use `docs\release-freeze-handoff.md` and `tools\release-freeze-handoff.ps1`. This generates the CPack ZIP, SHA256 hashes, and a handoff manifest without marking external acceptance as passed.

For YOLO model-family productization, use `docs\yolo-model-support-matrix.md`. Phase 45 validates newer Ultralytics detection/segmentation nano model names only. P1 validates YOLOv5u standard P5 detection, YOLOv8 / YOLO11 / YOLO12 detection, YOLOv8 / YOLO11 instance segmentation, YOLO12 instance-segmentation `.yaml`, `.yaml` and `.pt` source types where the official asset resolves, YOLOv8 P2/P6 detection YAML architectures, and the official export-argument protocol. YOLO12 segmentation `.pt` rows are blocked in the recorded Ultralytics 8.3.171 environment because `yolo12n-seg.pt` cannot be resolved, so they must not be counted as passed until upstream official `yolo12*-seg.pt` weights resolve. YOLO26 records are historical evidence only; the former matrix script was deleted and cannot be rerun. YOLO26 NCNN is not a supported export/deployment target. These YOLO paths do not expand scope to YOLOv5 segmentation, YOLOv5 P6, classification, pose, OBB, anomaly, YOLO-World, YOLOE-26, tracking, or other tasks. Dedicated semantic segmentation is validated separately through SMP and is not a YOLO instance-segmentation capability.

For SMP semantic segmentation, use:

```powershell
.\tools\phase-smp-semantic-segmentation-smoke.ps1
```

`phase-smp-semantic-segmentation-smoke.ps1` remains the minimal adapter smoke: it generates a tiny Mask PNG semantic dataset, compiles the SMP trainer/evaluator, checks `segmentation_models_pytorch`, Torch, timm, ONNX, ONNX Runtime, Pillow, NumPy, and PyYAML, trains only when dependencies are available, exports `best.onnx`, runs evaluation, and verifies overlays. If dependencies are absent, use `-SkipTraining` for package/layout validation or treat the smoke summary as `blocked`, not passed.

历史 SMP GPU realtest 仅保留在归档证据中；当前 SMP 产品验收由官方 Python 适配器和 `runRuntimeDeliveryWorkflow` 负责，不再调用 Worker smoke CLI。

For OBB rotated-box detection, use:

```powershell
```

OBB v1 accepts `taskType=obb_detection`, `datasetFormat=yolo_obb`, `trainingBackend=ultralytics_yolo_obb`, and `modelFamily=yolo_obb`. Training, first ONNX export, and evaluation are official Ultralytics OBB operations. Packaged inference, overlay, benchmark, and deployment validation use AITrain C++ ONNX Runtime with rotated quadrilateral output. NCNN is not an OBB v1 deployment target and must be reported as unsupported/rejected, not failed acceptance. TensorRT engine export is optional status evidence only. Public DOTA/DOTA-subset matrix results are workflow/benchmark evidence and must not be represented as customer-domain industrial precision.

For Anomalib anomaly detection, use:

```powershell
.\tools\phase-anomaly-anomalib-smoke.ps1
.\tools\phase-anomaly-mvtec-quality-matrix.ps1 -PythonExecutable .\.deps\envs\anomalib\python.exe
```

Anomaly v1 accepts `taskType=anomaly_detection`, `datasetFormat=anomaly_folder`, `trainingBackend=anomalib_patchcore|anomalib_efficientad`, `modelFamily=anomaly_detection`, and `runtime=anomalib_python`. The smoke generates a minimal anomaly folder, compiles `python_trainers\anomaly\anomalib_adapter.py`, and only runs PatchCore/EfficientAD train/evaluate/infer/benchmark when Anomalib dependencies are available. Missing Anomalib or EfficientAD `imagenetDir` must be recorded as `blocked`, not passed. EfficientAD uses Anomalib 2.5 `modelSize=small|medium` and training `batchSize=1`; any other value is an invalid request. `.ckpt` inference must route through Anomalib `Engine.predict(..., ckpt_path=...)`. This path validates Worker-managed Python/Anomalib artifacts and does not claim AITrain C++ ONNX/TensorRT/NCNN anomaly deployment.

`phase-anomaly-mvtec-quality-matrix.ps1` is the public MVTec evidence lane. It uses `bottle`, `hazelnut`, and `leather` with PatchCore and EfficientAD, writes `anomaly_mvtec_quality_matrix_summary.json/.csv/.md` under `.deps\anomaly-mvtec-quality-matrix`, and keeps each row's train/evaluate/infer/benchmark requests, logs, reports, heatmap, overlay, mask, and benchmark output. MVTec data is not committed; place the official archive at `.deps\datasets\downloads\mvtec_ad\mvtec_anomaly_detection.tar.xz`, pass `-MvtecArchiveUrl`, or pre-materialize categories under `.deps\datasets\materialized\mvtec-ad\<category>`. The script can materialize from the official archive, so manual pre-extraction is optional. The 2026-06-18 local default matrix passed 6/6 rows and wrote `.deps\anomaly-mvtec-quality-matrix\anomaly_mvtec_quality_matrix_summary.json`. The matrix remains public-dataset workflow evidence, not customer-domain production precision.

## Phase 49 Lite: Delivery Closeout Workbench

The GUI delivery-closeout surfaces aggregate evidence; they do not replace the scripts or Worker report commands. Use the workbench to display imported JSON/Markdown acceptance evidence in `环境 > 交付证据`, run report-only Worker commands, and review `passed` / `blocked` / `failed` / `hardware-blocked` status.

Current GUI surfaces:

- `数据集 > 质量与复核`: enter a committed review `ArtifactId`; the GUI reads quality/review JSON by package-relative member name after inventory/hash verification. It never accepts an arbitrary local JSON path or opens staging files; filter and export an X-AnyLabeling review list.
- `模型库 > 评估报告`: review model evaluation report records and visualized report details.
- `部署验证 > 部署验证 / 推理验证`：两者都只接受已登记、Manifest 和哈希校验通过的  模型包；分别运行部署验证与单图推理。
- `系统设置 > 内置能力`: review the built-in capability matrix and backend boundaries.
- `环境 > 交付证据`: summarize local RC, clean Windows, TensorRT, package integrity, customer OCR, diagnostics, and deployment validation evidence.
- Customer OCR acceptance wizard: collect Det dataset, Rec dataset, System images, official Det/Rec/System reports, and write customer OCR manifest/summary outputs.
- Export post-validation: validate ONNX by runnable inference where possible; preserve TensorRT `hardware-blocked`; validate NCNN by runtime inference for YOLO detection/segmentation when NCNN SDK/runtime and a sample image are available, otherwise report failed/blocked explicitly.
- Diagnostics bundle: collect Worker self-check, environment profile, GPU/runtime state, recent task logs/request snippets, artifact index, built-in capability state, and license summary.

The real execution entries remain:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\release-freeze-handoff.ps1
.\tools\customer-ocr-validation.ps1
.\tools\ui-workbench-walkthrough.ps1
```

当前 Worker 产品协议只保留统一 Workflow 命令；诊断、OCR 验收、环境检查、Runtime Delivery 和外部证据导入均通过对应 Workflow，并以 TaskId/ArtifactId/EvidenceId 返回结果。旧 `runCustomerOcrAcceptance`、`collectDiagnostics`、裸路径部署和 `--*-smoke` 业务入口均已删除，不得作为验收入口。长任务实现仍必须位于 Worker/core 边界，不能进入 `MainWindow`。底层 ONNX Runtime 单次同步 `infer` 进入后不可中途抢占，取消会在该次调用返回后收口。NCNN 验收只覆盖产品矩阵允许且 Manifest 合同完整的 Detection/Segmentation，不得扩展到 OBB、SMP、异常检测、OCR 或未知 decoder；TensorRT 的官方 YOLO decoder 与真实 `infer` 尚未实现，probe/engine 或历史外部证据不能作为 TensorRT 推理通过结论。

NCNN evidence refresh on 2026-05-16:

- Hyuto YOLOv8 detection ONNX -> NCNN passed runtime deployment validation with `predictionCount=14` under `.deps\github-ncnn-smoke\hyuto-yolov8\runtime-output`.
- nihui `ncnn-android-yolov8` preconverted YOLOv8n-seg pnnx/DFL NCNN passed segmentation runtime deployment validation with an explicit AITrain sidecar and `predictionCount=100` under `.deps\github-ncnn-smoke\nihui-yolov8n-seg-ncnn\runtime-output\deployment-validation`.
- Hyuto and X-AnyLabeling YOLOv8-seg ONNX -> `onnx2ncnn` attempts currently fail preflight because the generated NCNN param still contains unsupported `Shape` layers. The expected behavior is a failed validation report, not a Worker crash.
- NCNN failed reports now include `errorCode`, `failureCategory`, `nextAction`, and `diagnosticHints`. Expected categories are `sdk_missing`, `sample_missing`, `sidecar_missing`, `unsupported_layer`, and `runtime_failed`.

## Phase 17: Local Baseline Freeze

Use this mode before changing release or acceptance documentation:

```powershell
.\tools\acceptance-smoke.ps1 -LocalBaseline
```

Expected result:

- `harness-check.ps1` configures, builds, and passes CTest.
- No build outputs, downloaded packages, model weights, datasets, or `.deps` files are staged.
- Scaffold/baseline wording remains explicit in docs and UI text.

For the isolated official PaddleOCR smoke, run:

```powershell
.\tools\phase16-ocr-official-smoke.ps1
```

This validates official PaddleOCR train/export/inference wiring on a tiny generated dataset. The script checks out a pinned PaddleOCR source ref and records the requested/resolved ref in the report. It does not validate OCR accuracy.

For the full official PaddleOCR Det + Rec + System chain, run:

```powershell
.\tools\phase31-paddleocr-full-official-smoke.ps1
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv4
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv6 -PPOCRv6Tier tiny
.\tools\phase-ppocrv6-model-matrix-smoke.ps1
```

This validates official Det train/export, official Rec train/export, and official `predict_system.py` inference with `use_angle_cls=false`. It checks the Det and Rec inference configs, official reports, `official_system_prediction.json`, `system_results.txt`, and visualized output images. The default is PP-OCRv5 mobile Det/Rec; `-OcrVersion PP-OCRv4` switches back to the legacy v4 mobile presets, and `-OcrVersion PP-OCRv6 -PPOCRv6Tier tiny|small|medium` selects matching v6 Det/Rec presets. `phase-ppocrv6-model-matrix-smoke.ps1` checks all six v6 Det/Rec presets in prepare-only mode and runs one v6 tiny full-chain smoke. It is still a tiny CPU smoke run, so it validates wiring and artifacts rather than OCR quality.

For the PP-OCRv5 GPU official production-chain gate, run:

```powershell
.\tools\phase50-paddleocr-v5-gpu-official-chain.ps1 -UseGpu
```

This wrapper first verifies that the selected OCR Python environment has a CUDA-enabled PaddlePaddle build. GPU mode is the default; `-UseGpu` is accepted as an explicit switch. If CUDA Paddle is missing, it writes a blocked summary instead of downgrading to CPU. A passing run must produce Det, Rec, and System official reports, System prediction output, the production OCR acceptance report, and the chain summary.

## Phase 18: Package Acceptance

From the source tree, validate the install layout with:

```powershell
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

From a packaged build directory, validate the already-installed layout with:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

Expected result:

- `AITrainStudio.exe` and `aitrain_worker.exe` exist.
- The built-in capability registry is reported through `aitrain_worker.exe --builtin-capabilities`; it is the single source of truth for YOLO, semantic segmentation, anomaly detection, PaddleOCR, and dataset interoperability.
- Runtime folders, docs, examples, Python trainers, requirements, and this acceptance script are present.
- Worker self-check emits JSON and reports missing optional runtimes clearly.

## Phase 19: TensorRT Acceptance

RTX 4090 D TensorRT acceptance has passed for the current validation lane, with evidence archived in `docs\validation\rtx4090-validation-evidence-20260615.json`. Older GTX 1060 / SM 61 hardware remains `hardware-blocked` for TensorRT 10 and must not be treated as passing.

Acceptance requires:

- Worker self-check resolves CUDA, cuDNN, TensorRT, TensorRT Plugin, TensorRT ONNX Parser, and ONNX Runtime components needed for ONNX-to-engine export.
- 当前本地门禁只检查包内运行时依赖和 Runtime Delivery 的能力分类；不存在独立的 TensorRT smoke CLI。
- 真实 TensorRT decoder/infer 尚未实现，不能把 SDK、engine 或历史外部结果写成产品推理通过。
- 若重新开放 RTX/SM 75+ 外部验收，必须从已登记 ModelPackage 运行 Runtime Delivery，提交完整 SDK、硬件、Manifest、Artifact 和报告证据；当前本地代码不提供裸 ONNX 路径入口。

## Phase 43 Lite: External Acceptance Handoff

Before sending a package to an external tester, read:

```text
docs\external-acceptance-handoff.md
docs\acceptance-templates\clean-windows-acceptance-result.md
docs\acceptance-templates\tensorrt-acceptance-result.md
```

Clean Windows package acceptance from the unpacked package root:

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

TensorRT 外部验收只接受已登记 ModelPackage 的 Runtime Delivery 证据，不再接受裸 ONNX 或旧 smoke CLI。返回填好的模板、`acceptance_summary.json`、完整控制台输出、Worker self-check JSON、包体布局摘要、GPU/驱动证据、Manifest/Artifact 清单及 Runtime Delivery 报告；将不支持硬件的结果与历史 RTX 4090 D 证据分开保存。

## Phase 44 Lite: Release Freeze Handoff

From the source tree, generate a traceable package handoff:

```powershell
.\tools\release-freeze-handoff.ps1
```

This runs the local RC closeout by default, generates the CPack ZIP, computes SHA256 hashes, and writes:

- `build-vscode\release-freeze-handoff\release_handoff_manifest.json`
- `build-vscode\release-freeze-handoff\release_handoff_summary.md`

Send those files with the ZIP package and the Phase 43 external acceptance templates. This is still a local handoff preparation step; clean Windows and TensorRT acceptance remain external until evidence is returned.

## Phase 45: YOLO New-Version Productization

Run the required YOLO11 and YOLO12 detection/segmentation model matrix:

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1
```

如需单独确认 YOLO12 行，可显式启用该矩阵选项（不代表任何旧产品命令兼容）：

```powershell
.\tools\phase45-yolo-model-matrix-smoke.ps1 -IncludeYolo12
```

Expected artifacts:

- `yolo_model_matrix_summary.json` under `.deps\phase45-yolo-model-matrix` by default.
- For each required model: `ultralytics_training_report.json`, `best.pt`, and exported ONNX.
- If CTest is available, the script runs C++ ONNX Runtime regression checks with `AITRAIN_ACCEPTANCE_SMOKE_ROOT` pointed at the Phase 45 work directory.

Required Phase 45 models are `yolo11n.yaml`, `yolo11n-seg.yaml`, `yolo12n.yaml`, and `yolo12n-seg.yaml`. This is a wiring/artifact/productization smoke, not an accuracy benchmark.

## P1: YOLO Full Preset and Export-Argument Matrix

Run the required full YOLO detection/instance-segmentation matrix:

```powershell
.\tools\acceptance-smoke.ps1
```

The script writes `p1_yolo_full_matrix_summary.json` under `.deps\phase-p1-yolo-full-matrix` by default. `status=passed` requires every required row with a resolvable official asset to pass; download failures, official model-resolution failures, GPU/TensorRT gaps, and INT8 calibration failures are recorded as failed/blocker evidence rather than skipped or silently downgraded. YOLO12 segmentation `.pt` is the current known model-resolution blocker in Ultralytics 8.3.171 and must be recorded as `blocked_missing_official_weight` or equivalent until upstream weights resolve.

Required rows:

- Standard detection: YOLOv5u P5 plus YOLOv8 / YOLO11 / YOLO12, scales `n/s/m/l/x`, source types `.yaml` and `.pt`.
- Standard instance segmentation: YOLOv8 / YOLO11, scales `n/s/m/l/x`, source types `.yaml` and `.pt`; YOLO12, scales `n/s/m/l/x`, source type `.yaml`, with `.pt` rows blocked unless official `yolo12*-seg.pt` weights resolve.
- YOLOv8 detection architecture YAML: P2 and P6 variants, scales `n/s/m/l/x`.

YOLOv5u detection rows use `yolov5n/s/m/l/x.yaml` architecture entries and `yolov5nu/su/mu/lu/xu.pt` pretrained entries. Original `ultralytics/yolov5` repository weights, YOLOv5 segmentation, and YOLOv5 P6 variants are outside the required P1 matrix.

Each row must produce a completed Worker training task, `best.pt`, `best.onnx` or official ONNX export path, and `ultralytics_training_report.json` containing `model`, `backend`, `metrics`, and `ultralyticsExportArgs`. `.pt` source rows must preserve the `.pt` model name in the report to prove the pretrained fine-tuning route was used.

P1 also covers official YOLO export parameters from both the training page and model export page:

- ONNX supports `dynamic` and `half`, but rejects `int8=true`.
- TensorRT supports INT8 only through official Ultralytics export and requires compatible GPU/TensorRT plus calibration data.
- NCNN rejects `dynamic`, `half`, `int8`, and `end2end=true`; `.pt -> ncnn` first creates a static FP32 traditional official ONNX intermediate, then runs `onnx2ncnn`.

The full P1 matrix is intentionally separate from `harness-check.ps1` because it downloads/runs many official YOLO models and is expected to be slow on CPU.

## YOLO26 Compatibility Matrix

YOLO26 detection/instance-segmentation matrix results are historical only. The former matrix script and its probe/focused/full modes were deleted; `.deps\phase-yolo26-model-matrix` is retained only as archived evidence and cannot be used as a current gate. YOLO26 NCNN is not an accepted target.

YOLO26 semantic segmentation, classification, pose, OBB, tracking, YOLOE-26, and other task variants are not supported.

## Historical Phase 46/47 OCR ONNX Wiring

Phase 46/47 C++ OCR ONNX work is historical wiring evidence only. It is not a current OCR product route, deployment gate, benchmark, or customer acceptance requirement. The required OCR product acceptance path is PaddleOCR official Det/Rec/System reports.

The historical CTest suite covered the deterministic postprocess path with a synthetic probability map. That coverage is retained only as diagnostic background. New OCR acceptance should not require C++ OCR ONNX evidence. Historical behavior:

- single connected text region becomes one `ocr_detection` prediction;
- small noise regions are filtered by `minArea`;
- predictions include four-point polygons, normalized boxes, confidence, and pixel area;
- overlay rendering returns a valid image.

This phase validates C++ postprocess wiring only. It does not mark PP-OCRv5 official training/export accuracy as accepted, and the full official PaddleOCR Det+Rec+System path remains `tools\phase31-paddleocr-full-official-smoke.ps1` / official `predict_system.py`.

### Historical Phase 47 Evidence

Phase 47 evidence may remain in delivery archives to explain past wiring scope, but it is no longer a product route. Do not run Phase 47 as a new OCR closeout gate.

Use `tools\phase31-paddleocr-full-official-smoke.ps1`, `tools\production-ocr-acceptance.ps1`, `tools\customer-ocr-validation.ps1`, or the Phase 49 GUI customer OCR wizard for current OCR closeout.

Historical artifacts may include:

- `paddleocr_det_official.onnx`
- `paddleocr_det_official.onnx.aitrain-export.json`
- `paddleocr_det_onnx_smoke_summary.json`

`tools\phase47-paddleocr-det-onnx-smoke.ps1` is now a compatibility boundary check only: it writes a blocked official-only summary and exits with code 11. It no longer runs Phase 31, Paddle2ONNX conversion, or Worker C++ OCR ONNX smoke.

RTX 4090D historical Phase 47 evidence is archived in `docs\validation\rtx4090-validation-evidence-20260615.json`. Keep it as past wiring context only; do not use it as new OCR acceptance evidence.

Current OCR acceptance requires PaddleOCR official Det/Rec/System reports and representative data.

## Production OCR Acceptance

Use this gate only with representative, non-tiny OCR data and returned official reports:

```powershell
.\tools\production-ocr-acceptance.ps1 `
  -DetDataset <paddleocr-det-dataset> `
  -RecDataset <paddleocr-rec-dataset> `
  -SystemImages <end-to-end-image-folder> `
  -OfficialDetReport <paddleocr_official_det_report.json> `
  -OfficialRecReport <paddleocr_official_rec_report.json> `
  -OfficialSystemReport <paddleocr_official_system_report.json>
```

The script writes:

- `production_ocr_acceptance_report.json`
- `production_ocr_acceptance_summary.md`

Default thresholds are intentionally higher than tiny smoke data: at least 100 Det images, 1000 Rec samples, 100 System images, and Rec accuracy > 0.70. CER is recorded by default but is not blocking unless `-RequireRecCer` is supplied. If evidence is missing, the script exits blocked and records the missing checks instead of marking production OCR as accepted.

The repeatable production chain supports PP-OCRv4, PP-OCRv5, and PP-OCRv6 presets:

```powershell
.\tools\run-production-ocr-official-chain.ps1 -OcrVersion PP-OCRv5 -UseGpu -AllowBlocked
.\tools\run-production-ocr-official-chain.ps1 -OcrVersion PP-OCRv6 -PPOCRv6Tier medium -AllowBlocked
```

PP-OCRv5 and PP-OCRv6 production-chain runs still use only PaddleOCR official Det, Rec, and System reports. They do not add a PaddleOCR C++ local OCR route and do not claim PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document orientation classification, document unwarping, or text-line orientation classification coverage. PP-OCRv6 tiny follows the official language-coverage limitation and remains workflow evidence unless customer-domain data is accepted.

Current RTX 4090D validation note: the 2026-06-05 refresh/follow-up records passing evidence for LocalBaseline+Package, GUI walkthrough, TensorRT, CPUTrainingSmoke, Phase45, Phase47 Det ONNX+CTest, and public OCR GPU workflow. The follow-up summary and related historical RTX4090 evidence are archived in `docs\validation\rtx4090-validation-evidence-20260615.json`. The public OCR GPU workflow remains public Total-Text workflow evidence only; the 2026-06-05 public rerun passed under the current `accuracy>0.70` Rec gate, while the 2026-05-13 closeout remains a historical higher-accuracy public baseline.

Customer-domain production claims require customer/target-domain data and should use `tools\customer-ocr-validation.ps1` or the Phase 49 GUI customer OCR wizard. Public Total-Text, generated smoke data, and `.deps` samples can prove workflow execution only; they must remain `blocked` or smoke-only for production OCR readiness.

## Phase 20: Small Training Smoke

Run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets
```

The script tries to materialize Ultralytics official COCO8 / COCO8-seg datasets through the installed `ultralytics` package. If that fails, it falls back to the generated minimal datasets from `examples\create-minimal-datasets.py`.

To require real public materialization instead of fallback, run:

```powershell
.\tools\acceptance-smoke.ps1 -PublicDatasets -RequirePublicDatasets
```

Materialization details are written as JSON reports next to the generated datasets. The reports include the source yaml, download URL, Ultralytics version, output `data.yaml`, and fallback state.

The CTest step receives `AITRAIN_ACCEPTANCE_SMOKE_ROOT`, so segmentation and OCR ONNX Runtime tests use the artifacts generated by this smoke run, including custom `-WorkDir` values.

Expected artifacts:

- YOLO detection: `best.pt`, ONNX export, and `ultralytics_training_report.json`.
- YOLO segmentation: `best.pt`, ONNX export, and `ultralytics_training_report.json` with mask metrics when exposed.
- Official PaddleOCR Rec: `official_model\best_accuracy.pdparams`, `official_inference\inference.yml`, `official_prediction.json`, and `paddleocr_official_rec_report.json`.
- Official PaddleOCR full chain: Det `official_model\best_accuracy.pdparams`, Det `official_inference\inference.yml`, Rec `official_inference\inference.yml`, `official_system_prediction.json`, `system_results.txt`, visualization images, `paddleocr_official_det_report.json`, `paddleocr_official_rec_report.json`, and `paddleocr_official_system_report.json`.

If a public dataset requires interactive registration, record it as an external dataset blocker. Do not block the required smoke path as long as the generated minimal dataset path passes.

## Phase 33: Local CPU Small/Medium Training Smoke

Run:

```powershell
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
```

This mode does not download public datasets and does not run TensorRT. It generates deterministic `--profile cpu-smoke` data under `<WorkDir>\cpu-training-smoke\generated`, trains YOLO detection and YOLO segmentation for 3 CPU epochs at image size 128, runs official PaddleOCR Rec train/export/inference through `phase16-ocr-official-smoke.ps1`, and runs CTest against those artifacts. If the official OCR environment is not configured, the mode reports a clear blocked/failed environment error instead of falling back to diagnostic CTC training.

Expected artifacts:

- `cpu_training_smoke_summary.json` with dataset counts, parameters, report paths, artifact paths, metrics, and elapsed time.
- YOLO detection and segmentation: `best.pt`, `best.onnx`, and `ultralytics_training_report.json`.
- Official PaddleOCR Rec: `official_model\best_accuracy.pdparams`, `official_inference\inference.yml`, `official_prediction.json`, `dict.txt`, and `paddleocr_official_rec_report.json`.

This is a stronger local integration smoke than `-PublicDatasets`, but it is still not a model accuracy benchmark.

## Phase 22-30: Local Usability Baseline

These phases do not change TensorRT acceptance. They make the local RC easier to use and re-check:

- Task history: GUI-started inference, dataset validation, and dataset split create SQLite task records and record Worker artifacts.
- Artifact browsing: task details show artifacts, metrics, and exports; JSON/text/image/ONNX/model-directory artifacts have a preview or a clear unsupported message.
- Dataset management: the GUI auto-detects YOLO detection, YOLO segmentation, PaddleOCR Rec, and PaddleOCR Det layouts, and split supports all four.
- PaddleOCR Det data: the GUI auto-detects `det_gt.txt` / `det_gt_train.txt`, validates PaddleOCR native detection rows, and split writes train/val/test label files plus `split_report.json`.
- Public datasets: COCO8 / COCO8-seg materialization is a standalone machine-readable script with required and fallback modes.
- Official OCR: `phase16-ocr-official-smoke.ps1` covers official Rec train, export, and recognition inference. `phase31-paddleocr-full-official-smoke.ps1` covers official Det + Rec train/export and official System inference.
- Annotation: the dataset page launches X-AnyLabeling as the default external annotation tool, detects its local path, and provides a post-labeling refresh/revalidation action.
- UX closeout: task history can be filtered by category, status, and search text; failed tasks show a short diagnostic next-step summary.
- Delivery closeout: sample review, environment-page delivery evidence, customer OCR validation, diagnostics, and deployment validation are available through the Phase 49 GUI surfaces and Worker report commands.

Suggested manual GUI walkthrough:

```powershell
.\tools\ui-workbench-walkthrough.ps1
```

RC walkthrough wrapper 会在 1280x820 非全屏视口覆盖 `总览`、`项目`、`数据集`、`训练实验`、`任务与产物`、`模型库`、`部署验证`、`环境` 和 `系统设置`，默认把 `ui_walkthrough_rc_summary.json` 写到 `.deps\UI-Walkthrough\rc`。QtTest 另外覆盖 `数据集 > 质量与复核`、`模型库 > 评估报告`、`部署验证 > 部署验证 / 推理验证`、`系统设置 > 内置能力 / 应用设置` 与 `环境 > 交付证据`。

If offline licensing stops startup at the registration dialog, the wrapper records `status=blocked` and `errorCode=license_required`. That is not a GUI layout pass; configure a valid license token and build-time public key, then rerun the wrapper.

`tools\local-rc-closeout.ps1` runs this walkthrough by default after harness/package smoke. Use `-SkipGuiWalkthrough` only for intentionally headless environments, and record that omission in the handoff notes.

For manual exploration beyond the automated pass, create or open a project, import generated detection, segmentation, OCR Rec, and OCR Det datasets, launch X-AnyLabeling from the dataset page, use post-labeling refresh/revalidation, validate and split each dataset, run one training/export/inference path, then confirm the task queue detail view lists report, checkpoint/model, ONNX, overlay, visualized OCR image, and prediction JSON/TXT artifacts. 数据集页不得暴露任意“打开数据目录”操作；裸路径仅允许出现在明确的导入/转换边界，修复流程必须使用 Session/Artifact 身份；同时打开 `数据集 > 质量与复核`、`部署验证`、`系统设置 > 内置能力` 和 `环境 > 交付证据`，确认复核清单导出、导出/推理校验、内置能力状态、诊断包、客户 OCR 门禁和部署验证条目可见。

X-AnyLabeling is detected from `AITRAIN_XANYLABELING_EXE`, the app directory, `tools\x-anylabeling`, `.deps\tools\annotation-tools\X-AnyLabeling`, or `PATH`. Keep downloaded binaries in `.deps\` unless a separate redistribution review is completed.

## Phase 21: Release Closeout

Before marking a release baseline:

```powershell
.\tools\harness-context.ps1
.\tools\harness-check.ps1
.\tools\package-smoke.ps1 -SkipBuild
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package
.\tools\phase31-paddleocr-full-official-smoke.ps1
.\tools\phase31-paddleocr-full-official-smoke.ps1 -OcrVersion PP-OCRv4
```

Then check:

- `docs\harness\current-status.md` remains the source of truth.
- `docs\yolo-model-support-matrix.md` remains the source of truth for productized YOLO model-family status.
- Phase 7 / Phase 10 TensorRT RTX 4090 D acceptance passed unless RTX / SM 75+ smoke passed.
- Third-party backend license notes remain visible, especially Ultralytics AGPL / Enterprise constraints.
- SMP semantic segmentation license/dependency notes remain visible; SMP acceptance is ONNX Runtime only, and NCNN/TensorRT export must not be required for SMP.
- Legacy C++ tiny detector, segmentation baseline, OCR baseline, small OCR CTC, and shipped Python mock trainer implementations remain removed from the product training path.
- Historical Phase 46/47 OCR ONNX wiring evidence remains archived, but current OCR acceptance is official-only through PaddleOCR Det/Rec/System reports and customer-domain data.
- PP-OCRv5/PP-OCRv6 support is scoped to official Det/Rec/System presets and reports. It does not add PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document direction classification, image correction, text-line direction classification, or PaddleOCR C++ local deployment to the accepted product route.
