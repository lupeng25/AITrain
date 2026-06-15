# AITrain Studio Acceptance Runbook

This runbook is the Phase 17-50 plus P1 acceptance path, with YOLO26 tracked as a separate compatibility phase. It freezes the local baseline, validates the packaged layout, prepares TensorRT external acceptance, runs small training smoke checks, covers the current local usability additions, includes the external acceptance handoff package, records a traceable release-freeze package identity, validates newer YOLO detection/segmentation model-family candidates, documents the delivery-closeout workbench, adds the PP-OCRv5 GPU official-chain gate, defines the P1 full YOLO preset/export-argument matrix, adds the independent YOLO26 detection/instance-segmentation matrix, and adds PP-OCRv6 Det/Rec/System official-chain acceptance. OCR acceptance is official-only through PaddleOCR Det/Rec/System reports. RTX 4090 D TensorRT smoke evidence is recorded under `.deps\\rtx4090-validation\\acceptance-tensorrt`; clean Windows and customer-domain OCR production evidence still require returned external/customer data.

## Acceptance Modes

Run the unified smoke script from the repository root:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\acceptance-smoke.ps1 -LocalBaseline
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
.\tools\acceptance-smoke.ps1 -PublicDatasets
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
.\tools\phase45-yolo-model-matrix-smoke.ps1
.\tools\phase-p1-yolo-full-matrix-smoke.ps1
.\tools\phase-yolo26-model-matrix-smoke.ps1
.\tools\phase-ppocrv6-model-matrix-smoke.ps1
.\tools\acceptance-smoke.ps1 -TensorRT
.\tools\customer-ocr-validation.ps1
```

The same script is also installed into packaged builds under `tools\acceptance-smoke.ps1`. From a package root, run:

```powershell
.\tools\acceptance-smoke.ps1 -Package
.\tools\acceptance-smoke.ps1 -TensorRT
```

All generated datasets, official downloads, trainer outputs, and smoke artifacts must stay under `.deps\` or another explicitly supplied work directory. Do not add them to source control.

Every `acceptance-smoke.ps1` run writes `acceptance_summary.json` to its work directory. The summary records requested modes, status, work directory, start/end timestamps, failure reason, and hardware-blocked reason when applicable.

For the current local release-candidate closeout path, use `docs\local-rc-closeout.md`. It is the Phase 42 Lite entry point for source build, package smoke, optional local baseline, optional CPU training smoke, Phase 41 environment profile GUI walkthrough, and boundary wording checks.

For external handoff, use `docs\external-acceptance-handoff.md` and the result templates under `docs\acceptance-templates`. These files define the package-root commands, TensorRT commands, required returned evidence, and the clean separation between local RC closeout and external acceptance.

For release-freeze package identity, use `docs\release-freeze-handoff.md` and `tools\release-freeze-handoff.ps1`. This generates the CPack ZIP, SHA256 hashes, and a handoff manifest without marking external acceptance as passed.

For YOLO model-family productization, use `docs\yolo-model-support-matrix.md`. Phase 45 validates newer Ultralytics detection/segmentation nano model names only. P1 validates YOLOv5u standard P5 detection, YOLOv8 / YOLO11 / YOLO12 detection, YOLOv8 / YOLO11 instance segmentation, YOLO12 instance-segmentation `.yaml`, `.yaml` and `.pt` source types where the official asset resolves, YOLOv8 P2/P6 detection YAML architectures, and the official export-argument protocol. YOLO12 segmentation `.pt` rows are blocked in the recorded Ultralytics 8.3.171 environment because `yolo12n-seg.pt` cannot be resolved, so they must not be counted as passed until upstream official `yolo12*-seg.pt` weights resolve. YOLO26 is validated by `tools\phase-yolo26-model-matrix-smoke.ps1` as a separate compatibility phase for detection and instance segmentation only. The shared Ultralytics 8.3.171 lifecycle lane blocked/failed all 20 YOLO26 rows, but the isolated 2026-06-15 targeted full lane passed 20/20 rows for training, official ONNX, AITrain C++ ONNX inference, and TensorRT. YOLO26 NCNN is not a supported export/deployment target. These paths do not expand scope to YOLOv5 segmentation, YOLOv5 P6, semantic segmentation, classification, pose, OBB, anomaly, YOLO-World, YOLOE-26, tracking, or other tasks.

## Phase 49 Lite: Delivery Closeout Workbench

The GUI delivery-closeout surfaces aggregate evidence; they do not replace the scripts or Worker report commands. Use the workbench to display imported JSON/Markdown acceptance evidence in `环境 > 交付证据`, run report-only Worker commands, and review `passed` / `blocked` / `failed` / `hardware-blocked` status.

Current GUI surfaces:

- `样本复核`: load `problem_samples.json`, `error_samples.json`, `rework_sample_set.json`, or evaluation reports; filter by source, reason, class, split, OCR edit distance / CER, or search text; export an X-AnyLabeling review list.
- `环境 > 交付证据`: summarize local RC, clean Windows, TensorRT, package integrity, customer OCR, diagnostics, and deployment validation evidence.
- Customer OCR acceptance wizard: collect Det dataset, Rec dataset, System images, official Det/Rec/System reports, and write customer OCR manifest/summary outputs.
- Export post-validation: validate ONNX by runnable inference where possible; preserve TensorRT `hardware-blocked`; validate NCNN by runtime inference for YOLO detection/segmentation when NCNN SDK/runtime and a sample image are available, otherwise report failed/blocked explicitly.
- Diagnostics bundle: collect Worker self-check, environment profile, GPU/runtime state, recent task logs/request snippets, artifact index, plugin state, and license summary.

The real execution entries remain:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\release-freeze-handoff.ps1
.\tools\customer-ocr-validation.ps1
.\tools\phase-ncnn-runtime-smoke.ps1 -NcnnRoot <ncnn-sdk-root> -OnnxPath <best.onnx> -SampleImagePath <sample.png> -OutputDir <smoke-output> -TaskType detection
.\build-vscode\bin\aitrain_worker.exe --ncnn-param-smoke <model.param> --image <sample.png> --output <smoke-output> --task-type segmentation
.\tools\ui-workbench-walkthrough.ps1
```

Worker command equivalents are `runCustomerOcrAcceptance`, `collectDiagnostics`, and `validateDeploymentArtifact`. These are report/validation commands and must stay outside `MainWindow`.

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
- The three built-in plugins load through `aitrain_worker.exe --plugin-smoke`.
- Runtime folders, docs, examples, Python trainers, requirements, and this acceptance script are present.
- Worker self-check emits JSON and reports missing optional runtimes clearly.

## Phase 19: TensorRT Acceptance

RTX 4090 D TensorRT acceptance has passed for the current validation lane, with evidence archived in `docs\validation\rtx4090-validation-evidence-20260615.json`. Older GTX 1060 / SM 61 hardware remains `hardware-blocked` for TensorRT 10 and must not be treated as passing.

To reproduce or refresh the evidence on an RTX / SM 75+ Windows machine or matching cloud GPU:

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

Acceptance requires:

- Worker self-check resolves CUDA, cuDNN, TensorRT, TensorRT Plugin, TensorRT ONNX Parser, and ONNX Runtime components needed for ONNX-to-engine export.
- `acceptance-smoke.ps1 -TensorRT` generates a small official Ultralytics YOLO ONNX artifact, or uses `-TensorRtOnnxPath <official.onnx>` when supplied.
- `aitrain_worker.exe --tensorrt-smoke <official.onnx>` builds a TensorRT engine from an official Ultralytics ONNX artifact. This is an official-artifact smoke with AITrain TensorRT export/runtime checks; it does not use the removed tiny-detector TensorRT inference fixture and is not an end-to-end Ultralytics Python runtime check.
- The result is recorded back in `docs\harness\current-status.md`; the current RTX 4090 D pass is already recorded.

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

RTX 4090 D TensorRT smoke evidence is recorded under `.deps\\rtx4090-validation\\acceptance-tensorrt`.

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT -WorkDir .deps\acceptance-tensorrt
```

Return the filled template, `acceptance_summary.json`, full console output, Worker self-check JSON, package layout summary, GPU/driver evidence for TensorRT, and the exact TensorRT smoke pass/fail/hardware-blocked output. Keep historical unsupported-GPU runs separate from the recorded RTX 4090 D pass.

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

The legacy YOLO12 command remains accepted, but it is no longer needed because YOLO12 is part of the default matrix:

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
.\tools\phase-p1-yolo-full-matrix-smoke.ps1
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

Run the separate YOLO26 detection/instance-segmentation matrix:

```powershell
.\tools\phase-yolo26-model-matrix-smoke.ps1
.\tools\phase-yolo26-model-matrix-smoke.ps1 -PrepareEnvironment -ProbeOnly -Device 0
.\tools\phase-yolo26-model-matrix-smoke.ps1 -Focused -Epochs 1 -Device 0
.\tools\phase-yolo26-model-matrix-smoke.ps1 -Full -Epochs 100 -Device 0
```

Full mode writes `yolo26_model_matrix_summary.json` under `.deps\phase-yolo26-model-matrix` by default and has 20 required rows:

- YOLO26 detection: `yolo26n/s/m/l/x.yaml` and `yolo26n/s/m/l/x.pt`.
- YOLO26 instance segmentation: `yolo26n/s/m/l/x-seg.yaml` and `yolo26n/s/m/l/x-seg.pt`.

Probe mode writes `yolo26_environment_self_check.json` and validates the isolated Python environment, CUDA Torch when GPU is requested, `cfg/models/26`, and nano `.yaml` / `.pt` model loading before training. Focused mode is the quick lifecycle gate for nano models and covers `yolo26n.yaml`, `yolo26n.pt`, `yolo26n-seg.yaml`, and `yolo26n-seg.pt`.

`end2end=auto` uses the loaded Ultralytics model config when it exposes an `end2end` default; otherwise it falls back to YOLO26 detection=true and other models=false. Reports must record the normalized boolean under `ultralyticsExportArgs.end2end`. Each passed YOLO26 row must produce a completed Worker training task, `best.pt`, ONNX, `ultralytics_training_report.json`, AITrain inference JSON, overlay output, and explicit ONNX/TensorRT deployment statuses. YOLO26 NCNN is not an accepted target.

Current status: in the recorded shared Ultralytics 8.3.171 full lifecycle environment, all 20 YOLO26 rows failed before useful training. `.yaml` rows report missing model config files, most `.pt` rows report missing official weights, and nano `.pt` rows expose package/code incompatibility (`SPPF.__init__()` argument mismatch and missing `Segment26`). The isolated YOLO26 matrix full summary passed on 2026-06-15 for 20/20 training, official ONNX export, AITrain C++ ONNX inference, and TensorRT deployment validation. YOLO26 NCNN is removed from the accepted deployment target list.

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

The RC walkthrough wrapper runs the 1280x820 non-fullscreen page set: `总览`, `项目`, `数据集`, `样本复核`, `训练实验`, `任务与产物`, `模型库`, `评估报告`, `模型导出`, `推理验证`, `插件`, `环境`, and `设置`. It writes `ui_walkthrough_rc_summary.json` under `.deps\UI-Walkthrough\rc` by default and should be treated as the repeatable GUI usability gate. The `环境 > 交付证据` tab is covered by QtTest because it is no longer a standalone main navigation page.

If offline licensing stops startup at the registration dialog, the wrapper records `status=blocked` and `errorCode=license_required`. That is not a GUI layout pass; configure a valid license token and build-time public key, then rerun the wrapper.

`tools\local-rc-closeout.ps1` runs this walkthrough by default after harness/package smoke. Use `-SkipGuiWalkthrough` only for intentionally headless environments, and record that omission in the handoff notes.

For manual exploration beyond the automated pass, create or open a project, import generated detection, segmentation, OCR Rec, and OCR Det datasets, launch X-AnyLabeling from the dataset page, use post-labeling refresh/revalidation, validate and split each dataset, run one training/export/inference path, then confirm the task queue detail view lists report, checkpoint/model, ONNX, overlay, visualized OCR image, and prediction JSON/TXT artifacts. Also open `样本复核` and `环境 > 交付证据` to confirm review-list export, diagnostics, customer OCR gate, and deployment validation entries are visible.

X-AnyLabeling is detected from `AITRAIN_XANYLABELING_EXE`, the app directory, `tools\x-anylabeling`, `.deps\annotation-tools\X-AnyLabeling`, or `PATH`. Keep downloaded binaries in `.deps\` unless a separate redistribution review is completed.

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
- Legacy C++ tiny detector, segmentation baseline, OCR baseline, small OCR CTC, and shipped Python mock trainer implementations remain removed from the product training path.
- Historical Phase 46/47 OCR ONNX wiring evidence remains archived, but current OCR acceptance is official-only through PaddleOCR Det/Rec/System reports and customer-domain data.
- PP-OCRv5/PP-OCRv6 support is scoped to official Det/Rec/System presets and reports. It does not add PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document direction classification, image correction, text-line direction classification, or PaddleOCR C++ local deployment to the accepted product route.
