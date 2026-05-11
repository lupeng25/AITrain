# Production OCR Acceptance Preparation

This runbook defines the evidence required before AITrain Studio can claim production OCR readiness. Tiny Phase 31/46/47 smoke artifacts validate wiring only; they must not be used as production OCR accuracy evidence.

## Scope

Production OCR acceptance covers three paths:

- PaddleOCR Det: representative text detection training/export evidence.
- PaddleOCR Rec: representative text recognition training/export metrics.
- PaddleOCR System: end-to-end Det+Rec inference evidence on representative images.

The current full-system acceptance path remains official PaddleOCR `predict_system.py`. C++ OCR Det ONNX postprocess is optional evidence unless the acceptance run explicitly sets `-RequireDetOnnxEvidence`.

## Data Requirements

Prepare data outside the repository, preferably under an ignored path such as `.deps\production-ocr-data` or on an external drive.

Minimum gate thresholds used by `tools\production-ocr-acceptance.ps1`:

| Evidence | Default minimum |
|---|---:|
| Det images | 100 |
| Rec labeled samples | 1000 |
| System images | 100 |
| Rec accuracy | >= 0.90 |
| Rec CER | <= 0.10 |

Recommended production dataset mix:

- Real customer or target-domain images, not generated tiny samples.
- Multiple lighting/background/camera conditions.
- Printed and scanned text if both are in product scope.
- Common failure cases: blur, low contrast, skew, small text, long lines, dense text, partial occlusion.
- Train/validation/test split documented with a stable snapshot or manifest.
- Sensitive data review completed before sharing evidence outside the source machine.

## Expected Layout

Detection dataset:

```text
det_dataset\
  images\
    train\...
    val\...
    test\...
  det_gt_train.txt
  det_gt_val.txt
  det_gt_test.txt
```

Recognition dataset:

```text
rec_dataset\
  images\
    train\...
    val\...
    test\...
  rec_gt_train.txt
  rec_gt_val.txt
  rec_gt_test.txt
  dict.txt
```

System images:

```text
system_images\
  *.jpg
  *.png
```

If your source data uses a different layout, convert it through the supported PaddleOCR Det/Rec import and split flow first, then run acceptance against the normalized output.

## Public Data Preparation

For a repeatable public-data preparation pass, use:

```powershell
.\tools\prepare-production-ocr-data.ps1 -WorkDir .deps\production-ocr-data
```

The script downloads the PaddleOCR-documented Total-Text archive, normalizes PaddleOCR Det labels, crops recognition samples from text boxes, writes `dict.txt`, copies end-to-end system images, and records `manifests\production_ocr_data_manifest.json`.

This public data pass is useful for exercising the production OCR acceptance flow, but it is not customer-domain production evidence. A local CPU run can produce official PaddleOCR Det, Rec, and System reports from this public data, but the gate must still be reported as blocked when metrics do not meet thresholds. The current local public-data evidence is blocked by Rec quality: the CPU-subset official Rec report has `accuracy=0.0` and `CER≈0.9653`.

The local CPU public-data evidence paths are:

- Det report: `.deps\production-ocr-data\reports\det_official\paddleocr_official_det_report.json`
- Rec report: `.deps\production-ocr-data\reports\rec_official_cpu_subset\paddleocr_official_rec_report.json`
- System report: `.deps\production-ocr-data\reports\system_official_public_cpu\paddleocr_official_system_report.json`
- Gate summary: `.deps\production-ocr-acceptance-public-totaltext-run\production_ocr_acceptance_summary.md`

## Required Reports

The acceptance gate expects JSON report files:

- `paddleocr_official_det_report.json`
- `paddleocr_official_rec_report.json`
- `paddleocr_official_system_report.json`
- Optional: `paddleocr_det_onnx_smoke_summary.json` when requiring C++ Det ONNX evidence.

The Rec report must expose `accuracy` or `acc`, and `cer` or `CER`, either at top level or under `metrics`.

## Command

Example:

```powershell
.\tools\production-ocr-acceptance.ps1 `
  -WorkDir .deps\production-ocr-acceptance `
  -DetDataset D:\AITrainOCR\det_dataset `
  -RecDataset D:\AITrainOCR\rec_dataset `
  -SystemImages D:\AITrainOCR\system_images `
  -OfficialDetReport D:\AITrainOCR\reports\paddleocr_official_det_report.json `
  -OfficialRecReport D:\AITrainOCR\reports\paddleocr_official_rec_report.json `
  -OfficialSystemReport D:\AITrainOCR\reports\paddleocr_official_system_report.json
```

To require the C++ OCR Det ONNX smoke:

```powershell
.\tools\production-ocr-acceptance.ps1 `
  -WorkDir .deps\production-ocr-acceptance `
  -DetDataset D:\AITrainOCR\det_dataset `
  -RecDataset D:\AITrainOCR\rec_dataset `
  -SystemImages D:\AITrainOCR\system_images `
  -OfficialDetReport D:\AITrainOCR\reports\paddleocr_official_det_report.json `
  -OfficialRecReport D:\AITrainOCR\reports\paddleocr_official_rec_report.json `
  -OfficialSystemReport D:\AITrainOCR\reports\paddleocr_official_system_report.json `
  -OcrDetOnnxSummary D:\AITrainOCR\reports\paddleocr_det_onnx_smoke_summary.json `
  -RequireDetOnnxEvidence
```

To run the full local official chain from prepared or auto-prepared data, use:

```powershell
.\tools\run-production-ocr-official-chain.ps1 `
  -WorkDir .deps\production-ocr-official-chain `
  -DataDir .deps\production-ocr-data
```

For a local CPU evidence run that is expected to remain blocked if Rec metrics are weak, use:

```powershell
.\tools\run-production-ocr-official-chain.ps1 `
  -WorkDir .deps\production-ocr-official-chain `
  -DataDir .deps\production-ocr-data `
  -UseRecCpuSubset `
  -RecBatchSize 32 `
  -RecEvalEverySteps 8 `
  -AllowBlocked
```

`-SkipExistingReports` reuses existing Det/Rec/System report JSON files under the chain work directory and reruns only the final gate. `-AllowBlocked` records blocked evidence with exit code 0; without it, a blocked production gate exits nonzero.

## Rec Metric Experiments

To iterate on Rec metrics without rerunning Det and System evidence, use:

```powershell
.\tools\run-production-ocr-rec-experiment.ps1 `
  -ExperimentName rec-exp-256x64-e2-cpu `
  -TrainLimit 256 `
  -ValLimit 64 `
  -Epochs 2 `
  -BatchSize 32 `
  -EvalEverySteps 8
```

For English-heavy public data, keep the official English Rec config and dictionary aligned with the downloaded English pretrained checkpoint:

```powershell
$dict = (Resolve-Path .deps\PaddleOCR\ppocr\utils\en_dict.txt).Path
.\tools\run-production-ocr-rec-experiment.ps1 `
  -ExperimentName rec-exp-enpre-endict-256x64-e1-cpu `
  -TrainLimit 256 `
  -ValLimit 64 `
  -Epochs 1 `
  -BatchSize 32 `
  -EvalEverySteps 8 `
  -OfficialConfig configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml `
  -DictionaryFile $dict `
  -UsePretrained `
  -PretrainedModel .deps\paddleocr-pretrained\en_PP-OCRv4_mobile_rec_pretrained
```

The latest local CPU iteration used the same English config, dictionary, and pretrained checkpoint with a larger subset:

```powershell
$dict = (Resolve-Path .deps\PaddleOCR\ppocr\utils\en_dict.txt).Path
.\tools\run-production-ocr-rec-experiment.ps1 `
  -ExperimentName rec-exp-enpre-endict-512x128-e2-cpu `
  -TrainLimit 512 `
  -ValLimit 128 `
  -Epochs 2 `
  -BatchSize 32 `
  -EvalEverySteps 16 `
  -OfficialConfig configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml `
  -DictionaryFile $dict `
  -UsePretrained `
  -PretrainedModel .deps\paddleocr-pretrained\en_PP-OCRv4_mobile_rec_pretrained.pdparams
```

The follow-up local-only CPU run doubled the subset again to test whether more public crops help without external GPU acceleration:

```powershell
$dict = (Resolve-Path .deps\PaddleOCR\ppocr\utils\en_dict.txt).Path
.\tools\run-production-ocr-rec-experiment.ps1 `
  -ExperimentName rec-exp-enpre-endict-1024x256-e2-cpu `
  -TrainLimit 1024 `
  -ValLimit 256 `
  -Epochs 2 `
  -BatchSize 32 `
  -EvalEverySteps 32 `
  -OfficialConfig configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml `
  -DictionaryFile $dict `
  -UsePretrained `
  -PretrainedModel .deps\paddleocr-pretrained\en_PP-OCRv4_mobile_rec_pretrained.pdparams
```

If a longer run finishes official training but stops before export/report generation, reuse the trained checkpoint with `-ExportOnly`:

```powershell
$dict = (Resolve-Path .deps\PaddleOCR\ppocr\utils\en_dict.txt).Path
.\tools\run-production-ocr-rec-experiment.ps1 `
  -ExperimentName rec-exp-enpre-endict-256x64-e2-cpu `
  -TrainLimit 256 `
  -ValLimit 64 `
  -Epochs 2 `
  -BatchSize 32 `
  -EvalEverySteps 8 `
  -OfficialConfig configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml `
  -DictionaryFile $dict `
  -ExportOnly `
  -PretrainedModel .deps\production-ocr-rec-experiments\rec-exp-enpre-endict-256x64-e2-cpu\official_rec\official_model\best_accuracy
```

Each run writes an experiment directory under `.deps\production-ocr-rec-experiments` by default, including:

- `paddleocr_rec_official_request.json`
- `run_experiment.ps1`
- `official_rec\paddleocr_official_rec_report.json`
- `rec_experiment_summary.json`

Use `-UsePretrained -PretrainedModel <checkpoint-base>` only when initializing from a compatible PaddleOCR Rec checkpoint. A Rec experiment summary is iteration evidence only; it is not a production acceptance pass. After a qualifying Rec report is produced, rerun `tools\production-ocr-acceptance.ps1` or the full official chain to record gate status.

GPU runs should use the same experiment script with `-UseGpu`, but the script first probes the selected Python/Paddle environment and fails before training if Paddle is not CUDA-enabled. The current local OCR Python environment reports `paddleVersion=3.3.1`, `compiledWithCuda=false`, `device=cpu`; the local GPU is `NVIDIA GeForce GTX 1060`, compute capability `6.1`. Treat this machine as GPU-blocked for official PaddleOCR Rec acceleration. Use an external RTX / compute-capability 7.5+ machine with a Paddle GPU wheel compatible with its Python/CUDA stack for production-speed Rec training.

Latest local Rec experiment evidence:

- Baseline run: `.deps\production-ocr-rec-experiments\rec-exp-256x64-e2-cpu\rec_experiment_summary.json`, 256 train samples, 64 validation samples, 2 CPU epochs, batch size 32, completed but blocked (`accuracy=0.0`, `CER=0.958901679695572`).
- Pretrained English run: `.deps\production-ocr-rec-experiments\rec-exp-enpre-endict-256x64-e1-cpu\rec_experiment_summary.json`, 256 train samples, 64 validation samples, 1 CPU epoch, official English config and dictionary, official English PP-OCRv4 Rec pretrained checkpoint. This completed train/export/single-image inference and improved to `accuracy=0.4062498730469146`, `CER=0.33262638414949297`, but remains below production thresholds.
- Pretrained English 2-epoch run: `.deps\production-ocr-rec-experiments\rec-exp-enpre-endict-256x64-e2-cpu\rec_experiment_summary.json`, training completed before tool timeout and export/report was recovered through `-ExportOnly`. Best validation metrics remained `accuracy=0.4062498730469146`, `CER=0.33262638414949297`.
- Larger pretrained English run: `.deps\production-ocr-rec-experiments\rec-exp-enpre-endict-512x128-e2-cpu\rec_experiment_summary.json`, 512 train samples, 128 validation samples, 2 CPU epochs, official English config and dictionary, official English PP-OCRv4 Rec pretrained checkpoint. This completed train/export/single-image inference and improved to `accuracy=0.6458332660590348`, `CER=0.17084158273310235`, but remains below production thresholds.
- Larger local-only CPU run: `.deps\production-ocr-rec-experiments\rec-exp-enpre-endict-1024x256-e2-cpu\rec_experiment_summary.json`, 1024 train samples, 256 validation samples, 2 CPU epochs, same official English config/dictionary/pretrained checkpoint. This completed train/export/single-image inference but dropped to `accuracy=0.1741071350845029`, `CER=0.33266419593813057`; the gate rerun under `.deps\production-ocr-acceptance-enpre-1024x256-e2` remains blocked only on `official_rec_metrics`.
- Local GPU preflight: `-UseGpu` is blocked before training on this machine because the selected OCR Python environment is CPU-only Paddle (`compiledWithCuda=false`) and the local GTX 1060 is not a valid target for the current PaddleOCR GPU acceleration baseline.
- Gate rerun: `.deps\production-ocr-acceptance-enpre-512x128-e2\production_ocr_acceptance_summary.md` remains `blocked`; Det data/report, Rec data size, and System images/report passed, and only `official_rec_metrics` failed.

## Returned Evidence

Archive or return:

- `production_ocr_acceptance_report.json`
- `production_ocr_acceptance_summary.md`
- `production_ocr_official_chain_summary.json` when using `run-production-ocr-official-chain.ps1`.
- The three official report JSON files.
- Dataset manifests or snapshot reports proving sample counts and split lineage.
- Representative prediction visualizations from official System inference.
- The completed `docs\acceptance-templates\production-ocr-acceptance-result.md`.

Only update release or harness status after this evidence is available. A blocked run is useful evidence; record it as blocked rather than weakening thresholds.
