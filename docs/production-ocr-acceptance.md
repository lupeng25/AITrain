# Production OCR Acceptance Preparation

This runbook defines the evidence required before AITrain Studio can claim production OCR readiness. OCR acceptance is official-only: use PaddleOCR official Det, Rec, and System outputs, not AITrain C++ OCR ONNX postprocess smoke artifacts.

## Scope

Production OCR acceptance covers three paths:

- PaddleOCR Det: representative text detection training/export evidence.
- PaddleOCR Rec: representative text recognition training/export metrics.
- PaddleOCR System: end-to-end Det+Rec inference evidence on representative images.

The full-system acceptance path is official PaddleOCR `predict_system.py`. C++ OCR Det/Rec ONNX postprocess is not production acceptance evidence. PP-OCRv5 is the default route through official Det/Rec `modelPreset` values, PP-OCRv4 remains explicitly selectable for compatibility, and PP-OCRv6 `tiny/small/medium` Det/Rec presets are explicitly selectable; no separate `paddleocr_ppocrv5_*` or `paddleocr_ppocrv6_*` backend is introduced.

PP-OCRv5/PP-OCRv6 coverage in this phase is limited to PaddleOCR official Det, Rec, and System OCR. It does not cover PP-StructureV3, PP-ChatOCR, PaddleOCR-VL, document orientation classification, document unwarping, text-line orientation classification, or PaddleOCR C++ local deployment. PP-OCRv6 tiny follows the official language-coverage limitation and must not be used as a customer-domain production claim without customer data evidence.

## Data Requirements

Prepare data outside the repository, preferably under an ignored path such as `.deps\production-ocr-data` or on an external drive.

Minimum gate thresholds used by `tools\production-ocr-acceptance.ps1`:

| Evidence | Default minimum |
|---|---:|
| Det images | 100 |
| Rec labeled samples | 1000 |
| System images | 100 |
| Rec accuracy | > 0.70 |
| Rec CER | recorded, not blocking by default |

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

This public data pass is useful for exercising the production OCR acceptance flow, but it is not customer-domain production evidence. Public Total-Text, generated smoke, and `.deps` examples can prove that the official Det/Rec/System workflow runs; they must not be used as customer-domain production-readiness proof.

Recorded public workflow evidence is indexed in `docs\harness\current-status.md` and `docs\delivery-evidence-index.md`. In short:

- Historical CPU public-data OCR runs are blocked evidence when Rec quality does not meet the current gate.
- The RTX 4090 D public Total-Text workflow passed the current `accuracy>0.70` Rec gate in the recorded validation lane.
- That RTX/public result remains workflow evidence only; customer-domain OCR claims require real customer or target-domain data and returned official reports.

## Required Reports

The acceptance gate expects JSON report files:

- `paddleocr_official_det_report.json`
- `paddleocr_official_rec_report.json`
- `paddleocr_official_system_report.json`

The Rec report must expose `accuracy` or `acc`, either at top level or under `metrics`. `cer` or `CER` is recorded when present and becomes blocking only when `-RequireRecCer` is set.

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

To run the full local official chain from prepared or auto-prepared data, use:

```powershell
.\tools\run-production-ocr-official-chain.ps1 `
  -WorkDir .deps\production-ocr-official-chain `
  -DataDir .deps\production-ocr-data
```

To run the same chain with PP-OCRv5 presets, use:

```powershell
.\tools\run-production-ocr-official-chain.ps1 `
  -WorkDir .deps\production-ocr-official-chain-v5 `
  -DataDir .deps\production-ocr-data `
  -OcrVersion PP-OCRv5 `
  -DetModelPreset PP-OCRv5_mobile_det `
  -RecModelPreset en_PP-OCRv5_mobile_rec
```

To run the same chain with PP-OCRv6 presets, use:

```powershell
.\tools\run-production-ocr-official-chain.ps1 `
  -WorkDir .deps\production-ocr-official-chain-v6 `
  -DataDir .deps\production-ocr-data `
  -OcrVersion PP-OCRv6 `
  -PPOCRv6Tier medium `
  -AllowBlocked
```

For the PP-OCRv5 GPU gate, use the wrapper instead of relying on implicit device fallback:

```powershell
.\tools\phase50-paddleocr-v5-gpu-official-chain.ps1 -UseGpu
```

The wrapper blocks before training when PaddlePaddle is not CUDA-enabled. GPU mode is the default; `-UseGpu` is accepted as an explicit switch. A blocked GPU gate is valid diagnostic evidence; it must not be rewritten as a CPU pass.

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

Use Rec metric experiments only as iteration evidence. They are useful for improving public or customer-domain Rec quality, but they are not acceptance results until the full production OCR gate is rerun with Det, Rec, and System evidence.

To iterate on Rec metrics without rerunning Det and System evidence, use `tools\run-production-ocr-rec-experiment.ps1`. Each run writes an experiment directory under `.deps\production-ocr-rec-experiments` by default with the request JSON, reproducible command, official Rec report, and `rec_experiment_summary.json`.

Rules:

- Use `-UsePretrained -PretrainedModel <checkpoint-base>` only with a compatible PaddleOCR Rec checkpoint.
- Use `-ExportOnly` only to recover export/report artifacts from a completed official training run.
- GPU runs must use `-UseGpu`; the script must block before training if the selected Python/Paddle environment is not CUDA-enabled.
- After a qualifying Rec report is produced, rerun `tools\production-ocr-acceptance.ps1` or the full official chain to record gate status.
- Keep experiment outputs under `.deps` or another ignored work directory. Do not commit public data, customer data, checkpoints, inference models, or generated reports.

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
