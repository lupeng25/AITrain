# Production OCR Acceptance Result

## Summary

- Date:
- Tester:
- Machine / environment:
- AITrain source commit or package version:
- Acceptance status: passed / blocked
- `production_ocr_acceptance_report.json` path:
- `production_ocr_acceptance_summary.md` path:

## Current RTX 4090 Validation Snapshot

- Date: 2026-05-12
- Machine / environment: NVIDIA GeForce RTX 4090 D, GPU Paddle environment `.deps\rtx4090-validation\python-ocr-gpu`
- Acceptance status: passed
- Acceptance report: `.deps\rtx4090-validation\production-ocr-acceptance-gpu-chain\production_ocr_acceptance_report.json`
- Acceptance summary: `.deps\rtx4090-validation\production-ocr-acceptance-gpu-chain\production_ocr_acceptance_summary.md`
- Result: data-size checks, official Det report, official Rec metrics, official System report, and Phase 47 Det ONNX evidence passed.
- Rec metrics: `accuracy=0.71874997504340365`, `CER=0.14153062880387579`
- Thresholds: `accuracy>0.70`; CER is recorded and not blocking by default.
- Interpretation: the historical `acc==0` report is old CPU/default blocked evidence; the current RTX 4090 run passes the lowered Rec accuracy gate.

Command used:

```powershell
.\tools\production-ocr-acceptance.ps1 `
  -WorkDir .deps\rtx4090-validation\production-ocr-acceptance-gpu-chain `
  -DetDataset .deps\rtx4090-validation\production-ocr-data\det_dataset `
  -RecDataset .deps\rtx4090-validation\production-ocr-data\rec_dataset `
  -SystemImages .deps\rtx4090-validation\production-ocr-data\system_images `
  -OfficialDetReport .deps\rtx4090-validation\production-ocr-official-chain-gpu\reports\det_official\paddleocr_official_det_report.json `
  -OfficialRecReport .deps\rtx4090-validation\production-ocr-official-chain-gpu\reports\rec_official\paddleocr_official_rec_report.json `
  -OfficialSystemReport .deps\rtx4090-validation\production-ocr-official-chain-gpu\reports\system_official\paddleocr_official_system_report.json `
  -OcrDetOnnxSummary .deps\rtx4090-validation\phase47-paddleocr-det-onnx\paddleocr_det_onnx_smoke_summary.json `
  -RequireDetOnnxEvidence
```

## Dataset Evidence

Detection dataset:

- Path:
- Image count:
- Split summary:
- Data source / domain:
- Sensitive data handling notes:

Recognition dataset:

- Path:
- Labeled sample count:
- Dictionary path:
- Split summary:
- Data source / domain:

System images:

- Path:
- Image count:
- Data source / domain:

## Official PaddleOCR Reports

Detection:

- `paddleocr_official_det_report.json` path:
- Train/export status:
- Checkpoint path:
- Inference model path:

Recognition:

- `paddleocr_official_rec_report.json` path:
- Train/export status:
- Accuracy:
- CER:
- Checkpoint path:
- Inference model path:

System:

- `paddleocr_official_system_report.json` path:
- Prediction status:
- Prediction JSON path:
- Visualization directory:

## Optional C++ Det ONNX Evidence

- Required for this run: yes / no
- `paddleocr_det_onnx_smoke_summary.json` path:
- Status:
- ONNX path:
- Overlay / prediction artifacts:

## Command Output

Paste the command used:

```powershell

```

Paste the final console output:

```text

```

## Blockers

List any blocked checks exactly as reported by `production_ocr_acceptance_report.json`:

- 

## Attachments

- Acceptance report:
- Official Det report:
- Official Rec report:
- Official System report:
- Dataset manifest / snapshot:
- Representative system visualizations:
