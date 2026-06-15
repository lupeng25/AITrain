# Production OCR Acceptance Result

## Summary

- Date:
- Tester:
- Machine / environment:
- AITrain source commit or package version:
- Acceptance status: passed / blocked
- `production_ocr_acceptance_report.json` path:
- `production_ocr_acceptance_summary.md` path:

## Current Evidence Reference

This file is a blank result template. Do not paste historical RTX, public Total-Text, generated-smoke, or `.deps` example results here as if they were the current run.

For already-recorded public/RTX workflow evidence, see:

- `docs\harness\current-status.md`
- `docs\delivery-evidence-index.md`

Customer-domain production OCR acceptance must use customer or target-domain data plus PaddleOCR official Det, Rec, and System reports. Public Total-Text and generated smoke data can prove workflow execution only.

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

## Historical Phase 47 Archive Context

- Historical Phase 47 OCR ONNX evidence path, if referenced only for archive context:
- Historical status:
- Historical ONNX path:
- Historical overlay / prediction artifacts:
- Current OCR acceptance note: C++ OCR ONNX evidence is historical wiring context only and must not be used as current OCR acceptance evidence.

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
