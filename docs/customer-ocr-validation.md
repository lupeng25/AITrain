# Customer OCR Validation

This gate separates customer-domain OCR evidence from public or generated smoke evidence.

Use it after the official PaddleOCR Det, Rec, and System chain has produced reports on a customer dataset. The script writes:

- `customer_ocr_validation_manifest.json`
- `customer_ocr_validation_summary.md`

Example:

```powershell
.\tools\customer-ocr-validation.ps1 `
  -CustomerDataset D:\customer-data\ocr-acceptance `
  -DetReport D:\runs\det\paddleocr_official_det_report.json `
  -RecReport D:\runs\rec\paddleocr_official_rec_report.json `
  -SystemReport D:\runs\system\paddleocr_official_system_report.json `
  -OutputDir .deps\customer-ocr-validation
```

Default gate:

- Customer dataset directory exists.
- Dataset path does not look like public/generated smoke data.
- Det, Rec, and System reports are attached.
- Rec `accuracy >= 0.70`.
- Rec `CER <= 0.30`.

Use `-AllowBlocked` when collecting blocked evidence without failing the shell step. Do not use public Total-Text or generated smoke results to claim customer-domain readiness.
