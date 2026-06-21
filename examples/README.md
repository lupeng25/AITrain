# AITrain Studio Minimal Datasets

This folder contains a generator for tiny datasets used to smoke-test the real Python trainer backends.

Generate all examples:

```powershell
python examples\create-minimal-datasets.py --output .deps\examples-smoke
```

Generated layout:

```text
examples-smoke/
  yolo_detect/
    data.yaml
    images/train/a.png
    images/val/b.png
    labels/train/a.txt
    labels/val/b.txt
  yolo_segment/
    data.yaml
    images/train/a.png
    images/val/b.png
    labels/train/a.txt
    labels/val/b.txt
  semantic_mask/
    classes.txt
    images/train/a.png
    images/val/b.png
    masks/train/a.png
    masks/val/b.png
  paddleocr_rec/
    dict.txt
    rec_gt.txt
    images/a.png
    images/b.png
  paddleocr_det/
    det_gt.txt
    images/a.png
    images/b.png
  requests/
    yolo_detect_request.json
    yolo_segment_request.json
    smp_semantic_request.json
    paddleocr_det_official_request.json
    paddleocr_rec_official_request.json
    paddleocr_system_official_request.json
```

The datasets are intentionally tiny. They validate file formats, trainer routing, artifact creation, and small CPU smoke runs; they are not intended for meaningful model quality.

`paddleocr_det/` uses the native PaddleOCR detection label format:

```text
relative/image.png<TAB>[{"transcription":"text","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
```

The generated SMP semantic segmentation request targets `smp_semantic_segmentation` over `semantic_mask`. The generated official PaddleOCR Det and System request JSON files default to `prepareOnly=true`, so they are suitable for Worker routing and artifact smoke tests without requiring a PaddleOCR checkout. The full official train/export/system inference chain is covered by `tools\phase31-paddleocr-full-official-smoke.ps1`.
