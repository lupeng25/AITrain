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
  paddleocr_rec/
    dict.txt
    rec_gt.txt
    images/a.png
    images/b.png
```

The datasets are intentionally tiny. They validate file formats, trainer routing, artifact creation, and small CPU smoke runs; they are not intended for meaningful model quality.
