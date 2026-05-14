# Dataset Conversion Matrix v2 Design

Date: 2026-05-14

Status: Approved for implementation planning

## Context

AITrain Studio already has a harness-first local training workbench with dataset validation, split, snapshots, quality checks, official YOLO/PaddleOCR training adapters, ONNX/TensorRT inference paths, model registry, evaluation, benchmark, delivery reports, and acceptance workflows. Current harness guidance still says to preserve Worker/core boundaries, keep long-running work out of `MainWindow`, and avoid claiming scaffold paths as real production training.

The dataset conversion layer has initial external-format support:

- COCO JSON to YOLO detection.
- COCO polygon segmentation to YOLO segmentation.
- Pascal VOC XML to YOLO detection.

Recent GitHub comparison against dataset tooling such as Labelformat, Datumaro, X-AnyLabeling, FiftyOne, and CVAT shows that the next useful product gap is not another training backend. The gap is a reliable conversion matrix with reverse exports, validation, and actionable conversion reports.

This v2 design extends dataset conversion to cover the first stable bidirectional detection/segmentation matrix while keeping scope limited to core, Worker, and tests. GUI exposure is deferred.

## Goals

- Add reverse conversion from YOLO detection to COCO detection.
- Add reverse conversion from YOLO detection to Pascal VOC XML.
- Add reverse conversion from YOLO segmentation to COCO polygon segmentation.
- Preserve the existing COCO/VOC to YOLO behavior.
- Keep `copyImages` available and default to copying images into the output dataset.
- Support reference-style output when `copyImages=false`, with clear report fields.
- Emit a conversion report that explains converted samples, skipped samples, skipped annotations, class map, output files, image copy/reference policy, and target validation.
- Route product-facing conversion through Worker commands, but keep parsing and writing logic in core.
- Cover the new matrix with focused QtTest tests and the standard harness gate.

## Non-Goals

- No GUI changes in this phase.
- No LabelMe or CVAT import in this phase.
- No OCR/PaddleOCR dataset conversion in this phase.
- No classification, pose, OBB, anomaly, YOLO-World, or YOLOE support.
- No Pascal VOC segmentation mask support.
- No SQLite schema changes.
- No plugin interface changes.
- No new training, inference, export, evaluation, or benchmark algorithms.
- No Python conversion dependency such as Datumaro in the first version.

## Supported Matrix

| Source format | Target format | v2 behavior |
|---|---|---|
| `coco_json` | `yolo_detection` | Existing bbox conversion remains supported. |
| `coco_json` | `yolo_segmentation` | Existing polygon conversion remains supported; RLE remains skipped with issues. |
| `voc_xml` | `yolo_detection` | Existing VOC bbox conversion remains supported. |
| `yolo_detection` | `coco_json` | New COCO `images`, `categories`, and bbox `annotations` output. |
| `yolo_detection` | `voc_xml` | New Pascal VOC XML files with `filename`, `size`, and `object/bndbox`. |
| `yolo_segmentation` | `coco_json` | New COCO polygon `segmentation` output. |

`voc_xml` to `yolo_segmentation`, `yolo_segmentation` to `voc_xml`, and all OCR conversions remain unsupported and should fail with `unsupported_target_format` or `unsupported_source_format` as appropriate.

## Architecture

The conversion API remains centered on:

- `DatasetConversionRequest`
- `DatasetConversionResult`
- `convertDataset(...)`

The current implementation is in `src/core/src/DatasetConversion.cpp`. v2 may keep a small implementation there, but if the file grows further, implementation should be split into private companion files:

- `DatasetConversionInternal.h`
- `DatasetConversionCoco.cpp`
- `DatasetConversionYolo.cpp`
- `DatasetConversionVoc.cpp`

The public API should not change unless a missing field blocks reporting. If an API change is required, it must remain backward compatible and be covered by tests.

Worker responsibilities stay narrow:

- Accept conversion requests.
- Call the core conversion API.
- Emit progress, report artifacts, and terminal task status.
- Avoid parsing annotation formats directly.

Core responsibilities:

- Parse source dataset layout.
- Normalize samples/classes/annotations into an internal representation.
- Write the requested target layout.
- Copy or reference images based on `copyImages`.
- Run target validation when a validator exists.
- Write `dataset_conversion_report.json`.

## Internal Dataset Model

Implement v2 around a small internal representation so each format parser/writer is simpler and testable:

- `ImageRecord`: stable image id, relative path, source path, width, height, split.
- `ClassRecord`: numeric class id and class name.
- `AnnotationRecord`: image id, class id, bbox, polygon points, source file, source line when available.
- `ConversionIssue`: existing issue shape with severity, code, source file, image path, category, and message.

The model is internal only. It is not a public SDK type and does not change plugin interfaces.

Split inference:

- YOLO source should read `data.yaml` when present.
- If `data.yaml` is missing, infer from `images/train`, `images/val`, `labels/train`, and `labels/val`.
- If only a flat `images`/`labels` layout exists, treat samples as `train` and record a warning issue.

Class names:

- Prefer `data.yaml` `names`.
- If missing, generate stable names such as `class_0`, `class_1`, and record a warning.
- Preserve numeric class ids in the generated target where the target format allows it.

## Output Rules

### YOLO output

Existing YOLO output behavior remains:

- `data.yaml`
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

The `val` path must point to `images/val`, not `images/train`.

### COCO output

Detection and segmentation output should write:

- `annotations.json` for single-file output, or
- `annotations/train.json` and `annotations/val.json` when split output is enabled.

The first implementation should prefer split-aware output because AITrain training workflows already use train/val layout. Report fields must list the generated annotation files.

Detection bbox output:

- COCO bbox uses `[x, y, width, height]` in pixel coordinates.
- `area` is `width * height`.
- `iscrowd` is `0`.

Segmentation polygon output:

- YOLO normalized polygon points are converted back to pixel coordinates.
- COCO `bbox` is the enclosing box of the polygon.
- `area` can be polygon area using the shoelace formula; if invalid, record an issue and skip the annotation.
- RLE output is not generated.

### Pascal VOC XML output

VOC detection output should write:

- `Annotations/<image-base>.xml`
- `JPEGImages/<image-name>` when copying images

Each XML file should include:

- `<annotation>`
- `<folder>`
- `<filename>`
- `<path>` when an absolute copied path is available
- `<size><width/><height/><depth/></size>`
- one `<object>` per bbox with `<name>` and `<bndbox>`

Bounding boxes should be clamped to image bounds. Invalid or zero-area boxes are skipped with issues.

## Image Copy and Reference Policy

`copyImages` stays supported and defaults to `true`.

When `copyImages=true`:

- Output datasets should be self-contained.
- Images are copied to the target layout.
- Existing target files may be replaced using the current safe copy behavior.

When `copyImages=false`:

- The converter should not mutate source images.
- COCO `file_name` may reference source-relative paths when practical.
- VOC output should still write annotation XML, but image files are not copied; report fields must state that images are referenced, not packaged.
- YOLO output should keep paths valid relative to the generated `data.yaml` when possible. If not possible, report a warning issue.

## Error Handling

The converter should continue past non-fatal sample and annotation problems.

Fatal failures:

- Unreadable source path.
- Unsupported source/target pair.
- Output directory cannot be created.
- Report cannot be written.
- No convertible samples or no convertible annotations.

Non-fatal issues:

- Missing image.
- Missing label.
- Empty label.
- Unknown class id.
- Invalid bbox.
- Invalid polygon.
- Missing image dimensions.
- Unsupported segmentation encoding.
- Duplicate output target name collision.

Every issue should include the most specific source file and image path available.

## Report Additions

`dataset_conversion_report.json` should include the existing result fields plus these v2 fields when available:

- `conversionMatrixVersion`: `2`
- `copyImages`: boolean
- `imagePolicy`: `copied` or `referenced`
- `outputFiles`: object containing annotation files, data yaml, image roots, label roots, and XML roots
- `splitCounts`: train/val/test counts when known
- `sourceValidation`: optional source-layout summary
- `targetValidation`: existing target validation object

The report should avoid claiming training readiness if target validation fails.

## Worker Integration

Worker should expose the v2 matrix through the existing dataset conversion path if one already exists. If not, add a `convertDataset` command that mirrors the core request:

- `sourcePath`
- `sourceFormat`
- `targetFormat`
- `outputPath`
- `options.copyImages`

Worker output:

- log start and selected matrix pair
- progress after parse and after write
- artifact for `dataset_conversion_report.json`
- artifact for target annotation files when useful
- failed status with `errorCode` and `errorMessage`

Do not add conversion logic to GUI or Worker.

## Testing

Focused QtTest coverage should include:

- `yoloDetectionConvertsToCoco`
- `yoloDetectionConvertsToVocXml`
- `yoloSegmentationConvertsToCocoPolygons`
- `copyImagesFalseKeepsReferencedPaths`
- `invalidYoloLabelsReportIssues`
- existing COCO detection to YOLO test
- existing COCO segmentation polygon to YOLO test
- existing COCO RLE skip test
- existing VOC XML to YOLO detection test

Worker coverage should verify at least one v2 reverse conversion request if Worker routing changes.

Required verification:

```powershell
.\tools\harness-check.ps1
```

## Implementation Order

1. Add or extract the internal conversion model.
2. Add YOLO source parser for detection and segmentation.
3. Add COCO writer for detection and polygon segmentation.
4. Add VOC XML writer for detection.
5. Preserve and regression-test existing COCO/VOC to YOLO behavior.
6. Extend reports with v2 fields.
7. Add Worker routing only if current Worker conversion entry point is incomplete.
8. Run the full harness gate.

## Acceptance Criteria

- All supported matrix pairs produce target datasets with expected labels/annotations.
- Invalid samples are reported without stopping valid samples from converting.
- `copyImages=true` produces self-contained output.
- `copyImages=false` avoids image copies and reports reference behavior clearly.
- VOC remains detection-only and rejects unsupported segmentation targets clearly.
- Full harness check passes.

## Deferred Follow-Up

- GUI conversion workflow.
- LabelMe and CVAT import.
- Dataset merge/filter/statistics workflow inspired by Datumaro.
- Sample review enhancements inspired by FiftyOne-style difficult sample queues.
- OCR/PaddleOCR dataset conversion.
- VOC segmentation mask support, only if a concrete customer workflow requires it.
