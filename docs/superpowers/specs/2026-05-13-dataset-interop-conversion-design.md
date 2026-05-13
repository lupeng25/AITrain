# Dataset Interop Conversion Design

Date: 2026-05-13

Status: Approved for implementation planning

## Context

AITrain Studio is now a local Windows + Qt Widgets computer-vision training workbench with Worker-managed training, SQLite metadata, dataset validation, dataset split, dataset snapshots, quality reports, sample review, model registry, evaluation, benchmark, export, inference validation, delivery acceptance, diagnostics, and customer OCR evidence workflows.

The current dataset training path accepts normalized AITrain layouts:

- YOLO detection
- YOLO segmentation
- PaddleOCR Det
- PaddleOCR Rec

The GUI already exposes labels for COCO JSON, VOC XML, and LabelMe JSON, but the current Worker-backed validation and split paths only promise the normalized formats above. GitHub comparison against CVAT, Label Studio, Datumaro, FiftyOne, ClearML, MLflow, Ultralytics, and PaddleOCR suggests the next useful product step is data-loop hardening, not more training backends.

This design adds a focused external-format conversion loop so imported data can become trainable AITrain datasets without introducing a new data platform or Python dependency.

## Goals

- Convert common external annotation formats into AITrain-supported YOLO datasets.
- Keep original source data read-only.
- Route conversion through `aitrain_worker`; keep conversion logic out of `MainWindow`.
- Reuse existing YOLO validation after conversion.
- Record conversion and validation outputs as existing task artifacts.
- Make conversion failures actionable with source file, image, category, and reason details.

## Non-Goals

- No new training algorithms or model backends.
- No SQLite schema changes.
- No Datumaro or other Python conversion backend in the first version.
- No OCR dataset inference from COCO, VOC, or LabelMe.
- No RLE mask to polygon conversion in the first version.
- No GUI-embedded annotation editor.
- No cloud dataset management, collaboration, or remote storage.

## Supported Format Matrix

| Source format | Target format | First-version behavior |
|---|---|---|
| COCO JSON | YOLO detection | Convert `bbox` annotations into YOLO bbox labels. |
| COCO JSON | YOLO segmentation | Convert polygon `segmentation` annotations into YOLO polygon labels. RLE masks are skipped with report issues. |
| Pascal VOC XML | YOLO detection | Convert `object/name/bndbox` into YOLO bbox labels. |
| Pascal VOC XML | YOLO segmentation | Not supported in the first version. |
| LabelMe JSON | YOLO detection | Convert `rectangle` shapes into bbox labels; optionally convert polygon shapes to enclosing bbox. |
| LabelMe JSON | YOLO segmentation | Convert polygon shapes into YOLO polygon labels. |

## Architecture

The feature is a Dataset Interop extension inside the existing architecture:

- GUI: adds conversion inputs and a start button on the current dataset workbench.
- Worker: adds a `convertDataset` command and emits progress, artifacts, conversion payload, validation payload, and terminal messages.
- Core: owns parsing, normalization, conversion report writing, and post-conversion validation.
- Repository: uses existing task and artifact records; no schema change.

The expected source ownership is:

- `src/core`: conversion API and implementation.
- `src/worker`: Worker command routing and artifact emission.
- `src/app`: form controls, task creation, result display, and output-path handoff.
- `tests`: focused conversion and Worker coverage.
- `docs`: user-facing and harness status updates.

## Data Flow

1. User selects source path, source format, target format, output directory, and conversion options.
2. GUI creates a repository task with `task_type=dataset_conversion` and plugin id `com.aitrain.plugins.dataset_interop`.
3. GUI sends `convertDataset` to the Worker.
4. Worker calls the core conversion API.
5. Core parses the source dataset and writes normalized YOLO output under the selected output directory.
6. Core writes `dataset_conversion_report.json`.
7. Core runs the existing target validator:
   - `validateYoloDetectionDataset` for YOLO detection.
   - `validateYoloSegmentationDataset` for YOLO segmentation.
8. Worker emits conversion and validation report artifacts.
9. Worker completes only when conversion succeeds and target validation passes.
10. GUI fills the converted output path back into the dataset path field and switches the format selector to the target format.

The default output directory is:

```text
<project>/datasets/normalized/<source-name>-<target-format>
```

When no project is open, the fallback is a sibling `normalized` directory near the source dataset, matching the existing split-output behavior.

## Conversion Rules

### COCO JSON

The converter reads standard `images`, `annotations`, and `categories`.

- `images[].id` maps annotations to image records.
- `images[].file_name`, `width`, and `height` define the sample.
- `categories` define class names.
- `annotations[].bbox` converts to YOLO detection labels.
- Polygon `annotations[].segmentation` converts to YOLO segmentation labels.
- RLE masks are skipped and recorded as non-blocking issues.
- An annotation with missing image id, missing category id, invalid bbox, or invalid polygon is skipped.

### Pascal VOC XML

The converter reads XML files from the selected directory.

- `filename` or `path` resolves the image.
- `size/width` and `size/height` define image dimensions.
- `object/name` defines the class.
- `object/bndbox` converts to YOLO detection labels.
- Missing size, missing image, empty class, and invalid bbox are recorded as issues.

### LabelMe JSON

The converter reads JSON files from the selected directory.

- `imagePath` resolves the image.
- `imageWidth` and `imageHeight` define dimensions.
- `shapes[].label` defines the class.
- `rectangle` shapes convert to YOLO detection labels.
- `polygon` shapes convert to YOLO segmentation labels.
- When the target is detection and `polygonToBox=true`, polygon shapes convert to enclosing bboxes and are marked in the report.
- Unsupported shape types are skipped with issues.

## Class Mapping

Class ids must be stable and reproducible.

- COCO uses source category id ordering after filtering to categories that appear in converted annotations.
- VOC and LabelMe use normalized class-name sorting.
- The final mapping is written into `data.yaml` and `dataset_conversion_report.json`.
- Empty class names are errors for the affected object.
- Duplicate names after normalization are reported as warnings and resolved to one output class.

## Output Layout

The converter writes a standard YOLO layout:

```text
output/
  data.yaml
  images/
    train/
  labels/
    train/
  dataset_conversion_report.json
  dataset_validation_report.json
```

The first version does not split converted data into train/val/test. It writes all converted samples into `train`, then users can run the existing dataset split flow to create train/val/test folders. This keeps conversion separate from split policy and avoids duplicating the existing split feature.

By default, images are copied into the output dataset. If `copyImages=false`, the converter may preserve relative references only when the target validator can resolve them. The recommended GUI default is `copyImages=true`.

## Worker Command

Request:

```json
{
  "command": "convertDataset",
  "taskId": "...",
  "sourcePath": "...",
  "sourceFormat": "coco_json",
  "targetFormat": "yolo_detection",
  "outputPath": "...",
  "options": {
    "copyImages": true,
    "allowEmptyLabels": false,
    "polygonToBox": true,
    "maxIssues": 500
  }
}
```

Expected Worker messages:

- `progress`
- `artifact` with `kind=dataset_conversion_report`
- `artifact` with `kind=dataset_validation_report`
- `datasetConversion`
- `datasetValidation`
- `completed` or `failed`

The Worker should fail the task when conversion is blocked or when post-conversion validation fails.

## Report Contract

`dataset_conversion_report.json` includes:

- `sourceFormat`
- `targetFormat`
- `sourcePath`
- `outputPath`
- `convertedAt`
- `sampleCount`
- `convertedSampleCount`
- `skippedSampleCount`
- `annotationCount`
- `convertedAnnotationCount`
- `skippedAnnotationCount`
- `classMap`
- `issues`
- `targetValidation`
- `artifacts`

Each issue includes:

- `severity`: `error`, `warning`, or `info`
- `code`
- `sourceFile`
- `imagePath`
- `category`
- `message`

The report must distinguish blocking failures from non-blocking skipped samples or annotations.

## Error Handling

Blocking failures:

- Source path does not exist.
- Source JSON or XML cannot be parsed.
- No convertible samples remain.
- Output directory cannot be created or written.
- Target validation fails after conversion.

Non-blocking issues:

- Missing image for one sample.
- Invalid object bbox.
- Invalid or unsupported polygon.
- COCO RLE mask.
- Empty object class.
- Unsupported LabelMe shape type.

Non-blocking issues are written to the report and may reduce converted counts. They do not mark the Worker task failed unless they leave no usable converted dataset or target validation fails.

## GUI Design

Add a compact conversion section to the existing dataset workbench rather than creating a new page.

Controls:

- Source path selector.
- Source format combo: COCO JSON, VOC XML, LabelMe JSON.
- Target format combo: YOLO detection, YOLO segmentation.
- Output directory field.
- Options:
  - Copy images.
  - Allow empty labels.
  - Convert LabelMe polygons to detection boxes.
- Start button: `转换为训练数据集`.

After successful conversion:

- Fill the dataset path field with the output directory.
- Set dataset format to the target format.
- Show a conversion summary.
- Keep the normal validate, split, snapshot, quality report, and training paths unchanged.

The 1280x820 workbench layout must remain usable. Conversion controls should not push current validate, split, snapshot, quality, or X-AnyLabeling actions out of reach.

## Testing

Core tests:

- Minimal COCO bbox to YOLO detection.
- Minimal COCO polygon to YOLO segmentation.
- COCO RLE is skipped and reported.
- Minimal VOC XML to YOLO detection.
- Minimal LabelMe rectangle to YOLO detection.
- Minimal LabelMe polygon to YOLO segmentation.
- LabelMe polygon to detection bbox when `polygonToBox=true`.
- Missing images, invalid bbox, invalid polygon, and empty class reporting.
- Converted output passes existing YOLO validation.

Worker tests:

- `convertDataset` emits conversion and validation report artifacts.
- Successful conversion sends `datasetConversion`, `datasetValidation`, and `completed`.
- Blocking failures send `failed` with a clear message.
- Post-conversion validation failure sends `failed` and preserves the conversion report.

GUI verification:

- Conversion controls fit the current dataset page at 1280x820.
- Successful conversion fills the dataset path and target format.
- Report paths and long output paths do not cause horizontal overflow.

## Documentation Updates

Update these files during implementation:

- `docs/user-guide.md`
- `docs/training-backends.md`
- `docs/harness/current-status.md`
- `docs/product-roadmap-local-training-platform.md`

The docs must keep the boundary explicit: converted data becomes YOLO detection or YOLO segmentation training input; this feature does not add OCR conversion, new training backends, or Datumaro support.

## Acceptance

For code changes, run:

```powershell
.\tools\harness-check.ps1
```

For UI changes, also run a 1280x820 Qt walkthrough over the dataset page and adjacent workflow pages.

Manual acceptance:

1. Convert a small COCO detection dataset.
2. Convert a small COCO polygon segmentation dataset.
3. Convert a small VOC detection dataset.
4. Convert LabelMe rectangle and polygon examples.
5. Confirm conversion reports are recorded as task artifacts.
6. Confirm converted datasets can be validated and split through the existing workflow.
7. Confirm training defaults still choose official YOLO detection or segmentation backends after selecting the converted dataset.

## Implementation Boundaries

Keep these constraints in force:

- Training, evaluation, export, inference, benchmark, conversion, and report logic stay outside `MainWindow`.
- Long-running conversion work runs in `aitrain_worker`.
- `ProjectRepository` remains the SQLite access layer.
- Source files compile as UTF-8.
- UI text uses `QStringLiteral`.
- Scaffold and official-backend boundaries remain explicit.
