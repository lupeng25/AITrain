# 数据集格式转换指南

最后更新：2026-05-15

本文说明 AITrain Studio 当前已实现的数据集格式转换能力、GUI 使用流程、输出产物和限制。阶段状态以 `docs/harness/current-status.md` 为准。

## 当前支持矩阵

GUI 只暴露已经接入 Worker 和核心转换实现的组合：

| 源格式 | 目标格式 | 状态 | 说明 |
|---|---|---|---|
| COCO JSON | YOLO Detection | 支持 | 生成 YOLO 检测布局和 `data.yaml`。 |
| COCO JSON | YOLO Segmentation | 支持 | 使用 COCO segmentation 多边形生成 YOLO 分割标签。 |
| Pascal VOC XML | YOLO Detection | 支持 | VOC 只支持检测框转换，不支持分割。 |
| YOLO Detection | COCO JSON | 支持 | 按 split 生成 COCO annotations JSON。 |
| YOLO Detection | Pascal VOC XML | 支持 | 生成 `Annotations` 和可选 `JPEGImages`。 |
| YOLO Segmentation | COCO JSON | 支持 | 转为 COCO segmentation JSON。 |

明确不支持或未实现：

- YOLO Segmentation -> Pascal VOC XML：Pascal VOC XML 不承载当前分割多边形输出。
- LabelMe JSON：核心入口会返回 `not_implemented`，GUI 当前不作为可选格式暴露。
- PaddleOCR Det / Rec：当前转换矩阵不转换 OCR 标注。
- 分类、姿态、OBB、异常检测、YOLO-World、YOLOE：不在当前矩阵内。

## GUI 流程

1. 打开“数据集”页。
2. 在“格式转换”区域选择源格式。
3. 选择目标格式。目标下拉框会按源格式自动过滤为可支持组合。
4. 选择输入路径：
   - COCO JSON：选择 COCO annotation JSON。
   - Pascal VOC XML：选择 XML 文件或包含 XML 的目录。
   - YOLO：选择包含 `data.yaml` 或标准 `images` / `labels` 的数据集根目录。
5. 选择输出目录。建议使用新的空目录。
6. 点击“开始转换”。
7. 在进度条、日志、任务列表和“任务与产物”页查看状态。
8. 转换完成后打开输出目录和 `dataset_conversion_report.json`。
9. 如需训练，手动选择或导入转换后的输出目录，再运行数据集校验。

转换由 `aitrain_worker` 执行。GUI 只负责表单预检、任务调度、进度/日志展示和结果渲染，不在 GUI 线程里执行长任务。

## 输出产物

转换成功后，输出目录通常包含：

- 目标格式数据文件，例如 `data.yaml`、`labels`、`images`、`annotations_*.json` 或 `Annotations`。
- `dataset_conversion_report.json`。
- 可选复制后的图片目录。未复制图片时，输出标签会引用源图片路径或相对路径，迁移包时要特别检查。

`dataset_conversion_report.json` 主要字段：

| 字段 | 含义 |
|---|---|
| `ok` | 转换是否成功。 |
| `sourceFormat` / `targetFormat` | 源格式与目标格式。 |
| `sourcePath` / `outputPath` | 输入与输出路径。 |
| `sampleCount` / `convertedSampleCount` / `skippedSampleCount` | 样本总数、已转换数、跳过数。 |
| `annotationCount` / `convertedAnnotationCount` / `skippedAnnotationCount` | 标注总数、已转换数、跳过数。 |
| `classMap` | 类别映射。 |
| `copyImages` / `imagePolicy` | 图片复制策略。 |
| `outputFiles` | 关键输出文件路径。 |
| `sourceValidation` | 源数据读取和统计信息。 |
| `targetValidation` | 目标数据集校验结果。 |
| `issues` | 警告或错误列表。 |

## 质量边界

- 转换通过不等于训练可用。训练前仍必须在“数据集”页对转换后的输出目录重新运行校验。
- 转换产物不会自动注册为项目数据集，也不会自动替换当前选中数据集。
- 建议把转换报告作为交付证据保存，尤其是跨工具或跨团队移交时。
- 如果报告里有 `skipped*` 计数或 warning，需要人工确认是否可接受。
- COCO / VOC / YOLO 的类别命名、图片路径和 split 语义来自源数据；转换不会自动修复错误标注。

## 常见问题

### 为什么转换后没有自动进入训练？

转换是中间处理步骤。为了避免误用半成品，AITrain Studio 要求用户手动选择转换输出目录并重新校验。

### 为什么 VOC 不能输出分割？

当前 Pascal VOC XML 输出只覆盖检测框。分割数据建议输出到 COCO JSON 或 YOLO Segmentation。

### 为什么 LabelMe 不在界面里？

LabelMe parser 当前未实现。若通过低层接口强行传入 `labelme_json`，结果会返回 `not_implemented`。

### 能否转换 OCR 数据？

当前不能。PaddleOCR Det / Rec 仍使用各自的原生数据布局和校验流程。
