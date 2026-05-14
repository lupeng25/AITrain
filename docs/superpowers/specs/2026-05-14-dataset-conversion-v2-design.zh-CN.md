# 数据集转换矩阵 v2 设计

日期：2026-05-14

状态：已批准进入实现计划

## 背景

AITrain Studio 当前已经具备本地训练工作台的主要闭环：数据集校验、划分、快照、质量检查、官方 YOLO/PaddleOCR 训练适配器、ONNX/TensorRT 推理路径、模型库、评估、benchmark、交付报告和验收流程。当前 harness 约束仍要求保持 Worker/core 边界，不把长任务放进 `MainWindow`，也不能把 scaffold 能力描述成真实生产训练能力。

现有数据集转换层已经支持：

- COCO JSON 转 YOLO detection。
- COCO polygon segmentation 转 YOLO segmentation。
- Pascal VOC XML 转 YOLO detection。

对照 Labelformat、Datumaro、X-AnyLabeling、FiftyOne、CVAT 等 GitHub 同类工具后，当前更值得补齐的不是新的训练后端，而是稳定的数据格式转换矩阵：支持反向导出、转换后校验，以及可定位问题的转换报告。

本 v2 设计把数据转换扩展到第一版稳定的 detection/segmentation 双向矩阵。范围限定为 core、Worker 和 tests，GUI 入口后置。

## 目标

- 新增 YOLO detection 到 COCO detection 的反向转换。
- 新增 YOLO detection 到 Pascal VOC XML 的反向转换。
- 新增 YOLO segmentation 到 COCO polygon segmentation 的反向转换。
- 保留现有 COCO/VOC 到 YOLO 的转换行为。
- 保留 `copyImages` 选项，并默认复制图片到输出数据集。
- 支持 `copyImages=false` 的引用式输出，并在报告中明确记录。
- 输出转换报告，记录转换样本数、跳过样本数、跳过标注数、类别映射、输出文件、图片复制/引用策略和目标校验结果。
- 产品侧转换通过 Worker 路由，但格式解析和写入逻辑放在 core。
- 用 focused QtTest 覆盖新增转换矩阵，并通过标准 harness gate。

## 非目标

- 本阶段不改 GUI。
- 本阶段不做 LabelMe 或 CVAT 导入。
- 本阶段不做 OCR/PaddleOCR 数据集转换。
- 不支持 classification、pose、OBB、anomaly、YOLO-World、YOLOE。
- 不支持 Pascal VOC segmentation mask。
- 不改 SQLite schema。
- 不改插件接口。
- 不新增训练、推理、导出、评估或 benchmark 算法。
- 第一版不引入 Datumaro 等 Python 转换依赖。

## 支持矩阵

| 源格式 | 目标格式 | v2 行为 |
|---|---|---|
| `coco_json` | `yolo_detection` | 保留现有 bbox 转换。 |
| `coco_json` | `yolo_segmentation` | 保留现有 polygon 转换；RLE 仍跳过并记录 issue。 |
| `voc_xml` | `yolo_detection` | 保留现有 VOC bbox 转换。 |
| `yolo_detection` | `coco_json` | 新增 COCO `images`、`categories` 和 bbox `annotations` 输出。 |
| `yolo_detection` | `voc_xml` | 新增 Pascal VOC XML 文件，包含 `filename`、`size` 和 `object/bndbox`。 |
| `yolo_segmentation` | `coco_json` | 新增 COCO polygon `segmentation` 输出。 |

`voc_xml` 到 `yolo_segmentation`、`yolo_segmentation` 到 `voc_xml` 以及所有 OCR 转换仍不支持，应明确返回 `unsupported_target_format` 或 `unsupported_source_format`。

## 架构

转换 API 继续围绕以下类型：

- `DatasetConversionRequest`
- `DatasetConversionResult`
- `convertDataset(...)`

当前实现位于 `src/core/src/DatasetConversion.cpp`。v2 可以继续保留小规模实现；如果文件继续膨胀，实施时应拆成私有 companion 文件：

- `DatasetConversionInternal.h`
- `DatasetConversionCoco.cpp`
- `DatasetConversionYolo.cpp`
- `DatasetConversionVoc.cpp`

除非报告字段确实受阻，否则不修改 public API。如果必须改 API，应保持向后兼容并补测试。

Worker 职责保持窄边界：

- 接收转换请求。
- 调用 core 转换 API。
- 发出进度、报告 artifact 和最终任务状态。
- 不直接解析标注格式。

Core 职责：

- 解析源数据集布局。
- 把样本、类别和标注规整为内部结构。
- 写出目标格式布局。
- 按 `copyImages` 复制或引用图片。
- 在存在 validator 时运行目标校验。
- 写出 `dataset_conversion_report.json`。

## 内部数据模型

v2 建议围绕一个小型内部表示实现，让每个格式 parser/writer 更简单且更容易测试：

- `ImageRecord`：稳定 image id、相对路径、源路径、宽、高、split。
- `ClassRecord`：数值 class id 和 class name。
- `AnnotationRecord`：image id、class id、bbox、polygon points、source file、可用时的 source line。
- `ConversionIssue`：沿用现有 issue 结构，包含 severity、code、source file、image path、category 和 message。

该模型仅供内部使用，不作为 public SDK 类型，也不改变插件接口。

Split 推断规则：

- YOLO 源数据集优先读取 `data.yaml`。
- 如果缺少 `data.yaml`，从 `images/train`、`images/val`、`labels/train`、`labels/val` 推断。
- 如果只有扁平 `images`/`labels` 布局，则把样本视为 `train`，并记录 warning issue。

类别名规则：

- 优先读取 `data.yaml` 的 `names`。
- 如果缺失，则生成稳定名称，例如 `class_0`、`class_1`，并记录 warning。
- 在目标格式允许时保留数值 class id。

## 输出规则

### YOLO 输出

现有 YOLO 输出行为保持不变：

- `data.yaml`
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

`val` 路径必须指向 `images/val`，不能指向 `images/train`。

### COCO 输出

Detection 和 segmentation 输出应写入：

- 单文件输出时写 `annotations.json`，或
- split 输出时写 `annotations/train.json` 和 `annotations/val.json`。

第一版应优先使用 split-aware 输出，因为 AITrain 训练流程已经基于 train/val 布局。报告字段必须列出生成的 annotation 文件。

Detection bbox 输出：

- COCO bbox 使用像素坐标 `[x, y, width, height]`。
- `area` 为 `width * height`。
- `iscrowd` 为 `0`。

Segmentation polygon 输出：

- YOLO 归一化 polygon 点转换回像素坐标。
- COCO `bbox` 为 polygon 外接框。
- `area` 可用 shoelace 公式计算；如果 polygon 无效，记录 issue 并跳过该标注。
- 不生成 RLE 输出。

### Pascal VOC XML 输出

VOC detection 输出应写入：

- `Annotations/<image-base>.xml`
- `JPEGImages/<image-name>`，当复制图片时写入。

每个 XML 文件应包含：

- `<annotation>`
- `<folder>`
- `<filename>`
- `<path>`，当可获得复制后的绝对路径时写入
- `<size><width/><height/><depth/></size>`
- 每个 bbox 一个 `<object>`，包含 `<name>` 和 `<bndbox>`

Bounding box 应 clamp 到图片边界。无效或零面积 bbox 应跳过并记录 issue。

## 图片复制与引用策略

`copyImages` 保持支持并默认 `true`。

当 `copyImages=true`：

- 输出数据集应自包含。
- 图片复制到目标布局。
- 可沿用当前安全复制行为替换已有目标文件。

当 `copyImages=false`：

- 转换器不修改源图片。
- COCO `file_name` 可尽量使用相对源路径。
- VOC 输出仍写 annotation XML，但不复制图片；报告字段必须说明图片是引用而非打包。
- YOLO 输出应尽量保持相对 `data.yaml` 的路径有效。如果无法保证，记录 warning issue。

## 错误处理

转换器应在非致命样本或标注问题上继续处理后续有效数据。

致命失败：

- 源路径不可读。
- 不支持的源/目标格式组合。
- 输出目录无法创建。
- 报告无法写入。
- 没有任何可转换样本或可转换标注。

非致命 issue：

- 图片缺失。
- label 缺失。
- label 为空。
- 未知 class id。
- bbox 无效。
- polygon 无效。
- 图片尺寸缺失。
- 不支持的 segmentation 编码。
- 输出目标文件名冲突。

每条 issue 应包含能获得的最具体 source file 和 image path。

## 报告字段增强

`dataset_conversion_report.json` 应包含现有结果字段，并在可用时增加这些 v2 字段：

- `conversionMatrixVersion`: `2`
- `copyImages`: boolean
- `imagePolicy`: `copied` 或 `referenced`
- `outputFiles`: 包含 annotation files、data yaml、image roots、label roots、XML roots
- `splitCounts`: 已知时记录 train/val/test 计数
- `sourceValidation`: 可选源布局摘要
- `targetValidation`: 现有目标校验对象

如果目标校验失败，报告不得声称数据集已经训练就绪。

## Worker 集成

如果现有 dataset conversion 路径已经完整，Worker 应通过该路径暴露 v2 矩阵；如果不完整，则新增 `convertDataset` 命令，并与 core request 对齐：

- `sourcePath`
- `sourceFormat`
- `targetFormat`
- `outputPath`
- `options.copyImages`

Worker 输出：

- 记录开始日志和选择的格式组合。
- 解析完成和写入完成后发出进度。
- 为 `dataset_conversion_report.json` 发出 artifact。
- 在有价值时为目标 annotation 文件发出 artifact。
- 失败时带上 `errorCode` 和 `errorMessage`。

不要把转换逻辑加到 GUI 或 Worker。

## 测试

Focused QtTest 覆盖应包含：

- `yoloDetectionConvertsToCoco`
- `yoloDetectionConvertsToVocXml`
- `yoloSegmentationConvertsToCocoPolygons`
- `copyImagesFalseKeepsReferencedPaths`
- `invalidYoloLabelsReportIssues`
- 现有 COCO detection 到 YOLO 测试
- 现有 COCO segmentation polygon 到 YOLO 测试
- 现有 COCO RLE skip 测试
- 现有 VOC XML 到 YOLO detection 测试

如果 Worker 路由有变更，至少增加一条 v2 反向转换 Worker 覆盖。

必须运行：

```powershell
.\tools\harness-check.ps1
```

## 实施顺序

1. 添加或抽取内部转换模型。
2. 添加 YOLO detection 和 segmentation 源 parser。
3. 添加 COCO detection 和 polygon segmentation writer。
4. 添加 VOC XML detection writer。
5. 保留并回归测试现有 COCO/VOC 到 YOLO 行为。
6. 增强报告字段。
7. 仅在当前 Worker conversion 入口不完整时扩展 Worker 路由。
8. 运行完整 harness gate。

## 验收标准

- 所有支持的格式组合都能生成符合预期的目标数据集。
- 无效样本会被记录，不阻塞有效样本转换。
- `copyImages=true` 生成自包含输出。
- `copyImages=false` 不复制图片，并在报告中明确说明引用行为。
- VOC 保持 detection-only，并对不支持的 segmentation 目标给出明确错误。
- 完整 harness check 通过。

## 后续延期项

- GUI 转换 workflow。
- LabelMe 和 CVAT 导入。
- 借鉴 Datumaro 的数据集合并、过滤、统计 workflow。
- 借鉴 FiftyOne 的难例样本队列增强。
- OCR/PaddleOCR 数据集转换。
- VOC segmentation mask 支持；仅在有明确客户工作流时再做。
