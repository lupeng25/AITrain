# 数据集转换 GUI 入口设计

日期：2026-05-14

状态：设计已确认，等待规格审阅。

## 背景

AITrain Studio 已经具备数据集校验、划分、快照、质量报告和样本预览等数据集工作流。当前数据集转换能力已经通过 core API 和 Worker 命令存在，支持 COCO JSON、Pascal VOC XML、YOLO detection 和 YOLO segmentation 之间的已实现转换矩阵，但 GUI 还没有直接入口。

本设计把数据集转换作为数据集页的一个轻量操作入口接入现有 Qt Widgets 工作台。GUI 只负责表单、预检、Worker 调用和结果展示，不解析标注格式，不把转换逻辑放进 `MainWindow`，也不新增数据库 schema。

## 用户目标

- 在数据集页直接把一个已支持格式的数据集转换成另一个已支持格式。
- 在点击转换前发现基础输入错误，例如目录不存在、目标格式不可用、输出目录不可写或输入输出相同。
- 转换运行时看到当前进度、日志和取消入口。
- 转换完成后看到输出路径、转换报告摘要和 Worker 日志。
- v1 不自动把转换结果登记到项目数据集库，避免用户误把未校验产物加入训练流程。

## 范围

### 目标

- 在“数据集操作”区内联新增“格式转换”表单。
- 只展示当前已实现的转换格式和有效转换组合。
- 源格式选择后联动过滤目标格式。
- 支持输入目录、输出目录、源格式、目标格式四个必填项。
- 点击转换前执行轻量预检。
- 预检失败时在表单顶部显示错误摘要，并在对应字段下显示具体原因。
- 通过现有 `WorkerClient::requestDatasetConversion(...)` 调用 Worker。
- 通过现有 `WorkerSession::convertDataset(...)` 和 `aitrain::convertDataset(...)` 完成实际转换。
- 转换过程在表单内显示进度、日志和取消按钮。
- 转换完成后只显示输出路径、报告路径、摘要和日志，不自动写入 `ProjectRepository` 数据集表。

### 非目标

- 不新增 LabelMe、CVAT、OCR、classification、pose、OBB 或 anomaly 数据集转换。
- 不新增真实训练、推理或导出能力。
- 不修改 core 转换矩阵语义。
- 不新增专门的转换历史表，不把页面日志写入数据库。
- 不自动登记转换产物为项目数据集。
- 不新增独立顶层页面、弹窗向导或矩阵式高级页面。
- 不把长任务放进 GUI 线程。

## 支持格式和组合

v1 GUI 只展示已实现格式：

| 源格式 | 可选目标格式 |
|---|---|
| `coco_json` | `yolo_detection`, `yolo_segmentation` |
| `voc_xml` | `yolo_detection` |
| `yolo_detection` | `coco_json`, `voc_xml` |
| `yolo_segmentation` | `coco_json` |

格式显示名应面向用户：

| 格式 ID | 显示名 |
|---|---|
| `coco_json` | COCO JSON |
| `voc_xml` | Pascal VOC XML |
| `yolo_detection` | YOLO Detection |
| `yolo_segmentation` | YOLO Segmentation |

不支持的组合不在目标格式下拉框中出现。若后续通过代码路径触发了不支持组合，仍应由 Worker/core 返回失败信息，GUI 不把它伪装成可用功能。

## UI 结构

入口放在 `buildDatasetPage()` 的“数据集操作”面板内，位置在现有数据集目录、格式、划分参数和操作按钮附近。它应和现有 `InfoPanel`、`ActionStrip`、`QFormLayout` 风格一致，不新增与工作台风格不一致的大型卡片或向导。

建议结构：

- 小标题：`格式转换`
- 顶部状态标签：默认显示“选择源格式、目标格式和输出目录后开始转换。”
- 表单字段：
  - 源格式：`QComboBox`
  - 目标格式：`QComboBox`，随源格式联动
  - 输入目录：`QLineEdit` + 选择按钮
  - 输出目录：`QLineEdit` + 选择按钮
- 操作区：
  - `转换数据集` 按钮
  - `取消转换` 按钮，只有转换运行中可用
- 反馈区：
  - `QProgressBar`
  - `QPlainTextEdit` 日志，当前页面显示即可，不落库
  - 输出路径/报告路径摘要标签
- 字段级错误：
  - 可用 `QLabel` 放在相关字段下方，默认隐藏
  - 顶部状态标签显示错误摘要

默认输入目录可以复用当前数据集目录。若用户在数据集库表格中选择了数据集，转换输入目录同步到该路径，源格式同步到当前格式。

## 预检规则

点击转换前执行轻量预检：

1. 源格式不能为空。
2. 目标格式不能为空。
3. 源格式和目标格式必须是 GUI 支持矩阵中的有效组合。
4. 输入目录不能为空，且必须存在。
5. 输出目录不能为空。
6. 输出目录必须不同于输入目录，比较时使用规范化后的绝对路径。
7. 输出目录的父目录必须存在，或者 GUI 能够创建输出目录。
8. 输出目录不可写时阻止转换。
9. Worker 正在运行时阻止启动新转换，并提示等待当前任务完成。

预检失败时不启动 Worker。顶部摘要写出首要问题，例如“请修正 2 个字段后再转换。”字段下方写具体原因，例如“输出目录不能与输入目录相同。”

## 运行和取消

转换启动后：

- 禁用源格式、目标格式、输入目录、输出目录和转换按钮。
- 启用取消按钮。
- 进度条归零。
- 日志区域追加“开始转换数据集。”
- 如复用 `createRepositoryTask(...)` 创建任务记录，只记录任务状态和产物引用；任务类型应使用现有最接近的数据集/校验类任务，不新增 schema，不保存页面日志。
- 调用 `WorkerClient::requestDatasetConversion(...)`。

取消沿用现有 Worker 取消路径。GUI 不直接终止 core 转换逻辑，只请求 Worker 取消；若 Worker 需要进程级终止，由现有 `WorkerClient` 机制处理。

转换结束后：

- 恢复表单可编辑。
- 禁用取消按钮。
- 成功时显示输出目录、报告路径、转换样本数和跳过数。
- 失败时显示 Worker/core 返回的 `errorCode` 和 `errorMessage`。
- 不调用 `ProjectRepository::upsertDatasetValidation(...)`。
- 不自动刷新数据集库为“已登记数据集”；用户如需使用产物，应手动选择输出目录并执行校验。

## 消息处理

现有 Worker 已发送 `datasetConversion` 消息。GUI 需要新增或补齐处理分支：

- 在 `MainWindow::handleWorkerMessage(...)` 中识别 `datasetConversion`。
- 将 payload 交给新的 `updateDatasetConversionResult(...)` 或同等小函数。
- 从 payload 中读取：
  - `ok`
  - `sourceFormat`
  - `targetFormat`
  - `sourcePath`
  - `outputPath`
  - `reportPath`
  - `sampleCount`
  - `convertedSampleCount`
  - `skippedSampleCount`
  - `annotationCount`
  - `convertedAnnotationCount`
  - `skippedAnnotationCount`
  - `errorCode`
  - `errorMessage`
  - `issues`
- 将 payload 以格式化 JSON 追加或显示在转换日志区域，便于定位转换问题。

全局 `progress` 消息仍可更新顶部状态栏和现有进度条；转换表单内的进度条也应在当前任务为转换任务时同步更新。

## 文件边界

预计实现会触及：

- `src/app/src/MainWindowDatasetPage.cpp`
  - 新增转换表单控件创建和布局。
  - 保持页面仍是单一数据集页，不新增顶层页面。
- `src/app/src/MainWindow.h`
  - 新增转换表单控件成员、槽函数声明和结果更新函数声明。
- `src/app/src/MainWindowActions.cpp`
  - 新增浏览输入目录、浏览输出目录、启动转换、轻量预检、取消转换相关行为。
- `src/app/src/MainWindowWorkerMessages.cpp`
  - 新增 `datasetConversion` 消息处理和结果展示。
- `src/app/src/MainWindowSupport.cpp` 或 `MainWindowSupport.h`
  - 如有必要，新增格式显示名、转换矩阵、路径规范化或错误文本帮助函数。
- `tests/`
  - 若现有 GUI 单元测试覆盖不足，应至少补 Worker/client 或 helper 层测试，验证转换请求 payload、联动矩阵和预检规则。

不应修改：

- `src/core/src/DatasetConversion.cpp` 的转换语义，除非 GUI 发现已有 API 明确缺口。
- SQLite schema。
- 插件接口。
- 模型训练、推理、导出页面。

## 错误处理

- 表单级错误：顶部状态标签显示摘要。
- 字段级错误：相关字段下方显示具体原因。
- Worker 启动失败：表单状态恢复，日志写入启动失败原因，并弹出关键错误。
- Worker 转换失败：不弹出阻塞式错误；在结果区显示失败摘要、错误码和错误信息。
- 转换报告存在时提供报告路径；报告不存在时只显示 Worker payload。
- 输入或输出路径显示使用 `QDir::toNativeSeparators(...)`，内部比较使用 `QDir::cleanPath(...)` 和绝对路径。

## 测试和验证

实施时应覆盖：

- 源格式切换后目标格式列表正确过滤。
- 无输入目录、无输出目录、输入目录不存在、输入输出相同、无效组合时预检失败且不启动 Worker。
- 有效表单会调用 `WorkerClient::requestDatasetConversion(...)`，payload 包含 sourcePath、outputPath、sourceFormat、targetFormat 和 options。
- 收到成功 payload 后显示输出目录和转换摘要。
- 收到失败 payload 后显示错误码和错误信息。
- Worker 正在运行时不能启动第二个转换任务。

最终验证命令：

```powershell
.\tools\harness-check.ps1
```

如 UI 发生明显布局变化，还应运行 Qt GUI walkthrough 或等效截图检查，确认移动尺寸和桌面尺寸下文本不重叠、不出现横向滚动。

## 验收标准

- 用户可以在数据集页完成一次有效数据集转换请求。
- GUI 不展示未实现格式。
- 目标格式随源格式联动过滤。
- 输入输出相同会在转换前被阻止。
- 预检错误显示在表单顶部和字段级位置。
- 转换运行时有表单内进度、日志和取消按钮。
- 转换结果不自动登记到项目数据集库。
- 代码仍通过 harness gate。
