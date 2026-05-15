# AITrain Studio 触碰即整理重构策略设计

## 背景

当前项目已经完成多轮保行为维护重构：

- `MainWindow` 已拆成多个 companion 文件，Qt Widgets shell 仍保持左侧导航、顶部状态栏、中央 `QStackedWidget` 架构。
- `ProductWorkflow.cpp` 已收敛为公共入口锚点，质量、评估、benchmark、交付、验收、pipeline 等逻辑已拆到 companion 文件。
- `DetectionTrainer.cpp`、`WorkerSession.cpp` 和核心测试已做过第一层热点拆分。
- 当前 harness 明确要求：不要默认继续 source-layout refactor；下一次重构必须由具体维护阻塞驱动。

因此，本策略不启动独立大重构，也不改变功能优先级。它只规定后续开发触碰热点文件时，如何顺手做小步、保行为整理。

## 目标

建立一条轻量维护规则：只在功能或修复实际触碰到相关区域时整理局部结构，让代码持续变薄、边界更清晰，同时避免为重构而重构。

成功标准：

- 不单独开启全项目结构重排。
- 不改变 Worker protocol、SQLite schema、插件接口、GUI 架构或报告 JSON 语义。
- 每次整理必须和当前任务直接相关，且能通过对应 harness gate 验证。
- 热点文件新增逻辑时优先放入已有 companion 或小 helper，而不是继续扩大单文件职责。

## 非目标

- 不做跨模块重命名或目录重排。
- 不拆公共 API，除非后续任务明确需要并另行设计。
- 不把训练、推理、导出、评估、benchmark、报告或 OCR 验收逻辑移动到 `MainWindow`。
- 不因为文件行数较大就单独创建重构任务。
- 不借重构扩大 Phase 40 分类、姿态、OBB、异常检测等后置功能范围。

## 触发规则

只有满足以下任一条件时，才允许在当前任务内做局部整理：

1. 新功能需要在同一个热点文件中新增明显重复流程。
2. 修复需要同时理解多个职责块，现有边界已经妨碍定位问题。
3. 新增测试很难覆盖目标逻辑，因为逻辑被绑定在 GUI 或大型流程函数里。
4. 当前改动会让某个热点文件继续膨胀，并且可以用低风险 helper 或 companion 消除膨胀。

不满足这些条件时，只做业务改动，不做整理。

## 候选热点与整理方向

### `src/app/src/MainWindowActions.cpp`

现状：该文件仍集中承载大量 Worker 启动、表单校验、任务登记、启动失败回写和 UI 状态更新流程。

触发条件：新增或修改 Worker 型 GUI 操作时。

整理方向：

- 抽取轻量任务启动 helper，统一检查 `worker_.isRunning()`、创建 repository task、启动失败回写、`workerPill_` 状态设置和 status bar 消息。
- 保留 `MainWindow` 作为编排层，不把核心训练、评估、导出或数据处理逻辑放回 GUI。
- 优先整理当前触碰的任务族，例如 dataset、model、delivery，不一次性改完所有 action。

### `src/core/src/DatasetConversion.cpp`

现状：COCO、VOC、YOLO 解析、写出、校验和报告生成仍在一个实现文件中，但已有 focused tests 覆盖转换矩阵。

触发条件：扩展 LabelMe、COCO RLE、更多转换矩阵，或修改现有转换语义时。

整理方向：

- 按格式拆 companion：`DatasetConversionCoco.cpp`、`DatasetConversionVoc.cpp`、`DatasetConversionYolo.cpp`、`DatasetConversionSupport.cpp`。
- 保持 public API `convertDataset(...)` 不变。
- 每次只拆当前触碰的格式，避免一次性机械搬运所有代码。

### `src/core/src/ProductWorkflowEvaluation.cpp`

现状：检测、分割、OCR Rec 评估共用一个入口，评估报告结构和错误样本逻辑已经较重。

触发条件：新增评估类型，或修改检测/分割/OCR 其中一类评估报告语义时。

整理方向：

- 按 evaluator companion 拆分：检测、分割、OCR 各自独立实现。
- 保持 `evaluateModelReport(...)` 对外入口稳定。
- 优先抽公共报告写入、summary markdown、error taxonomy helper，减少三类评估重复 JSON 组装。

### `src/core/src/ProjectRepository.cpp`

现状：多表初始化、insert/update 和 query mapper 集中在一个文件，但 public API 较小，当前风险可控。

触发条件：新增 SQLite 表、迁移字段，或新增多组 repository 查询时。

整理方向：

- 优先拆初始化/迁移 helper，再按 query mapper 分组。
- 不改变 `ProjectRepository.h` 的现有调用方式，除非 schema 任务另行设计。
- schema 变更必须同步测试和 current-status 边界说明。

## 验收方式

文档或策略变更：

```powershell
git diff --check
.\tools\harness-context.ps1
```

代码变更：

```powershell
.\tools\harness-check.ps1
```

UI 布局变更还需要至少一次 1280x820 walkthrough，确认关键操作可见、长路径和长文本不造成水平溢出。

## 决策

采用“触碰即整理”策略。当前不创建 implementation plan，不进入代码实现。后续只有当具体功能或缺陷修复触碰上述热点区域时，才把局部整理纳入对应任务的设计和实施计划。
