# AITrain Studio 开发架构说明

最后更新：2026-07-17

本文面向后续维护和扩展开发，说明当前架构边界、主要模块、扩展入口和验证要求。所有阶段事实以 `docs/harness/current-status.md` 为准。

## 总体边界

AITrain Studio 当前是 Windows + NVIDIA GPU 本地视觉训练工作台，核心原则是：

- GUI 只做编排、状态展示和用户输入，不执行长任务。
- 长任务进入 `aitrain_worker`。
- 训练优先通过 Worker 启动独立 Python 子进程。
- 元数据进入 `ProjectStore` / SQLite；GUI 读取经 Query Service/Presenter，业务写入经 `ProjectWorkspace`/Worker 收口。
- 模型、数据集、训练、验证、导出、推理扩展走内置能力注册表。
- scaffold、smoke、diagnostic 能力必须明确标注，不能写成真实生产训练能力。

## 运行时结构

```text
AITrainStudio.exe
  -> Qt Widgets workbench
  -> WorkerClient
  -> ProjectStore / Query Service / Presenter
  -> BuiltinCapabilityRegistry / capability matrix UI
  -> aitrain_worker.exe
       -> core workflow functions
       -> Python trainer subprocesses
       -> ONNX Runtime / TensorRT / report writers
       -> filesystem artifacts
```

控制面使用统一 JSONL 协议 over `QLocalSocket`；Python Adapter 使用认证 loopback 事件通道，stdout/stderr 只保留为原始诊断日志。任何协议字段变更都需要同步 Worker、GUI、Adapter 和测试。

## 源码地图

| 路径 | 职责 |
|---|---|
| `src/core` | 统一 Protocol、能力注册表、数据集校验/转换、训练/评估/交付 Workflow、`ProjectStore`/Artifact Store、ONNX/TensorRT 支持。 |
| `src/app` | Qt Widgets GUI、页面、Presenter、ApplicationEventRouter、Worker Client、预览、翻译和能力矩阵。 |
| `src/worker` | 独立任务进程、WorkerSession、数据集/训练/模型/交付命令入口。 |
| `src/core/CapabilityRegistry.cpp` | 内置 YOLO、semantic segmentation、anomaly detection、PaddleOCR、dataset interop 能力注册表。 |
| `src/license_generator` | 内部离线注册码生成器。 |
| `tests` | QtTest 和核心行为覆盖。 |
| `tools` | harness、acceptance、packaging、handoff、OCR validation 脚本。 |
| `docs/harness` | 当前状态、项目上下文、质量门禁和工作流约束。 |

## GUI 架构

必须保留当前 Qt Widgets workbench 架构：

- 左侧 `Sidebar`
- 顶部状态栏
- 中央 `QStackedWidget`
- `AppStyle`
- `InfoPanel`
- `StatusPill`

新增页面或重构页面时：

- 不把卡片嵌套在卡片里。
- 不把长任务放进 GUI 线程。
- 不把模型训练、导出、推理、验收逻辑塞进 `MainWindow`。
- GUI 文案优先使用 `QStringLiteral`，保持 UTF-8 编译。
- 中文源文案通过 Qt translation / `LanguageSupport` 派生英文界面。
- 宽度、长路径、错误文本和中英切换都要考虑无横向溢出。

## Worker 任务模式

新增 Worker 任务建议按以下顺序：

1. 在 core 中定义可复用的数据结构和纯逻辑。
2. 在 WorkerSession 中添加命令入口。
3. 通过 Worker message 返回进度、日志、结果、失败或取消状态。
4. 在 GUI 中只添加表单、按钮、状态、预览和任务记录。
5. 需要持久化时通过 `ProjectWorkspace`/Worker 调用 `ProjectStore` 写入项目数据库。
6. 添加 focused test，再按风险运行 harness。

任务必须支持：

- 明确的请求 JSON。
- 可诊断的错误码和错误消息。
- ArtifactId、包内相对成员和清单哈希记录；不把物理 Artifact Store 路径交给 GUI 或控制面。
- 取消或失败后的状态恢复。
- 不阻塞 GUI。

## 数据与产物边界

- 数据集索引、任务、指标、Artifact、模型包和 Workflow/Evidence 关系进入 SQLite。
- 大文件、报告、图片、ONNX、engine、ZIP 等写入 Artifact Store；SQLite 只保存 ArtifactId、相对成员、字节数和 SHA-256，不保存供 GUI 使用的物理路径。
- `.deps`、build 输出、模型权重、数据集和生成二进制不提交源码控制。
- 数据集转换、快照导入和划分均由 EvidenceRequired Workflow 原子登记目标 Snapshot；显式外部导入边界之外不接受裸路径结果。
- 交付报告和诊断包是证据产物，不应修改用户全局 Python、CUDA、驱动或数据。

## 能力注册边界

模型、数据集、训练、验证、导出和推理能力统一由 `BuiltinCapabilityRegistry` 描述。新增能力必须同时更新注册表、Worker 兼容性校验、GUI 选择器、环境自检和验收测试，不得在页面中散落后端字符串。

## 当前训练边界

以下历史实现已从产品训练路径中物理删除，不能作为 Worker 后端或验收 passed 依据重新暴露：

- `tiny_linear_detector`
- shipped `python_mock`
- tiny detection checkpoint/export
- segmentation baseline/scaffold training
- OCR baseline/scaffold training
- small PaddleOCR Rec CTC trainer

Generated smoke 或 public dataset smoke 只能证明官方链路接线和 artifact 生成，不能写成客户域生产精度结论。

当前生产训练入口是 Worker-managed 官方/上游适配器：Ultralytics YOLO detection/segmentation/OBB、SMP semantic segmentation、Anomalib PatchCore/EfficientAD，以及 PaddleOCR Det/Rec 官方适配器。YOLO/OBB、SMP、Anomalib 和 OCR 的运行时边界不同，必须按 `docs/harness/current-status.md` 和 `docs/training-backends.md` 的能力说明记录限制。

官方 Ultralytics、SMP、Anomalib、PaddleOCR、PaddlePaddle 等后端也需要独立依赖和许可证审查。Public smoke 只证明接线和 artifact 生成，不证明客户域精度。

## 文档与状态

关键文档：

- `HARNESS.md`
- `docs/harness/current-status.md`
- `docs/harness/project-context.md`
- `docs/harness/implementation-checklist.md`
- `docs/harness/quality-gates.md`
- `docs/harness/ui-guidelines.md`
- `docs/user-guide.md`
- `docs/dataset-conversion.md`
- `docs/delivery-evidence-index.md`
- `docs/operations-runbook.md`

计划文档必须用中文输出。修改阶段状态、交付证据或验收口径时，优先更新 `docs/harness/current-status.md`，再更新 README、用户手册和索引文档。

## 验证

文档或上下文变更至少运行：

```powershell
git diff --check
.\tools\harness-context.ps1
```

代码变更运行：

```powershell
.\tools\harness-check.ps1
```

UI 变更还需要按 `docs/harness/ui-guidelines.md` 做 walkthrough 或截图核验。训练、导出、推理、验收相关变更需要按 `docs/harness/quality-gates.md` 选择对应 smoke 或 acceptance 命令。
