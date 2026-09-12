# AITrain Studio 当前状态

## 工作台界面重设计（2026-09-12）

已完成批准方案及用户要求的收尾：三个日常入口、单层工作台视图、全项目目录搜索、浅色/深色主题、按项目保存并跨重启恢复训练草稿，以及完整英文工作台静态文案。项目激活和延迟建页不覆盖恢复草稿；失效数据、样本或后端需重新选择。

Schema 13、Worker Protocol V2、Artifact Journal V2、十四种 Worker 命令及八个官方训练后端保持不变。新增查询为只读投影，预览完整性校验继续在后台执行。

最终 `tools/harness-check.ps1` 通过：35/35 组 CTest、0 失败，耗时 1001.05 秒。原生 100% 界面回归 35 项通过；100%/125% 覆盖三个逻辑尺寸，150% 原生覆盖 1024×700，三个较大逻辑尺寸通过离屏验证。最后修正的搜索与英文分类布局已补验。命令、日志、关键文件及实际验收边界见 `docs/design/ui-workbench-closeout.md`，首轮记录保留在 `ui-workbench-implementation.md`。这些结果不代表新增客户域精度、干净机器或硬件验收。

历史基线整理日期：2026-07-27

本文只记录当前代码与产品边界。历史路线、旧阶段结论和已删除实现不属于当前实施依据；宽路线请查看 `docs/product-roadmap-local-training-platform.md`。

## 当前基线

- SQLite 项目格式为破坏性的 Schema 13。
- `project_meta` 是项目版本、ProjectId、显示名和 `open_generation` 的唯一来源；旧 `schema_info` 不再使用。
- Schema 12 及更早数据库只返回 `SchemaRebuildRequired`，打开过程不迁移、不修改旧库。
- `openProject`、`createProject`、`rebuildProject` 已分离：
  - 打开缺失目录或缺失 `.aitrain/project.sqlite` 时不创建任何内容；
  - 创建入口不覆盖已有项目；
  - 重建只删除已校验项目根目录中的 `.aitrain`，并生成新的 ProjectId。
- Studio 使用 `.aitrain.owner.lock` 持有项目所有者 Lease；Worker 子进程使用 `.aitrain.worker.lock`。两者均采用 `QLockFile` 和零 stale 超时，不强抢活动锁。
- 后台 prepare 完成恢复并递增 generation 后返回一次性 `PreparedProjectSession`；GUI 激活时复核 ProjectId 与 generation，候选失败不替换现有 Session。
- Worker 只通过 `openForWorkerChild()` 建立独立 SQLite 连接，不取得 Owner Lease、不执行全项目恢复、不递增 generation。

## 模块与构建边界

- `aitrain_foundation` 已删除。
- 产品事实由 Qt Core-only 的 `aitrain_product_contract` 统一提供。
- 视觉数据、标注集成、数据集转换、模型导出和视觉运行时分别由以下窄目标承载：
  - `aitrain_vision_data`
  - `aitrain_annotation_integration`
  - `aitrain_dataset_conversion`
  - `aitrain_model_export`
  - `aitrain_vision_runtime`
- 历史伪训练入口 `DetectionTrainer` 及其伞形头已删除；生产训练只走官方 Python 后端。
- Protocol V2、控制 Envelope、TaskCommand JSON、既有 backend/capability ID 保持不变。
- 不存在动态插件产品入口，不得重新引入运行期插件接口。

## 产品能力合同

- 八个生产训练 backend ID 均由 Product Contract 声明，并与一个 Workflow Profile、一个 Python Profile 一一对应。
- `officialArtifactFormat` 表示正式训练交付格式。
- `exportFormats` 只表示可继续导出的目标，不代表可执行 Runtime。
- `runtimeRoutes` 单独来自 Runtime Route Contract。
- Worker `--builtin-capabilities` 保留原字段，并输出 `trainingWorkflowContracts` 与 `contractsValid`。

当前 Runtime 产品边界：

- YOLO Detection / Segmentation：ONNX Runtime；只有已验证 NCNN 包可使用 NCNN。
- OBB v1：仅 ONNX Runtime。
- SMP：仅 ONNX Runtime。
- TensorRT engine 可以导出，但 AITrain 当前没有 TensorRT 推理解码器，推理路线为 `NotImplemented`。
- Anomalib：Worker-managed Python，不声明 AITrain C++ ONNX/TensorRT/NCNN Runtime。
- PaddleOCR：只通过官方 Det/Rec/System 报告和验收流程形成 `OfficialEvidence`。
- `ExternalEvidenceRequired` 不等于本机 Runtime 可执行。
- Runtime 探测失败不会自动切换到其他路线。

## Storage、Artifact 与 Evidence

- `ProjectStore` 是跨聚合事务 façade；GUI 不直接访问 Store。
- `ProjectDatabase` 是当前 Workspace/线程内唯一的 `QSqlDatabase` owner；Store 复用同一连接，不为查询重复创建连接。
- 任务、任务 Artifact、指标、Workflow Run、数据集、模型包和交付证据目录统一使用版本化 base64url keyset cursor；页大小只接受 1–200，游标查询类型不匹配会返回 `InvalidPageCursor`。
- Query Service 与各目录 Presenter 只接受 `PageRequest` 并返回/消费 `Page<T>`；Presenter 的“加载更多”沿 opaque cursor 追加，刷新第一页会重置旧游标。Evidence 构建会在只读查询序列中穷举所有 Artifact 与指标页，不以固定条数截断交付事实。
- Schema 13 已建立任务、数据集、模型包、Artifact 文件、指标、Workflow Run、terminalization 和 outbox 的实际查询索引。
- Storage 八个要求的边界类均已落地并承接真实 SQL：`ProjectMetaRepository`、`TaskEventRepository`、`WorkflowRepository`、`WorkflowTerminalizationStore`、`ArtifactCatalogRepository`、`DatasetCatalogRepository`、`ModelCatalogRepository`、`ProjectReadRepository`。Repository 共用 `ProjectDatabase` 且不自行开启事务，跨表写事务仍由 `ProjectStore` Unit of Work 控制。
- Evidence 的 `projectIdentity` 使用持久化 ProjectId，项目移动后身份不变。
- 训练模型使用 `project_snapshot` 来源绑定；显式外部导入使用 `external_declared`，不伪造项目 Snapshot 血缘。
- Artifact journal 为 v2，并在 rename 前冻结 inventory、SHA-256 和 completion action。
- staging 到 committed 的同卷原子 rename 是提交线性化点。
- rename 前失败可安全中止；rename 后只能完成原 completion 或进入 `PendingRecovery`，不能反转成业务失败。
- 数据库已提交但 journal 清理失败仍是业务成功，只报告 `cleanupPending`。
- committed 文件读取统一通过 `ArtifactStore::openVerified()` 与 `VerifiedArtifactReader`，集中核对相对路径、完整 inventory、大小、SHA-256 与符号链接；`artifactPath()` 已收为 Artifact 模块私有实现，上层不再构造或直接取得 committed 根目录。证据超限明确返回 `TooLarge`。
- discard 先把 committed 原子移动到 `.trash/<ArtifactId>`，再在事务中复核引用并删除目录记录；恢复会按数据库事实恢复或清理 trash。
- terminalization 保持 `sealed → evidence_attached → closed`：
  - Evidence Artifact catalog 与 attach 同事务；
  - 根任务状态、状态事件与 close 同事务；
  - EvidenceRequired 根任务在 attach 前进入终态会被数据库 trigger 拒绝；
  - outbox 只有在 handler 后置条件成立后才标记 applied。

## 数据集、模型和许可

正式数据集转换路线只有三条：

1. COCO bbox → YOLO Detection
2. COCO polygon → YOLO Segmentation
3. Pascal VOC bbox → YOLO Detection

COCO RLE 和其他组合明确返回不支持。转换输出只进入 Artifact staging，成功后形成 Dataset、Version 和 Snapshot 身份，不接受任意 committed 输出目录。

Split 先生成只包含源相对路径、目标相对路径、partition、大小与 SHA-256 的不可变 `DatasetSplitPlan`；规划阶段不再创建临时输出目录。Materialize 前复核源根摘要，并按计划只复制一次。

Resume/Checkpoint 裸路径入口已删除。未来若恢复，只能重新设计为基于 ArtifactId 或 ModelPackageId 的能力。

机器码无法取得稳定种子时返回 typed unavailable；注册码界面阻止继续并显示原因，不使用 `unknown-machine` 回退。

## Worker 与 Python

- start/cancel 控制帧先完成 token、sequence、kind 和批次校验，再产生副作用。
- Worker 使用独立的 `aitrain_worker_runtime` 与单个 `ActiveWorkflowContext` 管理任务种类、阶段、Workspace、根 TaskId、Worker Lease 和幂等取消；各 Workflow 不再保存重复的 running/workspace/taskId 状态。
- 取消是幂等请求；Worker 不在 Core 持久化终态前发送业务终态。
- Worker 业务终态统一从已持久化 `TaskSnapshot` 映射；终态写入或回读失败时退出并交给 Recovery，不发送猜测性 completed/failed/canceled。
- Socket 断开只触发取消和进程退出，不伪造终态；项目恢复负责收口。
- Environment Check 与 Training 共用 Python Profile resolver。
- 解释器候选顺序固定为：
  1. Profile 专用环境变量；
  2. `AITRAIN_PYTHON_EXECUTABLE`；
  3. 应用旁 `python_env`；
  4. CMake 明确配置的开发候选；
  5. PATH 的 `python` / `python3`。
- 显式环境变量无效时立即失败，不静默回退。
- Profile 与必查模块：
  - `yolo`：ultralytics、torch、onnx、onnxruntime；
  - `smp_semantic_segmentation`：segmentation_models_pytorch、torch、torchvision、timm、onnx、onnxruntime；
  - `anomaly_detection`：anomalib、torch、torchvision、lightning、timm、PIL、numpy、cv2；
  - `ocr`：paddle、paddleocr，并检查 PaddleOCR 源码目录。
- Task 参数不再接受 `pythonExecutable`。
- 每个 Profile 同时使用 `requirements-*.txt` 与 `locks/<profile>-windows.txt`。
- `adapter_runtime.py` 统一请求读取、事件通道、AdapterSdk、异常/取消映射和单终态生命周期，不抽象算法实现。
- AdapterSdk 终态状态为 `OPEN → TERMINAL_SENDING → TERMINAL_ATTEMPTED`；sink 抛错后也不补发第二终态。
- Anomalib benchmark 使用无事件副作用的单次推理，默认 warmup=3、iterations=20，只产生一个报告 Artifact 和一个终态。

## GUI

- 主导航固定为九页：总览、项目、数据集、训练实验、任务与产物、模型库、部署验证、环境、系统设置。
- 当前 Qt Widgets 工作台继续保持左侧栏、顶部状态栏和中央 `QStackedWidget`。
- `MainWindow` 仅保留 App Shell、路由、会话/任务协调器、刷新协调器和九个页面指针；项目 Workspace/Query Service 由 `ProjectSessionController` 独占，活动 TaskId/Workflow kind 由 `TaskRuntimeController` 独占。
- 长任务只通过 Worker 执行，训练逻辑不进入 MainWindow。
- GUI 只传 DatasetId、SnapshotId、ArtifactId、ModelPackageId 和 TaskId，不重新暴露 committed 物理路径。
- Resume 控件、字段、翻译和文档入口已删除。
- Runtime Delivery 只允许 Product Contract 判定为 `AitrainCpp + Supported` 且本机 `Available` 的路线；不会静默 fallback。
- `TaskRuntimeController` 是唯一 `WorkerClient` owner，统一管理 `Idle → Starting → Running → CancelRequested → Finalizing/Recovering → Idle`；取消只使用一组 30 秒协作窗口和 2 秒强制退出窗口。
- `WorkspaceReadModelCoordinator` 在单个 event-loop 内合并刷新域，并按任务、数据集/模型、选中任务、项目摘要、环境、交付证据的固定顺序执行。
- 任务、Artifact、Artifact 文件和指标分别由 `QAbstractTableModel` 提供；任务、Artifact、文件和指标使用独立 opaque cursor 与“加载更多”，Artifact 文件只在选择持久化 ArtifactId 后查询，预览继续异步校验。
- “任务与产物”已拆为 `TaskArtifactPage + TaskArtifactPageController`：页面拥有视觉控件，Controller 拥有筛选、分页、选择与 Presenter 生命周期；`MainWindow` 不再持有该页的表格、筛选器、按钮或 Presenter。
- Dashboard 已拆为 `DashboardWorkspacePage + DashboardPageController`；项目汇总和最近任务使用独立查询，其中最近任务固定读取 recent 10，不再复用任务中心的 100 条分页缓存。
- `ProjectSessionController` 是 GUI 唯一的项目 prepare/activate 协调器；创建、打开、重建分别传入显式操作，打开缺失目录不会被推断成创建。再次打开当前 canonical root 只刷新 GUI generation，不重复取得 Owner Lease。
- 项目页分别提供“创建项目”“打开项目”“重建项目”三个入口；重建必须经过明确确认，并提示会清除 `.aitrain` 中的元数据和已提交 Artifact、不会自动备份。
- 项目页已拆为 `ProjectWorkspacePage + ProjectPageController`：页面拥有表单、摘要控件和破坏性操作确认，Controller 构造项目会话命令并拥有项目摘要 Presenter；`MainWindow` 不再保存项目表单或摘要控件。
- 系统设置页已拆为 `SettingsWorkspacePage + SettingsPageController`：页面只拥有能力摘要、语言、默认目录和授权展示控件，Controller 负责产品能力合同投影与偏好持久化；顶部语言切换仅订阅同一 Controller，`MainWindow` 不再保存设置页业务控件。
- 模型库已拆为 `ModelRegistryWorkspacePage + ModelRegistryPageController`：页面只保存导入表单和模型目录视觉状态，Controller 拥有 Presenter、Manifest 草稿校验、Worker 命令构造和持久化 ID 选择；部署页只接收 `ModelPackageId`，`MainWindow` 不再保存模型库表单、表格或导入状态。
- 环境页的运行环境区域已拆为 `EnvironmentWorkspacePage + EnvironmentPageController`：页面渲染检查项与汇总，Controller 独占 Environment Presenter 和 Worker 命令；交付证据作为独立子页面嵌入，`MainWindow` 不再保存环境检查表格或状态控件。
- Runtime Delivery 已拆为 `RuntimeDeliveryWorkspacePage + RuntimeDeliveryPageController`：推理和部署验证共享 `RuntimeDeliveryFormData` 与命令构造，路线严格按 Manifest 顺序与 Product Contract、单次环境快照求交；仅列出 `AitrainCpp + Supported + Available`，多路线不默认、失效不替换，启动前重新校验，Benchmark 默认 warmup=3、iterations=20。
- 训练实验已拆为 `TrainingWorkspacePage + TrainingPageController`：页面拥有训练表单、运行监控、指标、日志和 Artifact 投影；Controller 拥有能力/任务/后端选择、Snapshot 四重身份绑定、训练命令构造及实时事件投影，`MainWindow` 不再保存训练表单或运行控件。
- 数据集页已拆为 `DatasetWorkspacePage + DatasetPageController`：页面拥有导入、转换、质量、划分、快照、标注修复与样本复核视觉状态；Controller 独占目录 Presenter、持久化身份选择、后台格式探测、五类 Worker 命令和复核 Artifact 异步读取代际，`MainWindow` 不再保存数据集表单、表格或预览状态。
- Worker 发布训练/部署 Artifact 事件时只消费 Core 返回的 `ArtifactId + relativePath`；活动源码不再拼接 `artifacts/committed/<id>`，运行产物 bundle 的成员路径也已改为明确的相对路径。
- Runtime benchmark 统一由 `RuntimeBenchmarkRunner` 执行，默认 warmup=3、iterations=20，并报告 setup、min、mean、p50、p95、p99、max、throughput 与 timingDefinition。
- 英文 TS 不允许 `unfinished`；语言切换继续采用重启生效。

## 构建与发布

- Python trainer 由 `SyncPythonTrainers.cmake` 同步到固定构建目录；同步前只清理该目录，并排除 Python cache。
- CTest 会写入 stale sentinel，验证同步可删除陈旧文件。
- `AITRAIN_REQUIRE_PYTHON_TESTS=ON` 时，缺少 Python、pytest 或测试注册会在 configure 阶段失败。
- Python CTest 运行完整 `pytest tests -q` 与独立 compileall。
- 安装组件：
  - `Runtime`：GUI、Worker、运行库、Python trainers、requirements/constraints、正式文档和最小示例；
  - `AcceptanceTools`：Harness、验收模板、正式 smoke、报告采集与交接材料。
- Runtime 不包含 Harness、roadmap、release freeze 历史材料、安装器源码或历史 phase 脚本。
- 主安装器只面向 Runtime；AcceptanceTools 独立交付；Python AI Environment 仍是独立包。

## 本地验证结果

2026-07-27 当前工作树已通过：

```powershell
.\tools\encoding-check.ps1
.\tools\architecture-check.ps1
.\tools\harness-check.ps1
.\tools\package-smoke.ps1
```

`harness-check.ps1`：35/35 CTest 通过，包含 Worker 活动上下文、完整 Python pytest、compileall 和 trainer stale 同步测试。

`package-smoke.ps1`：Runtime 与 AcceptanceTools 使用全新 install prefix；包根 GUI 启动、Worker self-check、内置 Product Contract、Schema 13 首次创建和二次打开均通过。

## 仍需外部证据的边界

以下内容没有因本次稳定化而被声明完成：

- TensorRT 真正推理解码器；
- Schema 12 数据迁移工具；
- 基于 ArtifactId 的正式 Resume；
- 客户域 OCR、OBB、异常检测和 SMP 精度证据；
- Clean Windows 外部验收；
- package-root TensorRT 外部复验；
- 新算法、云调度、多用户和远程协作。

任何 smoke、scaffold、公共数据结果或历史报告都不能替代上述外部证据。
