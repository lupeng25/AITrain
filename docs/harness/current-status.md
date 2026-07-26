# AITrain Studio 当前状态

更新时间：2026-07-26

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
- Evidence 的 `projectIdentity` 使用持久化 ProjectId，项目移动后身份不变。
- 训练模型使用 `project_snapshot` 来源绑定；显式外部导入使用 `external_declared`，不伪造项目 Snapshot 血缘。
- Artifact journal 为 v2，并在 rename 前冻结 inventory、SHA-256 和 completion action。
- staging 到 committed 的同卷原子 rename 是提交线性化点。
- rename 前失败可安全中止；rename 后只能完成原 completion 或进入 `PendingRecovery`，不能反转成业务失败。
- 数据库已提交但 journal 清理失败仍是业务成功，只报告 `cleanupPending`。
- committed 文件读取统一通过 `VerifiedArtifactReader`，集中核对相对路径、inventory、大小、SHA-256 与符号链接；证据超限明确返回 `TooLarge`。
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
- 长任务只通过 Worker 执行，训练逻辑不进入 MainWindow。
- GUI 只传 DatasetId、SnapshotId、ArtifactId、ModelPackageId 和 TaskId，不重新暴露 committed 物理路径。
- Resume 控件、字段、翻译和文档入口已删除。
- Runtime Delivery 只允许 Product Contract 判定为 `AitrainCpp + Supported` 且本机 `Available` 的路线；不会静默 fallback。
- `TaskRuntimeController` 是唯一 `WorkerClient` owner，统一管理 `Idle → Starting → Running → CancelRequested → Finalizing/Recovering → Idle`；取消只使用一组 30 秒协作窗口和 2 秒强制退出窗口。
- `WorkspaceReadModelCoordinator` 在单个 event-loop 内合并刷新域，并按任务、数据集/模型、选中任务、项目摘要、环境、交付证据的固定顺序执行。
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

2026-07-26 当前工作树已通过：

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
