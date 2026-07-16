# AITrain Studio V2 破坏性重构实施计划

日期：2026-07-15  
状态：执行中  
适用范围：AITrain Studio GUI、Core、Worker、Python Adapter、SQLite、数据集、模型运行时、工作流、授权与测试体系

## 1. 文档定位

本文把 V2 破坏性重构方案拆解为可执行任务，是后续架构重构的实施主计划。

文档关系：

- 当前产品能力和阶段事实以 `docs/harness/current-status.md` 为准。
- 产品方向和算法范围以 `docs/product-roadmap-local-training-platform.md` 为准。
- 本文只负责 V2 内部架构重构、任务顺序、删除范围和验收门禁。
- 实施过程中如果本文与当前产品后端边界冲突，以 harness 中的产品边界为准，并同步修订本文。

本项目尚未上线，本轮重构明确采用以下策略：

- 不兼容旧 SQLite 数据库。
- 不兼容 Worker Protocol V1。
- 不保留旧内部 C++ API。
- 不保留旧任务状态和旧报告字段的兼容读取。
- 本地开发项目、测试数据库和历史临时产物可以删除重建。
- 保留 Qt Widgets 工作台形态和官方/上游算法后端边界。
- 本轮不同时升级 Qt；Qt 工具链升级在 V2 稳定后单独立项。

## 2. 重构目标

V2 必须建立四条端到端强约束：

1. 每个任务、请求、事件和产物都能追溯到唯一 `taskId` 和 `requestId`。
2. 所有可使用输出都是不可变的、经过校验的原子提交产物。
3. 所有长任务和后代进程都由统一进程监管器托管和取消。
4. 所有模型进入推理、评估、导出和部署前都必须具有明确的 Model Manifest。

目标模块关系：

```text
Qt View
  -> Application Service / Presenter
    -> Domain Contract
      -> Storage / Worker Client / Artifact Store

Worker Host
  -> Job Handler
    -> Dataset Driver / Runtime Adapter / Workflow Step
      -> Process Supervisor / Python Adapter / Artifact Store
```

## 3. 全局实施规则

### 3.1 分支和提交规则

- 使用独立重构分支，建议名称：`codex/v2-destructive-refactor`。
- 每个任务编号至少对应一个可独立审查的提交，机械删除可按阶段合并提交。
- 每个阶段结束必须形成阶段总结，记录已删除内容、门禁结果和剩余风险。
- 不允许长期维持 V1/V2 双写；短期并行代码只用于完成垂直切片。
- 不新增 V1 到 V2 的协议、数据库或报告兼容适配器。

### 3.2 通用完成标准

每个编码任务至少满足：

- 新行为有自动化测试。
- 所有新增文本为 UTF-8。
- 长任务支持取消、进度和明确失败信息。
- 失败和取消不产生可使用的半成品。
- GUI 线程不执行阻塞等待或长任务。
- 不把 scaffold、smoke 或 report-only 行为描述为生产能力。

### 3.3 通用验证命令

```powershell
.\tools\encoding-check.ps1
.\tools\harness-check.ps1
git diff --check
```

阶段门禁不得依赖“失败后自动重试通过”来判定成功。对并发、进程和 UI 生命周期相关阶段，完整门禁需要连续运行三次并全部通过。

## 4. 阶段总览

| 阶段 | 主题 | 主要结果 | 前置阶段 |
|---|---|---|---|
| 0 | 基线与门禁恢复 | 当前 UI 崩溃和 Worker 不稳定测试被消除，建立 V2 ADR | 无 |
| 1 | Domain、Protocol V2、SQLite V2 | 强类型身份、统一状态机、任务协调器、新数据库 | 0 |
| 2 | Artifact Store 与 Dataset V2 | 原子产物、完整快照、格式驱动、无脏输出 | 1 |
| 3 | 进程监管与 Python Adapter V2 | Job Object、独立事件通道、可靠取消 | 1、2 |
| 4 | Model Package 与 Runtime V2 | 强制 Manifest、运行时适配器、准确能力状态 | 1、2、3 |
| 5 | Workflow 与 Evidence V2 | 可组合步骤、统一证据模型、可信报告 | 2、3、4 |
| 6 | GUI Presenter 化 | MainWindow 壳化、页面解耦、异步任务中心 | 1、2、4、5 |
| 7 | 授权、安全、清理与发布验收 | 删除 V1、授权加固、文档与发布门禁闭环 | 0–6 |

### 4.1 执行跟踪

本表只记录已通过自动化验证的实施结果；未列出的任务仍按后续章节执行，不能据此视为阶段完成。

| 任务 | 状态 | 已落实内容 | 验证 |
|---|---|---|---|
| V2-001 | 已完成（首个修复切片） | 修复评估报告视图清空时由表格信号触发的重入页面更新，避免页面切换访问冲突。 | UI 定向测试、完整 UI 测试通过。 |
| V2-004 | 已完成（首个修复切片） | Harness 不再把失败后的自动重试当作通过；失败时保留独立测试日志后直接阻断。 | 完整 Harness 首次执行通过。 |
| V2-005 | 已完成 | 已建立 V2 不兼容、依赖方向、任务事件、产物、进程树、模型 Manifest 与失败分类 ADR。 | 文档审查。 |
| V2-100～102 | 已完成 | 新增 `aitrain_v2_domain`：强类型 UUID、统一状态机、稳定失败码和值对象。 | `aitrain_v2_domain_tests`。 |
| V2-103 | 已完成（协议库） | 新增 `aitrain_v2_protocol`：严格 Envelope、kind 白名单、消息限额、顺序/重复/跨任务检查。 | `aitrain_v2_protocol_tests`。 |
| V2-104～105 | 已完成（任务与数据快照存储） | 新增 `aitrain_v2_storage`：新 SQLite Schema、WAL、外键、busy timeout、CAS 状态迁移、事件/指标/产物持久化，以及数据集、版本、快照到 Artifact 的可查询关系；V1 和旧 V2 开发数据库均按破坏性重构策略拒绝打开。尚未覆盖所有业务表的 Repository 用例。 | `aitrain_v2_storage_tests`、`aitrain_v2_application_tests`。 |
| V2-106、V2-108 | 已完成（Fake Worker 垂直切片） | 新增 `aitrain_v2_application`：TaskCoordinator 通过协议校验持久化 Worker 事件，具备幂等取消请求与最终取消状态；Fake Worker 覆盖进度、指标、产物、成功和取消终态。 | `aitrain_v2_application_tests`。 |
| V2-107 | 已完成（规划内核） | 新增不可变 `ExecutionPlanV2` 与 Capability Planner：统一校验 capability、任务、数据集、训练/评估后端、导出格式和运行时路由，并以 SHA-256 摘要供 Worker 复核。尚未接入 GUI/真实 Worker 启动命令。 | `aitrain_v2_application_tests`。 |
| V2-200 | 已完成（首个切片） | 新增 `aitrain_v2_artifact`：staging、SHA-256 清单、manifest、同卷目录改名提交、`artifact_files` 原子登记与失败补偿删除。尚未接入全部工作流。 | `aitrain_v2_artifact_tests`。 |
| V2-201 | 已完成（staging 恢复） | `ArtifactStoreV2` 为 staging 写入独立元数据；启动恢复时保留活动任务 staging，清理不存在任务及失败/取消任务的 staging，并产生诊断文本；从不扫描或删除 committed 目录。 | `aitrain_v2_artifact_tests`。 |
| V2-205 | 已完成（自包含快照导入与受控登记已验证） | Snapshot V2 保留完整性标记、Driver/版本、类别定义、稳定根哈希、文件大小与 SHA-256。Storage schema 11 取消以 `rootPath` 判定 Dataset 身份：稳定 `DatasetId` 可登记多个 Version/Snapshot，每个 Snapshot 的可执行根独立指向其 committed `dataset_snapshot_v2` Artifact；训练 Workflow 通过持久化外部输入绑定跨任务消费该 Artifact。新增 `runDatasetSnapshotImportWorkflowV2`：显式外部 `sourcePath` 只存在于导入边界，依次执行 `PlanSnapshotImport → MaterializeAndRegisterSnapshot`，冻结清单后逐文件重验复制、再次运行 Driver、提交自包含快照、登记真实 ID，并以 EvidenceRequired 收口。旧 `createDatasetSnapshot` Socket/Client/Worker 路径事件和 GUI `ProjectRepository` 双写已删除；名称当前仍仅作审计。 | 2026-07-16 Storage/Application/Dataset/Worker/AITrainStudio/UI/Platform 目标构建成功；新 Dataset、同 Dataset 新 Version、源变化、取消、旧命令 Unsupported、无路径和自包含边界定向 CTest 6/6 通过（64.80 秒）。 |
| V2-202～203 | 已完成（七格式 Driver） | `DatasetDriverV2` 已为 YOLO Detection/Segmentation/OBB、Semantic Mask、Anomaly Folder、PaddleOCR Det/Rec 提供独立 Driver，并通过内置注册器拒绝格式重叠。布局可识别但数据有误时仍返回 Inspection，再由 validate 输出稳定问题 code。 | `aitrain_v2_dataset_tests`。 |
| V2-204 | 已完成（Split V2 产品接线已验证） | 七个 Driver 的 Plan 固化源/目标相对路径、split、sample key、SHA-256、字节数和源 rootHash；Semantic Mask 已移除绝对 `sourceImage/sourceMask`，统一使用相对路径、哈希和字节数。新增 `runDatasetSplitWorkflowV2` 三步 EvidenceRequired 工作流：`PlanSplit → MaterializeSplit → RegisterSnapshot` 只消费已登记 Dataset/Version/Snapshot/Artifact 四重身份，Plan Artifact 不保存绝对路径，Materialize 只写空 staging，Register 从 committed split inventory 重验并重打包纯目标树、再次运行 Driver 后提交独立自包含 Snapshot。请求和响应不含 source/output/report/artifact 裸路径；旧 `splitDataset` Socket/Client/Worker handler 与 GUI 输出目录、legacy Repository 双写已删除。 | 2026-07-16 修复 Qt 5.12 迭代 API 后，Core/Application/Worker/AITrainStudio/OCR Worker/UI/Platform 目标构建成功；身份错配/tamper/取消、自包含、Semantic 无绝对路径、旧命令 Unsupported 与 UI 边界定向 CTest 5/5 通过（68.03 秒）。 |
| V2-206 | 已完成（像素语义修复） | 新增 `SemanticMask` 原始字节读取：仅接受 Grayscale8/Indexed8，分别保留灰度值/palette index；验证、快照、质量检查、ONNX 推理/后处理与 SMP Python 适配器均已移除颜色亮度推导。 | `aitrain_dataset_tests` 的调色板索引回归、`aitrain_smp_semantic_python_tests`。 |
| V2-207 | 已完成（三条原子路径 + GUI↔Worker V2 接线已验证） | `DatasetConversionServiceV2` 支持 COCO bbox → YOLO Detection、COCO polygon → YOLO Segmentation、Pascal VOC bbox → YOLO Detection，并保持完整源冻结、确定输出 Plan、目标 Driver 校验和原子提交。新增 `runDatasetConversionWorkflowV2` 两步 EvidenceRequired 封装：`Convert` 只允许显式外部 source path 导入边界；`RegisterSnapshot` 从 committed conversion Artifact 的 Storage inventory 逐文件重验长度/哈希，只重打包纯目标数据树，排除 conversion plan/report，再次运行目标 Driver 后提交独立自包含 `dataset_snapshot_v2` 并登记真实 Dataset/Version/Snapshot。请求不接受 output/report/artifact path，响应只返回 Dataset/Version/Snapshot/Conversion Artifact/Snapshot Artifact/Evidence ID 与摘要。目标 DatasetId 可由 GUI 预分配；`targetDatasetName` 因 schema 无名称字段仅进入 Workflow 审计参数，摘要明确 `targetDatasetNamePersisted=false`，不伪造持久化名称。旧 `convertDataset` Socket command、Client、Worker handler 和路径结果事件已删除；独立 CLI 与无状态 Materialize 算法暂留。 | 2026-07-16 Core/Application/Worker/AITrainStudio/OCR Worker/UI/Platform 目标构建成功；Worker 成功、BackendUnsupported、取消、旧命令 Unsupported、无路径泄露及 UI payload 回归通过。修复后 Worker/Application/Conversion 定向 CTest 3/3 通过（44.80 秒）。 |
| V2-208 | 已完成（本机稳定可注入故障矩阵） | 已覆盖中文/空格路径、大小写无关目标冲突、未计划输出、plan 后源变化、复制中取消、192 个额外小文件、32 MiB 稀疏大文件的分块哈希，以及可移植相对路径边界。物化写入、报告写入和提交边界提供仅用于测试的稳定 I/O seam，分别模拟磁盘耗尽、目标锁定和提交目标占用；所有故障均验证 committed Artifact 为零。Worker kill 以 Failed 任务遗留 staging 模拟，恢复只清理该 staging，并验证活动任务 staging 与已提交 Artifact 保持不变。真实物理磁盘耗尽和跨机器文件锁语义仍属于外部平台验收，不以本机注入结果替代。 | `aitrain_v2_dataset_conversion_tests`，定向 CTest 1/1 通过（3.65 秒）。 |
| V2-300 | 进行中（进程树监管器与 GUI 控制面切片） | Windows `ProcessTreeSupervisor` 已提供 Job Object 监管底座；GUI `WorkerClient` ↔ `aitrain_worker` 本地 Socket 外层已迁移为 Protocol V2，启动参数预绑定 RequestId/TaskId，只接受 `command.start_task` / `command.cancel_task`，事件使用 `event.*`，并拒绝跨请求/任务、重复 MessageId、非递增 sequence、未知 kind、超限帧和终态后事件。Dataset/Annotation/Report 等业务命令仍暂存在 envelope payload，旧 Worker/Python 启动点的 Job Object 接入仍需逐项完成。 | `aitrain_v2_process_tests`、`aitrain_ocr_segmentation_worker_tests`、`aitrain_platform_tests`。 |
| V2-302 | 已完成（独立双端通道） | 新增 Python `adapter_event_channel_v2.py` 与 C++ `AdapterEventServerV2`：仅接受 literal loopback 地址，采用一次性 token 握手后发送带 request/task/sequence 的 Protocol V2 JSONL 事件，限制消息大小，并拒绝无效 token、非事件、跨任务、重复或乱序消息。V1 Worker 未接入；V2 Worker Host 启动编排和终态兜底仍待实现。 | `aitrain_adapter_sdk_python_tests` 假服务端测试、`aitrain_v2_process_tests` loopback 服务端测试。 |
| V2-301 | 已完成（V2 Adapter Host、任务协调与候选产物切片） | 新增 `PythonAdapterHostV2` 与 `TaskExecutionHostV2`：先启动认证事件服务，再注入一次性 endpoint、request/task 身份和取消文件环境变量；根进程启动后绑定 Windows Job Object，取消先写协作信号、宽限期后强制终止 Job。Host 通过 `TaskCoordinator` 持久化事件，并对“无终态退出”合成唯一 Failed/Canceled 终态。对于多步骤 Workflow，Host 还可托管既有 Running 任务：Adapter 终态仅审计入库、候选文件先原子提交为 Artifact，根任务只能由 Workflow Runner 收口，避免双重终态写入。目录型候选、V1 Worker 接入仍待实现。 | `aitrain_v2_process_tests`、`aitrain_v2_application_tests`。 |
| V2-307～308 | 已完成（进程树残留与日志资源门禁） | Windows 回归覆盖孙进程忽略协作信号后的取消强杀、Adapter 崩溃、Host/Worker 所有者销毁与 Job 关闭，逐项验证进程树归零；强制取消同时验证无 staging、无 Artifact 且 SQLite 只有一个终态。Adapter SDK 将实时结构化日志限制为 48 KiB UTF-8、输出队列限制为 256 行，完整官方输出流式写入文件并自动声明为 Artifact 候选；事件服务还限制未终止 JSONL 的 socket 缓冲。 | `aitrain_v2_process_tests`、`aitrain_v2_protocol_tests`、`aitrain_v2_application_tests`、`aitrain_adapter_sdk_python_tests`。 |
| V2-400 | 已完成（Manifest 契约与 Runtime 准入底座） | 新增严格 `ModelManifestV2`：要求模型包、来源 task/snapshot、SHA-256、产物内相对入口、输入输出张量、预后处理/decoder、类别、opset、导出版本、runtime route 和 verified 状态；缺失/无效 Manifest 为 Unclassified/Invalid，不能进入未声明 runtime。`RuntimeAdapterV2` 统一定义 probe/load/infer/benchmark/deploymentValidate，并在调用具体适配器前校验 route、目录边界、符号链接与 SHA-256。尚未接入 V1 runtime 或模型导入 UI。 | `aitrain_v2_model_manifest_tests`。 |
| V2-401 | 已完成（受控模型导入、目录读取、项目工作区与 GUI 首条入口） | 新增 `ModelImportServiceV2`：只接受用户明确确认的 Manifest 语义；建立导入任务，复制常规外部文件到 Artifact staging，计算 SHA-256，原子提交后登记 Model Package。Worker/GUI 分配的 UUID 直接作为 V2 `TaskId`，不再为同一动作创建双身份。复制、Artifact 哈希提交和登记前均检查取消；取消或失败会清理 staging/未引用已提交 Artifact，无法清理时明确终态为失败而非伪装成已取消。`ArtifactStoreV2` 同时清理无对应 staging 的元数据孤儿。`StorageV2::modelPackages()` 提供按创建时间倒序、带 limit 的已登记模型包目录；`ProjectWorkspaceV2` 管理项目内 `.aitrain-v2` 工作区，恢复 interrupted task/staging，并将导入、目录查询和 `ModelPackageId → RuntimeInvocationV2` 准备收敛到同一应用服务。模型库页已提供 ONNX 文件与 Manifest 草稿选择，并通过 Worker `importModelV2` 执行导入，GUI 不复制文件或计算哈希。入口路径越界、符号链接、无效 Manifest 或登记失败均拒绝；登记失败时删除未引用 Artifact。自动识别不参与模型类型决策；Manifest 草稿仍由用户/官方导出工具明确提供。 | `aitrain_v2_application_tests`、`aitrain_v2_artifact_tests`、`aitrain_v2_storage_tests`、`aitrain_v2_model_manifest_tests`、完整 Harness。 |
| V2-402 | 已完成（统一 Runtime Adapter 契约） | `RuntimeAdapterV2` 统一提供 `probe`、`validateModel`、`infer`、`benchmark`、`deploymentValidate`；所有实现先验证 Manifest、Artifact 目录边界、入口文件与哈希，再进入具体运行时。 | `aitrain_v2_runtime_adapter_tests`。 |
| V2-406 | 已完成（Runtime Capability Matrix） | 新增 `RuntimeCapabilityMatrixV2`，统一返回 Supported、UnsupportedByProduct、RuntimeNotImplemented、RequiresSdk、RequiresDependency、RequiresHardware、RequiresExternalEvidence，并输出对应 `runtimeStatus`；明确 OBB 不进入 NCNN、SMP 不进入 NCNN/TensorRT、异常检测与 OCR 需要各自官方/Worker 证据路径，且不再把软件、SDK、依赖、硬件和产品边界混为 `hardware-blocked`。Worker `--self-check` 与环境检查结构化报告已输出同一矩阵；GUI/交付报告消费尚待迁移。 | `aitrain_v2_model_manifest_tests`、`aitrain_v2_runtime_adapter_tests`、`aitrain_ocr_segmentation_worker_tests`。 |
| V2-403 | 已由 V2-503 收口 | 此阶段建立 ONNX Runtime Adapter、模型包解析、Manifest/哈希准入和预测/overlay 产物合同；原临时单步 Worker 入口已在 V2-503 中删除，产品只保留完整 Runtime Delivery 六步工作流。 | 历史底座由 `aitrain_v2_model_manifest_tests` 与 Runtime Delivery 回归继续覆盖。 |
| V2-404 | 已完成（NCNN Adapter 合同层） | 新增 NCNN V2 Adapter；仅接受显式 Detection/Segmentation decoder、静态 NCHW、唯一 blob、同名 `.bin` 及其 SHA-256，DFL 必须声明 strides/regMax。OBB、SMP、异常检测与 OCR 在 SDK 探测前即拒绝；SDK 可用时真实加载 param/bin 检出不兼容 layer。 | `aitrain_v2_runtime_adapter_tests`。 |
| V2-405 | 进行中（TensorRT 状态与合同层完成，真实 decoder 未实现） | 新增 TensorRT V2 Adapter 和 `tensorrt_engine` Manifest 格式，分别报告 SDK、依赖、GPU compute capability、engine build 与 runtime inference。官方 YOLO TensorRT decoder 当前明确返回 `RuntimeNotImplemented`，不得宣称真实推理或 GPU 验收成功。 | `aitrain_v2_runtime_adapter_tests`；真实 GPU 验收待外部证据。 |
| V2-407 | 进行中（Runtime Probe 统一状态已完成） | Matrix 已增加依赖缺失分类并输出统一 `runtimeStatus`，NCNN/TensorRT probe 已使用同一状态枚举；GUI 环境页和交付报告仍待全部改为直接消费 Probe Service。 | `aitrain_v2_runtime_adapter_tests`。 |
| V2-408 | 已完成（Runtime 合同测试） | 自动化覆盖 Manifest/格式不匹配、blob/tensor 合同、decoder 路由、NCNN `.bin` 哈希、禁用产品族，以及 TensorRT SDK/依赖/硬件/未实现 decoder 的精确分类。 | `aitrain_v2_runtime_adapter_tests`、`aitrain_v2_model_manifest_tests`。 |
| V2-303 | 已完成（SDK 底座） | 新增 `python_trainers/adapter_sdk.py`：统一结构化日志、进度、指标、产物候选、完成、失败、取消检查与直接子进程日志落盘/退出码记录；候选通过 `event.artifact_candidate` 与已提交的 `event.artifact` 明确区分，命令和环境内容不写入事件，避免泄露。进程树最终回收仍由未来 V2 Worker Host 的 Job Object 负责，现有官方 Adapter 尚待迁移。 | `aitrain_adapter_sdk_python_tests`、`aitrain_v2_protocol_tests`。 |
| V2-304 | 已完成（Ultralytics 训练、评估、导出 Adapter 迁移） | Detection 共享训练 Adapter 及 Segmentation/OBB 包装器，以及官方 `val()` evaluator、官方 exporter，均改由 SDK 发出日志、进度、指标、产物和终态；当 V2 Host 注入事件环境时使用受认证 loopback 通道，否则保留 JSONL stdout 作为 V1 过渡路径。V1 专用 `modelExport` 帧仅在 JSONL 路径保留；V2 以已提交导出 sidecar 为该元数据的事实来源。官方训练、评估、ONNX 导出的算法调用未替换。 | `aitrain_yolo_telemetry_python_tests`。 |
| V2-500 | 已完成（Workflow 持久化模型） | 将 V2 schema 升至 5（不兼容旧 V2 开发数据库）：新增有模板、根任务、顺序步骤、输入/输出 Artifact、后端、参数摘要、开始/结束时间、Failure 与重试次数的 Workflow Run/Step 持久化 API；同时落地数据集、版本、快照到 Artifact 的外键关系。步骤状态独立于任务状态，仅允许 `pending → running → succeeded/failed/canceled` 或显式 `skipped`；成功必须引用已提交 Artifact，失败后只能经显式 retry 回到 pending，拒绝不存在的 Artifact、跳序与并发覆盖。 | `aitrain_v2_storage_tests`、`aitrain_v2_application_tests`。 |
| V2-501 | 已完成（顺序 Workflow Runner） | 新增 `WorkflowRunnerV2`：按 ordinal 顺序执行 pending 步骤，自动把上一步已提交输出绑定为下游输入；已成功步骤可作为重开后的续跑检查点。执行器成功却未返回 Artifact 会被收口为失败；失败/取消时，当前步骤记录明确 Failure，后续 pending 步骤标记为 skipped，不尝试备用后端。同步 `run()` 供本地执行器使用，异步宿主则通过 `beginNextStep()` / `completeStep()` 只派发和收口已提交 Artifact；训练八步和 Runtime 六步均已使用该持久化模型。 | `aitrain_v2_application_tests`、`aitrain_v2_storage_tests`、`aitrain_v2_runtime_delivery_workflow_tests`。 |
| V2-502 | 进行中（全部生产训练后端垂直切片完成） | 训练 Workflow 已统一为由 `TrainingWorkflowProfileV2` 驱动的 `ValidateDataset → CreateSnapshot → Train → Evaluate → Export → DeploymentValidate → RegisterModel → RenderDeliveryReport` 八步；Worker、Workspace 与 GUI 不再各自维护后端列表，跨 Profile 混搭模板/评估/导出后端会在创建时拒绝。校验报告与快照均先提交为不可变 Artifact，再作为步骤输出和下游输入持久化。YOLO 三变体继续使用官方 Ultralytics 训练/评估/导出及 AITrain ONNX Runtime；SMP 仅声明 AITrain ONNX Runtime；Anomalib PatchCore/EfficientAD 使用 `anomalib_bundle` 和 Worker-managed Python；PaddleOCR Det/Rec 使用 `paddleocr_inference_bundle`、确定性 ZIP、完整 inventory/hash 和唯一 `paddleocr_official` runtime，不填充虚假的 ONNX tensor/opset。PaddleOCR 官方 train/eval/export/predict 只经 Adapter SDK 子进程托管，System adapter 作为独立 Det+Rec 组合 wiring（不是任一单组件训练步骤）也已迁移到认证事件通道并强制校验结果。八种 Profile 行均通过 Worker 八步 E2E，终态采用 SQLite 封存、Artifact commit journal 关联 Evidence、关闭根任务的持久化三阶段协议。当前剩余 GUI/Presenter 收口和 legacy 路径删除。 | `aitrain_v2_training_workflow_profile_tests`、`aitrain_v2_storage_tests`、`aitrain_v2_artifact_tests`、`aitrain_v2_application_tests`、`aitrain_v2_model_manifest_tests`、`aitrain_ocr_segmentation_worker_tests`、`aitrain_smp_semantic_python_tests`、`aitrain_anomalib_adapter_python_tests`、`test_paddleocr_v2.py`、`test_paddleocr_system_v2.py`。 |
| V2-503 | 已完成（统一六步 Runtime Delivery 产品入口） | `ProjectWorkspaceV2` 固定执行 `ImportOrResolveModel → ValidateManifest → RunInferenceSmoke → Benchmark → DeploymentValidate → RenderDeliveryReport`；GUI 两个页面统一派发 `runRuntimeDeliveryWorkflowV2`，Worker 独占根任务、六步执行和 Evidence 收口。控制结果只返回 ArtifactId 与结构化状态，不传模型或 Evidence 裸路径。旧单步命令和 Core/GUI 单步派发 API 已删除。Benchmark 是本机固定样本 smoke timing。ONNX 同步 infer 不可中途抢占，取消在调用返回后收口；NCNN 只支持产品矩阵允许且合同完整的 Detection/Segmentation，TensorRT decoder/真实 infer 仍未实现。 | `aitrain_v2_runtime_delivery_workflow_tests`、`aitrain_ocr_segmentation_worker_tests`、`aitrain_delivery_acceptance_ui_tests`、`aitrain_v2_model_manifest_tests`、`aitrain_platform_tests`；2026-07-16 定向 CTest 7/7 通过。 |
| V2-504 | 已完成（Evidence Bundle 事实模型） | 新增严格的 `EvidenceBundleV2`：包含项目身份、终态任务、Workflow Run、数据集 Snapshot、后端/环境、参数、指标、runtime/evaluation/benchmark 事实、已提交 Artifact 索引、限制和 Failure。拒绝非终态任务、成功任务携带 Failure、非成功缺少 Failure 及重复/无效 Artifact ID；运行时 Bundle 只从 V2 Storage 的已持久化记录读取，包含 Artifact 文件哈希/字节数和 Workflow 输入/输出引用。 | `aitrain_v2_evidence_tests`、`aitrain_v2_application_tests`。 |
| V2-505 | 已完成（统一 Renderer 与终态报告 Artifact） | `EvidenceRendererV2` 从同一个已校验 Bundle 生成 JSON、Markdown、HTML 与 Model Card；Model Card 直接嵌入 Evidence 而不复制或重新判断 task/runtime 状态。`ProjectWorkspaceV2` 仅在根任务终态后，使用 `ArtifactStoreV2` 将四份报告一次性提交为 `evidence_bundle_v2`，提交前校验任务身份、终态、Failure；成功、失败和取消 Workflow 均有回归覆盖。旧交付报告页面和完整多步骤 Delivery Workflow 尚未迁移。 | `aitrain_v2_evidence_tests`、`aitrain_v2_application_tests`。 |
| V2-506 | 已完成（七种 Driver Core + GUI↔Worker V2 接线已验证） | `ValidateSnapshot → AnalyzeQuality → ProduceRepairManifest → RenderQualityReport` 四步 Workflow 只接受 Storage 已登记 Snapshot 及其 committed Artifact，每个源文件在使用前重新匹配 manifest 的相对路径、字节数和 SHA-256；四步分别提交校验、问题样本、修复/X-AnyLabeling 复核清单及 JSON/Markdown/HTML 报告 Artifact，成功、失败和取消均通过 EvidenceRequired 三阶段协议形成唯一根任务终态与 Evidence。规则覆盖七种 Dataset Driver。GUI↔Worker 统一使用 `runDataQualityWorkflowV2`，请求同时携带同一条登记记录的 DatasetId、DatasetVersionId、SnapshotId、Snapshot ArtifactId 和结构化选项，禁止 dataset/report/output 裸路径；Core 在 `ValidateSnapshot` 内核对身份，响应只返回 Workflow/Artifact/Evidence ID 与结构化摘要。旧 `validateDataset` / `curateDataset` Socket command、Client 和 Worker handler 已物理删除；终态由 V2 Query/Presenter 刷新，不写 `ProjectRepository` 质量任务镜像。 | 2026-07-16 Core、Worker、AITrainStudio、OCR Worker、UI 与 Platform 目标构建成功；Worker 成功、身份错配、取消、旧命令 Unsupported、无路径泄露及 UI payload 定向 CTest 4/4 通过（79.11 秒）。 |
| V2-507 | 已完成（六格式 Core 与 GUI↔Worker 已验证） | 创建与同步使用两个 EvidenceRequired Workflow。创建只消费 committed repair/X-AnyLabeling Artifact，Session Artifact 固化 schema/kind、源 Dataset Version/Snapshot、问题 lineage、完整基线哈希、编辑白名单、输出 Artifact ID、时间及脱敏工具参数摘要，不保存外部工作目录；工作目录必须为空、独立且非符号链接，失败/取消仅清理本次复制内容。同步重新校验 Session、repair lineage、基线 Snapshot、目录身份、文件集合、相对路径、白名单、SHA-256 与目标 Driver；合法变更才把全部数据文件和 manifest 一起原子提交为自包含 `dataset_snapshot_v2` 并登记新 Dataset Version/Snapshot，原始数据不修改。Core 已按 Data Quality 实际 action 合同支持 YOLO Detection/Segmentation/OBB、Semantic Mask、PaddleOCR Det/Rec，逐格式限制可编辑文件和合法 action；Anomaly Folder 需要新增异常样本，超出文件集合不可变合同，因此精确返回 `BackendUnsupported`。GUI/Worker 统一使用 `createAnnotationSessionV2` / `syncAnnotationSessionV2`，payload 不传 repair/report/manifest 裸路径；Worker 创建 V2 根任务并调用 Core，结果只返回状态、Workflow/Artifact/Evidence ID，GUI 不写 `ProjectRepository`，终态后由 V2 Query/Presenter 刷新。旧 Socket 业务命令、裸路径 Worker 测试、`aitrain_worker` 独立 `--annotation-session-request` / `--annotation-sync-request` CLI、对应 WorkerRequests parser/type 及 V1 ProductWorkflow 会话实现均已物理删除。 | 2026-07-16 统一构建成功；包含六格式成功/冲突/越界/取消、Anomaly unsupported、OCR Worker/UI 的定向 CTest 5/5 通过（62.81 秒）。独立历史 CLI 删除切片待本轮统一构建。 |
| V2-508 | 已完成（受控导入与 ArtifactId-only 验收链路） | 固定 `ResolveEvidence → ValidateOfficialReports → EvaluateThresholds → RenderAcceptanceReport` 四步 Workflow 只消费三个 committed `paddleocr_*_official_report_v2` ArtifactId。显式 Packager/Importer Core 接收用户选择的 Det/Rec/System 原始官方报告，以及各自的 SnapshotId 或 committed Snapshot ArtifactId；它重新校验 Snapshot manifest、逐文件 SHA-256、官方 backend/schema/mode 与报告-Snapshot 绑定，从 `det_gt*.txt` / `rec_gt*.txt` 已提交事实计算不可填写的 sampleCount，并生成同时包含原报告、规范化报告和 lineage 的三种 committed Artifact 及导入 Evidence。System lineage 绑定本次 Det/Rec 规范化报告哈希；System 缺少真实 accuracy 时精确返回 `BackendUnsupported` / `ocr_report_import.system_accuracy_unsupported` 且反向补偿为零提交。GUI/Worker 已统一为 `importOcrOfficialReportsV2` 与 `runOcrAcceptanceWorkflowV2` 两步，Worker 独占根任务、取消和唯一终态，结果事件只返回报告/Evidence ArtifactId；旧 `runCustomerOcrAcceptance` Socket 链路已物理删除。Acceptance 只有在 `customer_domain`、足量样本和 Det hmean、Rec accuracy/CER、System accuracy 全达标时才生成 `productionAccepted=true`。Python Adapter 未修改；当前实现不替代客户数据采集，也不能声明客户域已具备生产能力。 | 2026-07-16 Packager、Acceptance、Worker、AITrainStudio、OCR Worker 与 UI 目标构建成功；成功导入/验收、缺 accuracy 零 Artifact、public 拒绝、取消、旧命令 Unsupported 与路径边界定向 CTest 5/5 通过（62.81 秒）。 |
| V2-600～601 | 进行中（路由、查询与两类只读 Presenter） | 新增只读 `WorkspaceRouter`，Sidebar 通过统一路由发出导航，UI 测试不再访问 MainWindow 私有控件；新增 `ProjectQueryServiceV2`，只返回 V2 Storage 中的 Task、Metric、committed Artifact、Workflow Run/Step 和项目汇总。`TaskArtifactPresenterV2` 与 `ProjectSummaryPresenterV2` 均不接收 Worker 原始消息、不写数据库。其余页面 Presenter 尚未建立。 | `aitrain_delivery_acceptance_ui_tests`、`aitrain_v2_application_tests`、`aitrain_task_artifact_presenter_v2_tests`、`aitrain_project_summary_presenter_v2_tests`。 |
| V2-602 | 已完成（只读汇总与项目写入已迁移） | “项目/总览”统计卡片及下一步提示统一由 `ProjectSummaryPresenterV2` 消费 SQLite V2 聚合事实；项目创建、初始化和 Dataset Catalog 查询均使用 V2 Storage/Application Service，不再通过 `ProjectRepository` 写入 legacy 数据库。ViewModel 不暴露 Artifact 路径或 Worker payload。 | `aitrain_project_summary_presenter_v2_tests`、`aitrain_delivery_acceptance_ui_tests`；统一构建通过。 |
| V2-604 | 进行中（训练输入身份化与跨任务 Snapshot 已完成） | 训练入口明确为单 Worker/单活动任务；Worker 忙时拒绝新训练，不再先在 GUI 建立 legacy queued Task 或 ExperimentRun。请求使用 Dataset/Version/Snapshot/Artifact 四重身份，Storage 在 Workflow 创建时原子保存 `dataset_snapshot` 外部输入绑定，限制跨任务 Artifact 的输入与输出所有权。GUI/Worker 拒绝 dataset/sample/Python/trainer/checkpoint/preflight 裸路径，部署样本仅接受 Snapshot 包内相对路径并由 Core 重验。训练交付报告和 Evidence 保存 producer→consumer lineage 与双哈希。旧裸路径复现入口已停用；V2 复现服务与不可达 GUI 源码物理删除仍待后续切片。 | `aitrain_v2_storage_tests`、`aitrain_v2_application_tests`、`aitrain_ocr_segmentation_worker_tests`、`AITrainStudio` 与 UI 目标。 |
| V2-605 | 已完成（任务与产物页 V2 化） | “任务与产物”页的任务列表和详情从 `ProjectQueryServiceV2` 读取；页面展示 V2 Task、committed Artifact 文件清单及 SHA-256/字节数、Metric、Workflow Step，不读取 legacy 数据库或 Worker payload。V2 Artifact 不暴露裸路径，旧打开目录/复制路径/导出等按钮已物理移除。 | `aitrain_task_artifact_presenter_v2_tests`、`aitrain_delivery_acceptance_ui_tests`。 |
| V2-606 | 进行中（Model Registry 只读 Presenter 已接入） | `ProjectQueryServiceV2` 与 `ModelRegistryPresenterV2` 只输出 ModelPackageId、Task/Snapshot/Artifact lineage、模型合同、runtime routes、limitations、verified 与时间，不暴露 Artifact Store 路径。模型库 V2 表、推理和部署选择器统一消费该 Presenter；运行时不再查询或渲染 legacy ModelVersion/Evaluation/Pipeline 路径记录，旧裸路径模型对比被明确停用。旧四类页面控件与不可达实现仍待物理删除。 | `aitrain_model_registry_presenter_v2_tests` 与 `aitrain_delivery_acceptance_ui_tests` 2/2 通过；`AITrainStudio` 构建通过。 |
| V2-607 | 进行中（环境自检 V2 已迁移） | 环境页统一派发 `runEnvironmentCheckWorkflowV2`，Worker 在受控 staging 探测后由 Core 执行 `ValidateEnvironmentFacts → RenderEnvironmentReport` 两步 EvidenceRequired 工作流，提交环境事实、脱敏报告和 Evidence。响应只返回 Task/Workflow/Artifact ID 与摘要；GUI 由 `EnvironmentCheckPresenterV2` 按 TaskId 查询已校验 committed 报告，不消费检查 payload、不获得机器路径，也不再逐行写 legacy `environment_checks`。旧 `environmentCheck` Socket 命令/事件已删除并纳入 Unsupported 回归。设置页、交付证据汇总和 Failure Catalog 仍待继续迁移。 | 2026-07-16 Core、Worker、AITrainStudio 与三组相关测试目标构建成功；Worker、Presenter/Core 与 UI 定向 CTest 3/3 通过（39.76 秒）。 |
| V2-608 | 已完成（WorkerClient 非阻塞生命周期与 V2 控制传输） | GUI Worker 启动不再 `waitForStarted`；进程退出不再使用 `processEvents`/`waitForReadyRead` 排空循环；Socket 清理不再同步等待。WorkerClient 与 Worker 的外层请求/事件已使用 Protocol V2 envelope 和独立双向 sequence，旧 `JsonProtocol` 不再承担该产品控制面。启动失败与协议拒绝均异步唯一收口。窗口关闭的完整受控退出页、payload 业务命令和 signal 到 Presenter 的迁移仍待后续阶段完成。 | `aitrain_delivery_acceptance_ui_tests`、`aitrain_ocr_segmentation_worker_tests`、`aitrain_platform_tests`、`AITrainStudio` 构建。 |
| V2-610 | 进行中（黑盒 UI 测试） | 删除 `#define private public`，为关键控件和 Router 提供 objectName/只读状态；九页及主要 Tab、长路径和后台任务压力覆盖仍待补齐。 | `aitrain_delivery_acceptance_ui_tests`。 |
| V2-700～702 | 已完成 | License Generator 私钥改为 Windows DPAPI 当前用户保护并收紧 DACL，不再保存、展示或导出明文；主程序使用 DPAPI 可信 UTC 检测超过 5 分钟的明显回拨。自动化覆盖正常/永久/过期、篡改、错误公钥、机器码、无效日期、回拨与受保护数据损坏。文档明确纯离线授权不能抵御管理员级完整篡改。 | `aitrain_license_security_tests`。 |
| V2-703 | 已完成（旧训练/裸路径链路物理删除） | GUI 训练强制解析 `TrainingWorkflowProfileV2`；Worker 已物理删除 `startTrain`、随机训练、伪 checkpoint、旧 Python Trainer、`runLocalPipeline` 及裸模型命令。`ProjectRepository`、`JsonProtocol`、`TaskModels`、`WorkerRequests`、V1 ProductWorkflow 实现和旧独立 smoke CLI 均已删除，主控制面统一为 Protocol V2。 | `aitrain_ocr_segmentation_worker_tests`、`aitrain_delivery_acceptance_ui_tests`、全量 CTest。 |

当前边界：模型导入和 ONNX Runtime 推理已具有 GUI → Worker → V2 Workspace/Manifest 的受控链路；训练已建立八步编排，并完成官方 YOLO Detection/Segmentation/OBB、SMP、Anomalib PatchCore/EfficientAD 及 PaddleOCR Det/Rec 的真实产品边界与 Worker 端到端自动化验收。PaddleOCR System 保持独立 Det+Rec 官方组合 wiring，不被误计为任一单组件训练交付或客户域验收。主控制外层已经是 Protocol V2，Annotation、Runtime Delivery、Data Quality、Dataset Conversion/Snapshot/Split、OCR Acceptance、Diagnostics Bundle、Environment Check 与 Training 业务 payload 已收口到 ArtifactId/登记 ID；Training 通过持久化外部输入绑定跨任务消费 committed Snapshot，部署样本只用包内相对路径，运行环境由 Worker 控制。Training GUI 的 legacy Task/Metric/Artifact/Experiment 双写已切断，Model Registry 运行时只读 V2 Presenter。V2 复现服务、项目初始化写侧及旧不可达源码仍待迁移和物理删除。推理与有限外部诊断/环境 probe 仍为 Worker 进程内同步调用，只能在单次调用前后观察取消。在剩余 GUI Presenter、payload 业务协议和旧路径删除完成前，不得宣称 V2 已替代全部生产链路。

## 5. 阶段 0：恢复可靠基线

### 5.1 阶段目标

在开始大规模重构前，先消除当前稳定 UI 崩溃和不稳定 Worker 测试，建立可以信任的门禁。阶段 0 不扩展产品功能。

### 5.2 任务清单

#### V2-000：冻结基线与问题清单

具体任务：

- 记录当前分支、编译器、Qt、CMake、Python 环境和 SDK 状态。
- 保存一次完整 `harness-check` 输出和失败日志。
- 将已确认问题按 `P1/P2` 和所属模块登记到阶段跟踪表。
- 为快照截断、输出污染、模型误识别、进程残留和 UI 崩溃建立可复现说明。

交付物：

- `docs/v2-refactor-baseline.md`
- 阶段 0 初始门禁日志

验收：

- 所有已知问题都有文件位置、触发条件和期望行为。
- 不把首次 Worker 测试失败简单归为偶发问题。

#### V2-001：定位并消除 UI 页面切换崩溃

具体任务：

- 用调试器、PageHeap 或等价内存工具定位 `showPage -> updateModelRegistry` 访问冲突。
- 删除 `QStackedWidget` 中“删除 placeholder 后立即插入页面”的生命周期模式。
- 为九个工作区建立稳定页面容器，页面容器在 MainWindow 生命周期内不销毁。
- 重型数据允许延迟加载，但 QWidget 本身保持稳定。
- 用 `QPointer` 或明确父子所有权处理跨页面控件引用。
- 禁止页面切换路径调用嵌套 `processEvents()`。

交付物：

- 稳定的 Workspace Router/Page Registry
- 页面快速切换压力测试

验收：

- 九个页面循环切换 1,000 次不崩溃。
- `aitrain_delivery_acceptance_ui_tests` 单独运行和全量运行均通过。
- 不通过增加 sleep、扩大超时或跳过测试来规避崩溃。

#### V2-002：清除 Worker 测试时序污染

具体任务：

- 为每个 Worker 测试使用唯一 server name、taskId、requestId 和临时目录。
- 测试结束时检查 Worker、Python adapter 和测试 fixture 进程均已退出。
- 清理跨测试复用的环境变量、QSettings 和静态协议 sequence。
- 为进程启动、ready、terminal、退出分别设置有意义的断言。
- 去掉只依赖大超时等待的测试结构。

交付物：

- 稳定的 Worker 测试夹具
- 测试结束进程泄漏检查工具

验收：

- OCR/语义分割 Worker 测试连续运行 20 次无失败。
- 测试完成后不存在残留 `aitrain_worker` 或 fixture Python 进程。

#### V2-003：建立关键行为特征测试

需要先锁定的行为：

- 官方 YOLO、SMP、Anomalib、PaddleOCR 后端边界。
- OBB 只允许当前支持的部署路线。
- SMP 不支持 NCNN/TensorRT。
- Anomaly 不声明 C++ ONNX/TensorRT/NCNN runtime。
- 任务取消、失败和完成的 UI 可见结果。
- 数据集校验失败不得启动训练。
- Artifact、Metric、Evaluation、Model Version 的基本查询结果。

说明：特征测试只锁定产品边界，不锁定将被删除的 V1 API 和错误行为。

#### V2-004：收紧 Harness

具体任务：

- 删除测试失败后的自动重试判定，或将重试结果仅作为诊断信息且最终仍失败。
- 保存每个测试的独立日志。
- 增加可选的连续三次运行入口。
- 增加进程残留和临时 staging 残留检查。

验收：

- 任意首次测试失败都会阻断门禁。
- 连续三次门禁能够输出每次独立结果。

#### V2-005：建立架构决策记录

至少新增以下 ADR：

- ADR-001：V2 不兼容策略。
- ADR-002：模块依赖方向。
- ADR-003：任务状态和事件模型。
- ADR-004：不可变 Artifact 与原子提交。
- ADR-005：Worker/Python 进程树所有权。
- ADR-006：Model Manifest 强制策略。
- ADR-007：失败分类模型。

### 5.3 阶段删除项

- 不稳定测试中的全局共享 server name。
- UI 测试中的 `#define private public`，如果阶段 0 能同时完成公开测试接缝；否则最迟在阶段 6 删除。
- 通过 sleep 或自动重试掩盖时序问题的逻辑。

### 5.4 阶段退出标准

- 完整 `harness-check` 连续三次全绿。
- UI 快速切换和 Worker 重复运行压力测试通过。
- 所有 V2 ADR 完成评审。
- 未修改产品算法范围。

## 6. 阶段 1：Domain、Protocol V2 与 SQLite V2

### 6.1 阶段目标

建立 V2 的身份、状态、协议和持久化基础。阶段完成后，最小 fake Worker 任务可以通过新链路完整执行。

### 6.2 建议新目标

```text
aitrain_domain
aitrain_application
aitrain_protocol_v2
aitrain_storage_v2
```

`aitrain_domain` 只依赖 QtCore，不依赖 QtWidgets、QtSql、ONNX、NCNN、TensorRT 或 Python。

### 6.3 任务清单

#### V2-100：定义强类型标识符和值对象

新增：

- `TaskId`
- `RequestId`
- `MessageId`
- `ArtifactId`
- `DatasetId`
- `DatasetVersionId`
- `SnapshotId`
- `ModelPackageId`
- `WorkflowRunId`

规则：

- 禁止使用空 ID。
- ID 解析失败返回显式错误。
- 日志和 JSON 序列化统一。
- 不允许在业务代码中回退到全局当前任务。

测试：

- 生成、解析、比较、哈希、JSON round-trip。
- 空值和非法值拒绝。

#### V2-101：定义统一任务状态机

状态：

```text
Created -> Queued -> Starting -> Running
Running -> CancelRequested -> Canceled
Running -> Succeeded
Running -> Failed
Starting -> Failed/Canceled
```

具体任务：

- 将合法迁移集中到 Domain。
- 定义终态幂等规则。
- 定义崩溃恢复规则：非终态任务在应用重启后标记为 `Failed(ProcessInterrupted)`。
- 定义取消请求与最终取消的区别。

测试：

- 全部合法迁移。
- 全部非法迁移。
- 重复终态事件。
- 取消和完成竞争。

#### V2-102：定义统一失败分类和诊断模型

建立 `FailureCode`、`FailureDetails`、`SuggestedAction`：

- InvalidRequest
- InvalidDataset
- ArtifactIncomplete
- BackendUnsupported
- RuntimeNotImplemented
- DependencyMissing
- SdkMissing
- HardwareUnsupported
- ArtifactIncompatible
- ProcessCrashed
- ProtocolViolation
- Timeout
- Canceled
- InternalError

要求：

- Domain 保存稳定 code，不保存本地化文本。
- GUI 中文文本和报告文本由统一 Catalog 生成。
- TensorRT 软件未实现不得映射为 HardwareUnsupported。

#### V2-103：设计并实现 Protocol V2 Envelope

必填字段：

```json
{
  "protocol": 2,
  "messageId": "uuid",
  "requestId": "uuid",
  "taskId": "uuid",
  "sequence": 1,
  "kind": "task.progress",
  "timestamp": "UTC ISO-8601",
  "payload": {}
}
```

具体任务：

- 定义 command/event kind 常量和 payload schema。
- 实现最大消息尺寸。
- 实现逐请求 sequence 验证。
- 实现 messageId 去重。
- 实现 requestId/taskId 关联检查。
- 未知协议版本、未知事件和非法 payload 必须失败。
- 控制事件和日志事件使用不同限额。

测试：

- 正常 round-trip。
- 缺字段、类型错误、超大消息。
- 重复、乱序、跨任务注入。
- 未知版本和未知事件。

#### V2-104：建立 SQLite V2 Schema

直接创建新数据库，不迁移 V1。

核心表：

- `projects`
- `datasets`
- `dataset_versions`
- `dataset_snapshots`
- `tasks`
- `task_events`
- `task_metrics`
- `artifacts`
- `artifact_files`
- `model_packages`
- `evaluation_reports`
- `workflow_runs`
- `workflow_steps`

具体要求：

- 启用 `PRAGMA foreign_keys=ON`。
- 启用 WAL。
- 设置 busy timeout。
- 所有引用字段建立外键和索引。
- 时间统一保存 UTC。
- 关键枚举增加 CHECK 约束。
- 复杂 JSON 只保存扩展详情，不作为唯一可查询状态。

验收测试：

- 孤儿 metric/artifact 插入失败。
- 删除受引用记录时行为明确。
- 并发读写和 busy timeout 行为可预期。
- 外键、索引和 pragma 在每次连接中生效。

#### V2-105：实现 Repository Transaction 与 Unit of Work

需要提供原子用例：

- 创建任务并追加 Created 事件。
- CAS 更新状态并追加状态事件。
- 提交 Artifact 和 Artifact Files。
- 提交 Workflow Step 和输出 Artifact。
- 注册 Model Package 及来源关系。

当前落实：`StorageV2` 已以 schema v5 保存规范化查询字段、完整 Manifest JSON，以及数据集/版本/快照到 Artifact 的关系；注册模型包时会校验 Manifest、来源 Artifact 所属任务及 SHA-256 文件归属，不匹配的模型包无法入库。旧 v2 schema 同样按破坏性重构策略拒绝打开。

禁止 GUI 分多次独立调用完成一个业务事务。

#### V2-106：实现 TaskCoordinator

职责：

- 创建任务和请求。
- 管理队列。
- 启动 Worker。
- 验证 Worker 事件关联关系。
- 将事件持久化。
- 处理取消、崩溃、超时和重复终态。
- 向 Presenter 发布只读任务快照。

禁止：

- Presenter 直接写数据库。
- WorkerClient 直接更新任务状态。
- MainWindow 保存活动任务 ID。

#### V2-107：实现 Capability Planner

输入完整组合：

- capability
- task type
- dataset format
- training backend
- evaluation backend
- export route
- runtime route

输出不可变 `ExecutionPlan` 或明确失败。

GUI 和 Worker 都调用同一 Planner。Worker 必须重新规划并核对计划摘要，不能信任 GUI 已校验的结果。

#### V2-108：完成 Fake Worker 垂直切片

最小流程：

```text
GUI/Application 创建任务
-> Protocol V2 启动 Fake Worker
-> progress/metric/artifact
-> succeeded/failed/canceled
-> SQLite V2 查询任务结果
```

Fake Worker 只存在测试目标，不随产品包发布。

### 6.4 阶段删除项

- V1 Protocol 编码、解码和 requestId 兼容逻辑。
- V1 SQLite Schema 和全部历史迁移代码。
- `paused` 历史迁移。
- `state_.training.currentTaskId` 及等价全局活动任务字段。
- WorkerClient 对 terminal event 和 process finished 的双重状态写入。
- GUI 内直接调用 `ProjectRepository::updateTaskState` 的路径。

### 6.5 阶段退出标准

- Fake Worker 垂直切片全绿。
- 状态更新全部通过 TaskCoordinator 和 CAS。
- 数据库不能产生孤儿记录。
- Protocol V2 能拒绝乱序、重复和跨任务消息。
- V1 数据库不能被 V2 打开，错误信息清楚提示删除重建。

## 7. 阶段 2：Artifact Store 与 Dataset V2

### 7.1 阶段目标

消除所有半成品、旧输出残留和不完整快照。数据校验、拆分、转换和快照统一采用格式驱动及原子产物。

### 7.2 任务清单

#### V2-200：实现 Artifact Store

目录约定：

```text
project/artifacts/<task-id>/
  staging/
  committed/
  artifact-manifest.json
```

具体任务：

- 创建任务唯一 staging 目录。
- 禁止覆盖已有 committed 目录。
- 写入文件时记录相对路径、大小和 SHA-256。
- 校验必需文件。
- 使用同卷原子重命名提交。
- 提交成功后再写 SQLite。
- 失败时标记 abandoned，不注册为可用 Artifact。

#### V2-201：实现 staging 恢复和清理

- 应用启动时扫描 abandoned/staging 目录。
- 正在运行任务的 staging 不清理。
- 已不存在任务或已终态失败的 staging 可安全清理。
- 清理动作生成诊断日志，不静默删除 committed 产物。

#### V2-202：定义 Dataset Driver 接口

接口至少包含：

```text
detect
inspect
validate
planSplit
materializeSplit
snapshot
```

所有方法接收：

- CancellationToken
- ProgressSink
- DiagnosticSink

#### V2-203：拆分各数据格式驱动

分别实现：

- YOLO Detection
- YOLO Segmentation
- YOLO OBB
- Semantic Mask
- Anomaly Folder
- PaddleOCR Det
- PaddleOCR Rec

每个驱动独立保存格式规则、问题 code 和测试 fixture。删除单一超大 Validator 中的任务分支。

#### V2-204：重构拆分为 Plan + Materialize

`planSplit` 只生成不可变清单并完成：

- 随机种子记录。
- 样本唯一性检查。
- basename 冲突检查。
- train/val/test 泄漏检查。
- 标签/掩码配对检查。
- 目标相对路径安全检查。

`materializeSplit` 只按清单写 staging，不能自行重新随机。

验收：

- 同一输入、种子和比例产生相同计划。
- 重复运行产生新的不可变 Artifact，不污染旧输出。
- 取消和复制失败不产生 committed 结果。

#### V2-205：实现 Snapshot V2

Manifest 必须包含：

- `complete=true`
- Dataset Driver 和版本。
- 数据集格式。
- 类别定义。
- 所有相关文件的规范化相对路径。
- 每个文件的大小和 SHA-256。
- 总文件数和总字节数。
- 根哈希。

规则：

- 采用流式遍历和增量 Manifest 写入。
- 不允许静默截断。
- 达到安全上限时返回 `FileLimitExceeded` 并阻止训练。
- 取消时不得生成完整快照记录。

重点测试：

- 超过 20,000 文件。
- 第 20,001 个文件变化。
- 文件新增、删除、重命名和内容变化。
- 遍历中取消。

#### V2-206：修复 Semantic Mask 像素语义

格式契约：

- `L` 模式读取原始 8-bit 灰度值。
- `P` 模式读取 palette index。
- RGB/RGBA mask 默认拒绝，除非未来明确增加颜色到类别映射。

C++：

- Indexed8 使用 scanline 原始字节。
- 不再使用 `qGray(mask.pixel())` 读取 class ID。

Python：

- 使用 `np.asarray(Image.open(path))` 读取原始 ID。
- 不得先 `.convert("L")`。

测试：

- 调色板颜色与索引不相等的 PNG。
- ignore index。
- class ID 越界。
- C++ 与 Python 读取结果一致。

#### V2-207：重写数据转换输出

- COCO/VOC/YOLO/X-AnyLabeling 转换全部写 Artifact staging。
- 转换前生成输出计划和冲突报告。
- 不允许以删除已有目标的方式覆盖。
- 转换完成后运行目标 Driver 校验。
- 只有目标校验通过才允许 commit。

#### V2-208：增加大数据集和故障注入测试

覆盖：

- 大量小文件。
- 大文件。
- 路径包含中文、空格和长文件名。
- 磁盘写失败。
- 目标文件被占用。
- 复制到一半取消。
- Worker 被杀死后恢复清理。

本机自动化已完成上述稳定可注入矩阵：大量小文件使用受限规模夹具，大文件使用 32 MiB 稀疏文件验证 1 MiB 分块哈希，不制造巨大真实磁盘占用；磁盘耗尽、报告目标锁定和提交目标占用通过明确 I/O seam 注入，失败后均无 committed Artifact。恢复测试还验证不会删除活动任务 staging 或既有 committed Artifact。真实物理磁盘耗尽及跨机器、跨文件系统的锁语义保留为外部平台验收，不据此宣称已覆盖所有 Windows/文件系统实现。

### 7.3 阶段删除项

- `collectFilesRecursive(..., maxFiles)` 静默截断实现。
- 所有直接写用户复用输出目录的拆分和转换函数。
- `copyFileReplacing` 覆盖式拆分路径。
- `DatasetValidators.cpp` 中已迁移的格式分支。
- Semantic Mask 的 `qGray` class ID 读取。
- Python Semantic Mask 的 `.convert("L")` class ID 读取。

### 7.4 阶段退出标准

- 快照完整性测试通过。
- 任意失败和取消都不能产生 committed Artifact。
- 同一拆分重复执行不会残留或复用旧文件。
- 所有数据格式均由独立 Driver 处理。
- C++ 与 Python Semantic Mask 读取一致。

## 8. 阶段 3：进程监管与 Python Adapter V2

### 8.1 阶段目标

保证取消、Worker 崩溃和 GUI 退出时，官方训练/推理后端及其全部后代进程都能退出。

### 8.2 任务清单

#### V2-300：实现 Windows Job Object 封装

新增 `ProcessTreeSupervisor`：

- 创建 Job Object。
- 设置 `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`。
- 将 adapter 根进程加入 Job。
- 禁止不受控 breakaway。
- 读取进程退出码和基本资源统计。
- Worker 异常退出时由 Job 自动清理后代进程。

#### V2-301：统一取消状态机

取消流程：

```text
CancelRequested
-> CancellationToken
-> adapter 协作取消
-> 宽限期
-> terminate
-> 强制关闭 Job
-> Artifact staging 标记 abandoned
-> Canceled terminal event
```

要求：

- 完成和取消竞争时只产生一个终态。
- 取消不是 Failed。
- 超时强杀也保留 `Canceled(force=true)` 详情。

#### V2-302：建立 Python Adapter 结构化事件通道

建议使用 Worker 创建的本机 loopback 端口：

- 只监听 `127.0.0.1`。
- 使用随机一次性 token。
- 端口和 token 通过环境变量或请求文件传入。
- Adapter 发送 Protocol V2 子协议事件。
- stdout/stderr 只作为原始日志。

需要处理：

- token 错误。
- 连接超时。
- 连接断开。
- 消息超限。
- adapter 先退出但未发送终态。

#### V2-303：提供 Python Adapter SDK

统一 Python API：

```text
emit_log
emit_progress
emit_metric
emit_artifact_candidate
emit_completed
emit_failed
check_canceled
run_child_process
```

`run_child_process` 负责：

- 日志落盘。
- 周期性取消检查。
- 退出码记录。
- 命令和环境摘要脱敏。

#### V2-304：迁移 Ultralytics Adapter

覆盖：

- Detection
- Segmentation
- OBB
- official ONNX export
- official `val()` evaluation

验收：

- fake official API 测试。
- 最小 1 epoch smoke。
- 训练中取消无残留进程。

#### V2-305：迁移 SMP 与 Anomalib Adapter

SMP：

- 继续只声明 ONNX Runtime 产品部署。
- 采用 Dataset Driver V2 的 Semantic Mask 契约。

Anomalib：

- 保持 Worker-managed Python runtime。
- 不新增 C++ ONNX/TensorRT/NCNN anomaly 声明。

#### V2-306：迁移 PaddleOCR Det/Rec/System Adapter

- Det/Rec 训练、导出和官方推理使用 Adapter SDK。
- System 使用官方 `predict_system.py`。
- 全部 `subprocess.Popen` 由统一 SDK 托管。
- 日志、命令、退出码和官方报告进入 Artifact staging。
- OCR 隔离环境路径必须显式记录。

#### V2-307：增加进程树与崩溃测试

测试场景：

- Adapter 启动孙进程。
- 孙进程忽略普通 terminate。
- GUI 请求取消。
- Worker 被直接 kill。
- GUI 退出导致 Worker 断开。
- Adapter 崩溃但孙进程继续运行。

验收：

- 每个场景结束后进程树为空。
- staging 未提交。
- SQLite 只有一个终态。

#### V2-308：增加资源和日志限制

- 控制单条日志长度。
- 限制内存缓冲区。
- 大日志直接流式写文件。
- GUI 只保留尾部窗口。
- 记录 CPU/GPU/内存采样时不得阻塞训练。

### 8.3 阶段删除项

- Worker 中随机指标计时训练和伪 checkpoint。
- 仅 terminate/kill 直接 Python 根进程的旧实现。
- Python Adapter 通过 stdout 混合传输结构化事件的实现。
- 分散在各 Adapter 中的重复 `Popen` 轮询逻辑。
- 依靠 GUI 超时杀 Worker 作为主要取消机制的路径。

### 8.4 阶段退出标准

- 所有官方 Adapter 使用统一 SDK 和事件通道。
- 取消、崩溃和 GUI 退出均无残留后代进程。
- 所有日志有尺寸限制和完整文件 Artifact。
- 产品包不包含 fake/mock trainer。

## 9. 阶段 4：Model Package 与 Runtime V2

### 9.1 阶段目标

删除运行时模型类型猜测，建立模型导入、Manifest、能力探测和运行时适配器体系。

### 9.2 任务清单

#### V2-400：定义 Model Manifest V2

必需字段：

- Model Package ID。
- model family。
- task type。
- source backend。
- source task/snapshot。
- 来源 Artifact SHA-256 与产物内相对入口路径。
- 输入输出节点和 tensor layout。
- input shape 或动态 shape 约束。
- preprocessing。
- postprocessing/decoder。
- class names。
- opset 和导出工具版本。
- 支持的 runtime route。
- 已验证状态和限制。

Manifest 缺失或无效时，模型状态为 `Unclassified`，不能进入产品推理和部署验证。

#### V2-401：实现模型导入用例

流程：

```text
读取模型和同目录官方报告
-> 提取候选元数据
-> 用户确认 task/model family
-> 运行安全探测
-> 生成 Manifest
-> 提交 Model Package Artifact
-> 注册数据库
```

自动识别只能提供候选，不能直接决定产品模型类型。

#### V2-402：定义 Runtime Adapter 接口

接口：

```text
probe
validateModel
infer
benchmark
deploymentValidate
```

统一 `RuntimeStatus`：

- Available
- RuntimeNotImplemented
- SdkMissing
- DependencyMissing
- HardwareUnsupported
- ArtifactIncompatible

#### V2-403：迁移 ONNX Runtime

- Detection、YOLO Segmentation、OBB、SMP 分别使用明确 decoder。
- decoder 只由 Manifest 选择。
- 删除 shape 启发式模型族判断。
- 外部模型没有 Manifest 时必须先导入。

重点测试：

- 无 Sidecar 的 `[1,84,8400]` YOLO ONNX 不会被当成 OCR。
- 二值分割输出不会被当成 OCR Det。
- Manifest 与真实 tensor 不一致时明确失败。

#### V2-404：迁移 NCNN Runtime

- 只注册已验证的 YOLO Detection/Segmentation 路线。
- OBB、SMP、Anomaly 和 OCR 不得通过默认分支进入 NCNN。
- 将 blob name、decoder 和输入尺寸写入 Manifest。
- 不支持 layer 或模型转换失败归类为 ArtifactIncompatible/BackendUnsupported。

#### V2-405：迁移 TensorRT Runtime 与导出

- SDK 探测、engine build、runtime inference 分开报告。
- 当前没有实现的 decoder 返回 RuntimeNotImplemented。
- GPU compute capability 不满足才返回 HardwareUnsupported。
- SDK 不存在返回 SdkMissing。
- engine 和当前 GPU 不兼容返回 ArtifactIncompatible。

不得再使用统一 `hardware-blocked` 覆盖所有原因。

#### V2-406：实现 Runtime Capability Matrix

由 Capability Planner 提供可查询矩阵，GUI、Worker、环境页和报告共用。

矩阵至少表达：

- Supported。
- UnsupportedByProduct。
- RuntimeNotImplemented。
- RequiresSdk。
- RequiresHardware。
- RequiresExternalEvidence。

#### V2-407：重构环境检查

- 删除 LibTorch 检查。
- 分开检查 SDK、DLL、硬件、decoder 实现和目标 Artifact。
- 环境页只展示与注册能力相关的依赖。
- 修复建议由 Failure Catalog 生成。

#### V2-408：模型运行时契约测试

建立 golden fixtures：

- YOLO Detection。
- YOLO Segmentation。
- YOLO OBB。
- SMP Semantic Segmentation。
- 不可分类 ONNX。
- tensor/manifest 不匹配模型。

### 9.3 阶段删除项

- `inferOnnxModelFamily()` 启发式产品判定。
- 通过输出 rank/shape 直接认定 OCR 的逻辑。
- 环境中的 LibTorch 行和相关文案。
- 各 Workflow 内自行判断 runtime 可用性的分支。
- TensorRT 软件缺失映射为 hardware-blocked 的逻辑。

### 9.4 阶段退出标准

- 无 Manifest 模型不能进入产品运行时。
- Runtime 状态分类准确且由所有模块共用。
- YOLO、SMP、OBB golden fixture 通过。
- 外部 ONNX 导入流程能明确处理不确定模型。

## 10. 阶段 5：Workflow 与 Evidence V2

### 10.1 阶段目标

将超大 Workflow 文件拆成可组合、可取消、可重试的步骤，并以统一 Evidence 模型生成报告。

### 10.2 任务清单

#### V2-500：定义 Workflow Run 与 Step 模型

Step 状态：

```text
Pending -> Running -> Succeeded/Failed/Canceled/Skipped
```

每一步记录：

- 输入 Artifact ID。
- 输出 Artifact ID。
- 执行后端。
- 参数摘要。
- 开始/结束时间。
- Failure Code。
- 重试次数。

#### V2-501：实现 Workflow Runner

- 默认顺序执行，不先引入通用 DAG 调度器。
- 支持步骤级取消。
- 支持从最后一个成功的不可变 Artifact 继续。
- 下游步骤只能消费 committed Artifact。
- Step 失败后默认停止，不隐式降级到其他后端。

#### V2-502：迁移训练流水线

标准步骤：

```text
ValidateDataset
CreateSnapshot
Train
Evaluate
Export
DeploymentValidate
RegisterModel
RenderDeliveryReport
```

规则：

- 训练必须绑定 complete Snapshot。
- Evaluation 使用官方规定后端。
- Export 只能消费成功训练或已导入 Model Package。
- Model Registration 必须关联来源 Task、Snapshot、Evaluation 和 Runtime 状态。

#### V2-503：迁移推理/部署流水线

状态：已完成。GUI 推理页和部署页共用 Worker `runRuntimeDeliveryWorkflowV2`，GUI 只提交 ModelPackageId、runtime route、样本图和选项；Worker/Core 独占六步状态、终态和 Evidence。旧单步命令及 Core/GUI 单步派发 API 已删除，结果只暴露结构化状态和 ArtifactId。

步骤：

```text
ImportOrResolveModel
ValidateManifest
RunInferenceSmoke
Benchmark
DeploymentValidate
RenderDeliveryReport
```

能力边界：Benchmark 仅为本机固定样本 smoke timing。ONNX Runtime 单次同步 `infer` 进入后不能中途抢占，取消请求在该次调用返回后才收口。NCNN V2 仅覆盖产品矩阵允许、且 Manifest 明确提供 tensor/blob/decoder 合同的 Detection/Segmentation；OBB、SMP、异常检测、OCR 和未知 decoder 必须返回精确不支持/未实现状态。TensorRT V2 当前只完成 probe、合同校验和 SDK/依赖/硬件/未实现分类，官方 YOLO decoder 与真实 `infer` 尚未实现，不能宣称 V2 TensorRT 推理或验收通过。

#### V2-504：定义 Evidence Bundle

统一包含：

- 项目与任务身份。
- Dataset Version/Snapshot。
- 后端与环境。
- 训练参数和官方版本。
- Metrics。
- Artifact Inventory。
- Runtime Status。
- Evaluation。
- Benchmark。
- Limitations。
- Failure/Suggested Action。

Evidence 只保存事实，不在 Renderer 中重新判断产品状态。

#### V2-505：实现报告 Renderer

分别实现：

- JSON Renderer。
- Markdown Renderer。
- HTML Renderer。
- Model Card Renderer。

所有 Renderer 使用同一 Evidence Bundle，状态和限制必须一致。

#### V2-506：迁移数据质量工作流

- 将质量规则拆成独立检查器。
- 问题样本、修复建议和严重级别使用稳定 code。
- X-AnyLabeling 清单作为 Artifact 输出。
- 不直接修改用户标签。

#### V2-507：重构 Annotation Session

Manifest 必须验证：

- schemaVersion。
- kind。
- dataset version/snapshot。
- dataset format。
- output artifact。
- createdAt。

将现有 `synced` 改为准确语义：

- Inspected。
- ChangesDetected。
- NoChanges。
- InvalidSession。

#### V2-508：迁移客户 OCR 和交付验收工作流

- OCR Det/Rec/System 证据只取官方 Adapter 输出。
- Public/generated smoke 与客户域证据分类保存。
- 无客户域数据时不得生成 production accepted 状态。
- TensorRT、NCNN、SMP、Anomaly、OBB 限制由 Capability Matrix 生成。

#### V2-509：删除超大 Workflow companion 逻辑

按迁移完成情况逐步删除：

- Snapshot 大函数。
- Quality 大函数。
- Benchmark 运行时分支。
- Acceptance 状态拼装。
- Delivery 中重复 Evidence 判断。

### 10.3 阶段退出标准

- 每个 Step 可单独测试和取消。
- Workflow 不读取 GUI 全局状态。
- JSON/Markdown/HTML 对同一状态给出一致结论。
- 失败步骤不会产生下游 Model Package 或 accepted 报告。
- ProductWorkflow 超大文件被拆除或缩减为薄入口。

## 11. 阶段 6：GUI Presenter 化

### 11.1 阶段目标

把 MainWindow 缩减为工作台壳，页面只通过 Application Service 和 Presenter 访问任务与数据。

### 11.2 任务清单

#### V2-600：建立 MainWindow Shell

MainWindow 只保留：

- Sidebar。
- Top Status Bar。
- QStackedWidget。
- Workspace Router。
- 全局窗口级命令。

禁止包含：

- SQL。
- Worker 消息解析。
- 模型类型判断。
- Workflow 业务分支。
- Artifact 文件解析。

#### V2-601：建立页面 View/Presenter 约定

每个页面拆分：

```text
PageView：控件、信号、展示
PagePresenter：订阅 Application Service、构建 ViewState、执行用例
```

Presenter 不互相访问控件。跨页面导航和选中对象通过 Workspace Router/Selection Service 传递。

#### V2-602：迁移项目与总览页面

- Project Service 负责创建项目和初始化 SQLite V2。
- Dashboard Query Service 提供聚合只读数据。
- View 不直接访问 Repository。

进展：只读查询与 Presenter/View 接线切片已完成。`ProjectQueryServiceV2` 只从
SQLite V2 的已持久化事实汇总任务状态、已提交 Artifact、数据集/版本/快照、
Model Package、Workflow 与 Evidence 可用数量；`ProjectSummaryPresenterV2` 将这些
事实映射为不含 Artifact 裸路径的只读 ViewModel，“项目/总览”统计卡片和下一步提示
均已迁移，且不解析 Worker payload 或读取 legacy `project.sqlite`。项目创建和初始化
写用例仍通过 legacy `ProjectRepository`；其他页面与写操作也尚未全部迁移。

#### V2-603：迁移数据集与质量页面

- Dataset Presenter 调用 Driver/Workflow 用例。
- 数据集列表使用 Dataset Version 和 Snapshot 状态。
- 拆分、转换、快照和质量任务全部进入 Task Center。
- 页面关闭或切换不影响后台任务。

#### V2-604：迁移训练页面

- 配置表单由 Capability Planner 生成可选项。
- 启动前展示 Execution Plan 摘要。
- 保留进度、指标曲线、日志、任务/产物说明和取消入口。
- 不再保存单一 current task；允许按 TaskId 订阅选中任务。

#### V2-605：迁移任务与产物页面

- 使用 Task Query Service。
- 任务事件、Metric、Artifact、Workflow Step 分区展示。
- Artifact 只展示 committed 结果。
- abandoned staging 只在诊断模式展示。

#### V2-606：迁移模型库与部署验证页面

- Model Package 替代裸路径模型。
- 显示 Manifest、来源 Snapshot、Evaluation、Runtime Matrix 和限制。
- 外部模型先进入 Import Wizard。
- 不允许页面自行调用模型族启发式判断。

#### V2-607：迁移环境、设置与交付证据页面

- 环境状态来自 Runtime/Backend Probe Service。
- 修复建议来自 Failure Catalog。
- 系统设置不再显示无产品用途的 LibTorch。
- 交付证据只汇总结构化 Evidence。

#### V2-608：移除 GUI 阻塞调用

- 删除 `waitForStarted()`。
- 删除 `waitForReadyRead()` GUI 循环。
- 删除嵌套 `processEvents()`。
- Worker 启动、连接和退出全部转为异步信号。
- 窗口关闭使用异步取消和受控退出页面。

#### V2-609：重构国际化

- UI 可见文本使用 `tr()`/`QStringLiteral` 中文源文案。
- 英文只由 `.ts/.qm` 提供。
- 删除 fallback 大字典。
- 删除遍历 Widget 当前文本进行翻译的机制。
- 设置继续采用重启后生效。
- 增加翻译缺失检查。

#### V2-610：重建 UI 测试

- 删除 `#define private public`。
- Presenter 使用 fake Application Service 测试。
- Router 提供只读当前页面状态。
- 使用 objectName 验证关键控件。
- 页面测试覆盖空状态、错误状态、长路径和后台任务状态。
- 增加 1280×820 非全屏 walkthrough。
- 增加九页及各主要 Tab 快速切换压力测试。

### 11.3 阶段删除项

- MainWindow companion 中已迁移的业务逻辑。
- GUI 直接 Repository 写入。
- GUI 直接解析 Worker Protocol payload。
- 全局 current task、current conversion task 等任务身份字段。
- `waitForStarted`、嵌套 `processEvents`、退出 drain 循环。
- `translateWidgetTree` 和 fallback translation map。
- UI 测试的 private/public 宏。

### 11.4 阶段退出标准

- MainWindow 只承担 Shell 职责。
- 所有页面通过 Presenter/Application Service 工作。
- GUI 线程无同步 Worker 等待。
- 九页快速切换压力测试通过。
- 1280×820 非全屏 walkthrough 无关键按钮裁切。
- 中英文界面均无混合回译和乱码。

## 12. 阶段 7：授权、安全、V1 清理与发布验收

### 12.1 阶段目标

完成安全加固，删除全部 V1 残留，更新文档和打包内容，并建立 V2 发布候选门禁。

### 12.2 任务清单

#### V2-700：加固 License Generator 私钥存储

状态：已完成。Generator 使用 Windows DPAPI 当前用户范围与受保护 DACL 持久化 `.aitrainkey`，不展示或导出明文私钥，且安装开关继续默认关闭。

- 使用 Windows DPAPI 加密私钥。
- 文件 ACL 仅允许当前用户。
- 默认不展示和导出明文私钥。
- 明文导出需要显式确认和风险提示。
- Generator 不进入客户发布包。
- 主程序只保留公钥。

#### V2-701：增加时钟回拨检测

状态：已完成。应用启动与注册码激活共用 DPAPI 保护的最近可信 UTC，使用 5 分钟校时容差；回拨和受保护状态损坏均作为独立授权状态拒绝通过。

- 保存 DPAPI 保护的最近可信 UTC 时间。
- 当前时间明显早于可信时间时返回 ClockRollbackDetected。
- 允许小范围系统校时容差。
- 文档明确纯离线授权不能抵御管理员级完整篡改。

#### V2-702：补齐授权自动化测试

状态：已完成。Windows QtTest 覆盖 DPAPI 私钥往返/损坏、正常/永久/过期、token 篡改、错误公钥、机器码不匹配、无效日期、回拨容差与可信时间损坏。

覆盖：

- 正常签发与验证。
- token 篡改。
- 错误公钥。
- 机器码不匹配。
- 已过期和永久授权。
- 无效日期。
- 时钟回拨。
- DPAPI 文件损坏。

#### V2-703：执行 V1 代码清理

使用 `rg` 和构建目标检查以下内容已删除：

- Protocol V1。
- SQLite V1 和迁移。
- legacy plugin/plugin_id。
- pause/paused。
- Worker 随机训练和 scaffold checkpoint。
- LibTorch 产品环境检查。
- ONNX 模型族猜测。
- hardware-blocked 字符串业务判断。
- currentTaskId 全局状态。
- 直接覆盖数据输出。
- fallback 翻译大字典。
- `#define private public`。

#### V2-704：更新架构与开发文档

至少同步：

- `HARNESS.md`
- `AGENTS.md`
- `docs/harness/current-status.md`
- `docs/harness/project-context.md`
- `docs/harness/implementation-checklist.md`
- `docs/harness/quality-gates.md`
- `docs/developer-architecture.md`
- `docs/user-guide.md`
- `docs/operations-runbook.md`
- `docs/acceptance-runbook.md`
- Worker/Python Protocol 文档

所有文档使用中文，产品后端边界保持准确。

#### V2-705：更新打包布局

- 只打包 Protocol V2 Worker 和 Adapter。
- 不包含 Fake Worker、测试 fixture、License Generator 私钥或 V1 数据库工具。
- 检查 Python Adapter SDK、官方 adapter、runtime DLL 和 Manifest schema。
- 安装后首次启动创建 SQLite V2。

#### V2-706：执行功能回归矩阵

至少覆盖：

- YOLO Detection 训练、官方评估、ONNX 推理和报告。
- YOLO Segmentation 训练、官方评估和 ONNX 推理。
- YOLO OBB 官方训练、评估和 ONNX Runtime 部署。
- SMP 训练、评估、ONNX Runtime 推理。
- Anomalib PatchCore/EfficientAD Worker-managed Python 路线。
- PaddleOCR Det/Rec/System 官方路线。
- 数据集校验、拆分、转换、快照和质量报告。
- 模型导入、Manifest、Benchmark、Deployment Validation。
- 取消、崩溃恢复和 Artifact 清理。

#### V2-707：执行发布候选门禁

本地必过：

```powershell
.\tools\encoding-check.ps1
.\tools\harness-check.ps1
.\tools\acceptance-smoke.ps1 -CpuTrainingSmoke
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
git diff --check
```

额外要求：

- 完整 harness 连续三次通过。
- 进程残留检查通过。
- staging 残留检查通过。
- UI walkthrough 通过。
- package smoke 通过。

外部证据继续按真实状态记录：

- Clean Windows package acceptance。
- 客户域 OCR。
- 客户/目标域 OBB、Anomaly、SMP 精度。
- 明确重开时的 package-root TensorRT。

没有返回证据的项目不得标记为通过。

### 12.3 阶段退出标准

- V1 源码、测试、文档和打包入口全部删除。
- 授权安全测试通过。
- 本地发布候选门禁连续三次全绿。
- 发布包不包含开发私钥、测试 fixture 或废弃后端。
- 外部验收状态与实际证据一致。

## 13. 跨阶段测试矩阵

| 风险 | 最早落地阶段 | 必须覆盖的测试 |
|---|---|---|
| UI 生命周期崩溃 | 0 | 快速页面切换、销毁、非全屏 walkthrough |
| Worker 测试污染 | 0 | 重复运行、唯一 socket、进程残留 |
| 任务串线 | 1 | requestId/taskId 不匹配、乱序、重复事件 |
| 状态竞争 | 1 | cancel/completed 竞争、重复终态、Worker 崩溃 |
| SQLite 孤儿记录 | 1 | 外键、事务、CAS、并发读写 |
| 输出半成品 | 2 | 复制失败、磁盘失败、取消、强杀、恢复清理 |
| 快照截断 | 2 | 超过 20,000 文件、尾部文件变化 |
| Palette mask 误读 | 2 | P 模式索引与调色板亮度不同 |
| Python 子进程残留 | 3 | 孙进程、忽略 terminate、Worker kill |
| ONNX 类型误识别 | 4 | 无 Sidecar YOLO、二值分割、不确定模型 |
| TensorRT 状态误归因 | 4 | decoder 缺失、SDK 缺失、硬件不支持 |
| 报告状态不一致 | 5 | JSON/Markdown/HTML golden 对比 |
| GUI 阻塞 | 6 | Worker 慢启动、慢退出、长日志 |
| 授权绕过 | 7 | 篡改、过期、机器码、时钟回拨 |

## 14. 推荐实施切片

虽然允许破坏性重构，仍应保持每个切片可编译、可测试。推荐顺序：

1. 先让旧基线稳定全绿。
2. 建立 V2 Domain/Protocol/SQLite，并完成 Fake Worker 垂直切片。
3. 选择“数据集校验”作为第一个真实 V2 Job Handler。
4. 迁移 Artifact Store、Snapshot 和数据拆分。
5. 选择 YOLO Detection 作为第一个完整训练到部署垂直切片。
6. 依次迁移 YOLO Segmentation、OBB、SMP、Anomaly、PaddleOCR。
7. 完成 Workflow/Evidence 后再迁移 GUI 页面。
8. 最后删除全部 V1 和执行发布验收。

每个真实能力的迁移完成标准都应包含：

```text
Dataset Driver
-> Capability Plan
-> Worker Job Handler
-> Process Supervision
-> Artifact Commit
-> Repository Record
-> Workflow Step
-> Presenter/UI
-> Tests/Smoke/Docs
```

## 15. 暂不纳入本轮重构

- Qt 6 升级。
- 云调度、远程 Worker 和多人协作。
- 动态插件或插件市场。
- 图像分类、姿态、YOLO-World、YOLOE、3D、RGB-D、视频和时序模型。
- Anomaly C++ ONNX/TensorRT/NCNN runtime。
- SMP NCNN/TensorRT。
- OBB NCNN。
- 没有真实实现和证据的新 TensorRT decoder。

这些范围如果未来加入，必须通过新的产品范围决策和 Capability Planner 扩展，不得借 V2 重构顺带接入。

## 16. V2 最终完成定义

V2 重构只有在以下条件全部满足后才能标记完成：

- 完整门禁连续三次通过，且没有自动重试掩盖失败。
- 任意任务事件都能追溯到唯一 taskId/requestId。
- 任意失败、取消或进程崩溃都不会留下可用半成品。
- 数据快照完整；无法完整时训练被阻止。
- 所有模型都有经过验证的 Model Manifest。
- 取消后不存在 Worker、Adapter 或官方工具后代进程。
- Runtime 状态准确区分软件、SDK、依赖、硬件和 Artifact 问题。
- MainWindow 只负责工作台 Shell。
- SQLite 具备外键、索引、事务和幂等状态更新。
- JSON、Markdown、HTML 和 GUI 使用同一 Evidence/Failure 事实。
- V1 协议、数据库、模型猜测、全局任务状态和脚手架训练已物理删除。
- 产品算法边界与当前 harness 一致。
- 未完成的外部验收继续明确标记为待证据，不伪装成通过。
