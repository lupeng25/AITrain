# AITrain 最终架构重构实施计划

## 目标与边界

本轮是上线前破坏性重构，不提供 V1/ 数据库、协议、目录或命名兼容层。旧开发项目直接重新创建。最终目标是让任务终态、事件副作用、Artifact、进程树、GUI 状态和验收工具使用同一套事实源。

固定边界：GUI 只负责交互和展示；训练、数据处理、运行时交付和环境探测全部由 Worker/Core 执行；YOLO、SMP、Anomalib、PaddleOCR 的官方后端边界保持不变；TensorRT、客户域 OCR 和客户域精度没有证据时不得标记通过。

## 最终架构

依赖方向固定为：

```text
foundation → domain → protocol/storage/artifact/process
                             ↓
                    dataset/model/runtime
                             ↓
                         workflow
                             ↓
                capabilities/application/Worker/GUI
```

删除 `aitrain_core`、`aitrain`、`*` target、旧裸路径命令、旧 smoke 脚本和旧双事实源。活动代码、测试、工具和 Harness 文档中不得再出现版本化命名。

项目目录固定为：

```text
<project>/.aitrain/project.sqlite
<project>/.aitrain/artifacts
<project>/.aitrain/artifact-transactions
<project>/.aitrain/scratch
<project>/.aitrain/quarantine
```

## 执行阶段

### 阶段 1：基础命名和模块边界

- 将 Domain、Protocol、Storage、Artifact、Dataset、Model、Runtime、Workflow、Process、Application 拆成独立 target。
- 删除 `aitrain_core` 的反向依赖。
- 将所有  类型、文件、命名空间、测试目标改为最终无版本命名。
- 每次只移动一组模块；移动提交与行为提交分离。

验收：各 target 可独立编译；`architecture-check.ps1` 不再发现 ``、`aitrain_core` 或旧命令。

### 阶段 2：数据库、任务状态和事件事务

- 新建最终 schema 1，不迁移旧数据库。
- 项目名称、根目录和 ID 写入 SQLite，删除 GUI 内存事实源。
- 引入 `TaskTerminalResolver`，规定取消优先规则。
- 引入单一 `applyAdapterEvent` 事务入口，事件、Metric、Artifact link、Workflow Step 和终态必须同事务提交。
- sequence 使用十进制字符串传输，数据库使用有界整数。
- 恢复查询分页循环处理，不使用固定 1000 条上限。

验收：取消竞态、重复事件、乱序事件、副作用失败回滚、request/task 错配和 1500 条恢复测试全部通过。

### 阶段 3：Artifact、路径和数据集身份

- Artifact staging 拒绝 symlink、junction、reparse point 和非普通文件。
- 引入 canonical containment 和 Windows handle 最终路径校验。
- 重建 Artifact commit journal 和幂等恢复逻辑。
- Dataset manifest 统一校验 `rootHash`、`fileCount`、`totalBytes`、driver、format 和逐文件 hash。
- Dataset Conversion 保留源相对目录，大小写折叠冲突在 Plan 阶段失败。
- Runtime Delivery 只消费已提交 sample Artifact，不接受外部裸样本路径。
- `.aitrain/scratch` 以 task owner 清单管理并在恢复时清理。

验收：符号链接越界、manifest 篡改、转换 basename 冲突、样本 TOCTOU、Artifact 各 journal 故障点和崩溃恢复测试通过。

### 阶段 4：Worker、Adapter、进程树和取消

- Worker 和 Adapter 连接都采用候选 socket、token handshake、5 秒握手超时和单一 authenticated socket。
- Worker 启动、ready、退出和取消都增加有限超时。
- 使用受监管 process host，在 Python 创建前加入 Job Object。
- 持续 drain stdout/stderr，日志保留尾部并设置磁盘上限。
- Adapter SDK 默认读取 `AITRAIN_CANCEL_FILE`；YOLO、SMP、Anomalib、OCR 在 batch/epoch/export 边界协作取消。
- Worker 结果从 `finished(bool, message)` 改成明确的 `Succeeded/Failed/Canceled` 结果。

验收：未认证连接占位、第二连接替换、ready 超时、孙进程残留、大日志阻塞、后端协作取消和强杀兜底测试通过。

### 阶段 5：异步项目恢复和 GUI Presenter

- Project Open 使用 `Closed → Probing → Recovering → Ready/Failed` 状态机。
- Artifact journal、Evidence、Interrupted Task 和 scratch 恢复由 Worker 执行。
- GUI 在恢复期间禁用写操作，不在 GUI 线程 hash 或遍历大目录。
- 所有任务终态由 `WorkerResult` 三态驱动。
- 推理展示只消费 Presenter DTO，不读取 Worker payload 或裸路径。

验收：大制品恢复不冻结 UI；取消显示为取消；失败显示失败；项目重启后名称、任务和 Artifact 一致。

### 阶段 6：验收、打包和文档

- 删除旧 `startTrain`、`runCustomerOcrAcceptance`、`semantic-onnx-smoke` 和旧 Phase smoke 引用。
- package smoke 按 `base/onnx/yolo/smp/anomaly/ocr/ncnn/tensorrt` profile 检查，缺少必需依赖必须失败。
- 修复 VS Code Worker 调试配置和 Harness stale path。
- 重写活动架构、状态、验收和训练后端文档。

验收：发布包只包含当前入口；旧命令和旧脚本引用扫描为空；base 及声明的 backend profiles 全部通过。

## 提交纪律

每个任务独立提交，禁止一个提交同时移动文件、修改状态机和修改 GUI。每次提交至少运行定向测试与 `git diff --check`；阶段结束运行：

```powershell
.\tools\harness-check.ps1
```

最终必须通过全量 CTest、Python pytest、architecture-check、encoding-check、package smoke 和 GUI 手工验收。
