# AITrain Studio V2 重构基线

日期：2026-07-15  
分支：`codex/v2-destructive-refactor`

## 已确认问题与处置

| 优先级 | 模块 | 问题 | 当前处置 |
|---|---|---|---|
| P1 | GUI / 评估报告 | 页面刷新期间清空表格会触发选择信号，重入页面预览更新并造成 UI 访问冲突。 | 已在 `EvaluationReportView::clear()` 屏蔽表格信号；UI 定向和全套测试通过。 |
| P1 | 测试 Harness | 旧 Harness 会把失败后的重试成功视为门禁通过，掩盖时序问题。 | 已删除该通过判定；首次失败直接阻断并保留日志。 |
| P1 | Worker 协议 | V1 协议缺少强类型任务/消息身份、严格 kind 和跨任务注入防护。 | V2 协议库与 Fake Worker 切片已建立；生产 Worker 尚待迁移。 |
| P1 | 元数据 | V1 SQLite 为渐进迁移表结构，外键和原子任务事件约束不足。 | V2 新库拒绝打开 V1 数据库，启用 WAL、外键、busy timeout 和 CAS。 |
| P2 | 任务编排 | GUI、WorkerClient 和 Repository 都可影响任务终态。 | V2 TaskCoordinator 已建立测试切片；生产调用路径尚待迁移。 |
| P2 | 数据/模型/流程 | 快照截断、旧输出污染、模型类型猜测、复杂 Workflow 文件职责过重。 | 列入阶段 2、4、5 的破坏性替换任务，尚未实施。 |
| P2 | 进程 | Python 后代进程所有权和取消边界不统一。 | 列入阶段 3，尚未实施。 |

## 基线验证

2026-07-15 已执行：

```powershell
.\tools\harness-check.ps1
ctest --test-dir build-vscode --output-on-failure
.\tools\encoding-check.ps1
git diff --check
```

结果：Harness 与独立 CTest 均为 16/16 通过；编码与 diff 检查通过。

## 边界说明

本文件不是发布结论。生产 GUI、`aitrain_worker`、V1 SQLite 和现有 Python Adapter 仍在迁移前状态，不能因为 V2 基础切片通过就宣称重构或生产替换已经完成。
