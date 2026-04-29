# Quality Gates

## 通用 Gate

所有代码改动必须满足：

- 编译通过。
- 测试通过。
- 不引入乱码。
- 不破坏现有 VSCode 工作流。
- 不把 build 产物加入源码管理。

标准命令：

```powershell
.\tools\harness-check.ps1
```

## UI Gate

UI 改动必须满足：

- 中文显示正常。
- 页面层级清楚。
- 控件间距一致。
- 表格有表头。
- 空状态有提示。
- 长文本不挤压主要操作。
- 训练页仍能显示日志、进度和曲线。

## Core Gate

`src/core` 改动必须满足：

- 公共类型命名稳定。
- JSON 协议可向后兼容，除非明确升级协议版本。
- SQLite schema 改动要有初始化逻辑。
- 任务状态变化不能散落在多个层里重复实现。

## Worker Gate

`src/worker` 改动必须满足：

- GUI 断开时 Worker 能退出或失败。
- Worker 失败时必须发出明确错误消息。
- 长任务必须可取消。
- 进度和日志输出不能依赖 stdout 解析作为唯一通道。

## Plugin Gate

插件改动必须满足：

- `manifest()` 信息准确。
- `datasetAdapter()` 对不支持的格式返回 `nullptr`。
- 插件不能直接依赖 `MainWindow`。
- 插件接口改动必须同步所有内置插件。

## Dataset Gate

数据集相关改动必须满足：

- 错误能定位到文件或行号。
- 校验失败不能允许启动训练。
- 大数据集校验要有截断或进度策略。
- 不修改原始数据，除非用户明确执行转换或划分。

## Training Gate

真实训练能力接入时必须满足：

- 最小数据集可以训练 1 epoch。
- loss 或 accuracy 有可观察指标。
- checkpoint 可保存。
- 失败原因可见。
- 不把模拟训练说成真实训练。

