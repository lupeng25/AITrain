# Agent Prompts

## 通用实现 Prompt

```markdown
请基于当前仓库实现以下任务。开始前先阅读：

- HARNESS.md
- docs/harness/project-context.md
- docs/harness/implementation-checklist.md
- docs/harness/quality-gates.md

要求：

- 保持变更范围小。
- 不破坏现有 VSCode 构建。
- 在 Windows PowerShell 中读取中文文档时使用 `-Encoding UTF8`；Git 枚举中文路径时使用 `git -c core.quotepath=false ...`。
- 长任务不要放进 GUI 线程。
- 生产训练入口只允许官方后端：Ultralytics YOLO detection/segmentation 和 PaddleOCR Det/Rec 官方适配器。
- 不要把已移除的 `tiny_linear_detector`、shipped `python_mock`、小型 PaddleOCR Rec CTC 或 C++ segmentation/OCR scaffold 训练描述为产品后端。
- 如果能力只是 scaffold、smoke、diagnostic helper 或 report-only workflow，必须明确标注。
- 不要在没有返回证据时声称 clean Windows、package-root TensorRT、客户域 OCR 或 unsupported-hardware TensorRT 已通过。
- 完成后运行 `.\tools\harness-check.ps1`。
- 如果涉及乱码或编码，额外运行 `.\tools\encoding-check.ps1`。

任务：

<在这里填写任务>
```

## UI 任务 Prompt

```markdown
请实现一个 UI 改动。开始前先阅读：

- HARNESS.md
- docs/harness/project-context.md
- docs/harness/ui-guidelines.md

要求：

- 保持当前左侧导航 + 顶部状态栏结构。
- 复用 AppStyle、InfoPanel、Sidebar、StatusPill。
- 不回退到 QTabWidget。
- 中文文本不能乱码。
- 检查中文文本时显式按 UTF-8 读取，不要根据 PowerShell 默认编码输出判断文件损坏。
- 完成后运行 `.\tools\harness-check.ps1`。
- 如果涉及乱码或编码，额外运行 `.\tools\encoding-check.ps1`。

任务：

<在这里填写 UI 任务>
```

## Core / Worker 任务 Prompt

```markdown
请实现一个 Core/Worker 改动。开始前先阅读：

- HARNESS.md
- docs/harness/project-context.md
- docs/harness/quality-gates.md

要求：

- 协议变更必须更新测试。
- Worker 失败必须返回明确错误。
- GUI 只消费状态和事件，不承载长任务。
- 完成后运行 `.\tools\harness-check.ps1`。

任务：

<在这里填写 Core/Worker 任务>
```
