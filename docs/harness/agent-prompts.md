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
- 长任务不要放进 GUI 线程。
- 如果能力只是 scaffold，必须明确标注。
- 完成后运行 `.\tools\harness-check.ps1`。

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
- 完成后运行 `.\tools\harness-check.ps1`。

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

