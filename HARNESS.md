# AITrain Studio Harness

Harness 的目标是让 AITrain Studio 更适合 vibe coding：每次改动都能快速获得项目上下文、明确任务边界、遵守架构约束，并用固定命令验证结果。

这里的 harness 不是测试框架本身，而是一套工程护栏：

- 明确当前系统是什么。
- 明确哪些文件负责什么。
- 明确任务应该如何拆分。
- 明确不能随意跨越的边界。
- 明确每次提交前必须运行的检查。
- 明确 AI 协作时应该读取哪些上下文。

## 快速入口

新任务开始前，先读：

1. `docs/harness/project-context.md`
2. `docs/harness/current-status.md`
3. `docs/harness/implementation-checklist.md`
4. `docs/harness/quality-gates.md`
5. 相关任务文档或当前用户需求

如果是 UI 任务，再读：

- `docs/harness/ui-guidelines.md`

如果是让 AI 执行任务，优先使用：

- `docs/harness/task-brief-template.md`
- `docs/harness/agent-prompts.md`

## 新对话自动入口

为了让新 AI 对话自动进入 harness 工作流，仓库提供了常见 AI 编码工具会识别的入口文件：

- `AGENTS.md`
- `.cursorrules`
- `.github/copilot-instructions.md`

这些文件都指向本 harness，并要求新任务开始前自动读取项目上下文、实现清单和质量门禁。

## 一键检查

在 PowerShell 或 VSCode task 中运行：

```powershell
.\tools\harness-check.ps1
```

这个脚本会：

- 初始化新版 MSVC 环境。
- 配置 `build-vscode`。
- 编译项目。
- 运行 CTest。

只查看项目上下文：

```powershell
.\tools\harness-context.ps1
```

## 编码与终端规则

- 仓库文本统一按 UTF-8 处理；中文或中英混排文件不能按系统 ANSI/GBK 猜测读取。
- 在 Windows PowerShell 中读取项目文本时，必须显式指定 UTF-8，例如 `Get-Content -Encoding UTF8`。
- 终端输出出现 `鐨勭洰鏍` 这类乱码时，先用 UTF-8 重新读取或做字节级检测，不要直接判断文件已损坏。
- 列出可能包含中文文件名的 Git 路径时，使用 `git -c core.quotepath=false ...`，或配置 `git config --global core.quotepath false`，避免文件名显示为 `\345\220...` 转义。
- 新增或修改源文件、文档、翻译文件时保持 UTF-8；除非是有意编码迁移，否则保留原文件的 BOM / no BOM 风格。

## 当前架构硬边界

- GUI 只做交互、状态展示和任务编排。
- 长任务必须进入 `aitrain_worker`，不能直接堵塞 GUI 线程。
- 插件通过 `IModelPlugin` 及其子接口扩展，不把模型逻辑塞进 `MainWindow`。
- SQLite 元数据通过 `ProjectRepository` 管理，不在 UI 里手写散落 SQL。
- 当前训练核心仍是 scaffold，不要在文档或 UI 中宣称已经完成真实 YOLO/OCR 训练。
- 真实训练实现必须分阶段接入，并用测试或小数据集验证。

## Definition of Done

一次 vibe coding 任务完成至少满足：

- 代码能编译。
- 相关测试通过。
- 没有破坏已有功能入口。
- UI 文本没有乱码。
- 新增行为有明确验收方式。
- 如果只是 scaffold，必须清楚标注，不伪装成完整能力。
