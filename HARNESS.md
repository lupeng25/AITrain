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

如果任务较宽或实现量较大，再读：

- `docs/product-roadmap-local-training-platform.md`

归档路线笔记不作为当前下一步计划、阶段状态或验收口径来源。当前阶段状态以 `docs/harness/current-status.md` 为准，宽口径方向以 `docs/product-roadmap-local-training-platform.md` 为准。

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

只检查仓库文本编码：

```powershell
.\tools\encoding-check.ps1
```

## 编码与终端规则

- 仓库文本统一按 UTF-8 处理；中文或中英混排文件不能按系统 ANSI/GBK 猜测读取。
- 在 Windows PowerShell 中读取项目文本时，必须显式指定 UTF-8，例如 `Get-Content -Encoding UTF8`。
- 终端输出出现 `鐨勭洰鏍` 这类乱码时，先用 UTF-8 重新读取或做字节级检测，不要直接判断文件已损坏。
- 列出可能包含中文文件名的 Git 路径时，使用 `git -c core.quotepath=false ...`，或配置 `git config --global core.quotepath false`，避免文件名显示为 `\345\220...` 转义。
- 新增或修改源文件、文档、翻译文件时保持 UTF-8；除非是有意编码迁移，否则保留原文件的 BOM / no BOM 风格。
- `.\tools\harness-check.ps1` 会先运行 `.\tools\encoding-check.ps1`，发现非 UTF-8 文本或未知 NUL 文本时直接失败。
- Harness 和 VSCode 构建任务会设置 `VSLANG=1033`，让 MSVC/CMake 日志优先输出英文，避免本地化编译器提示在终端里变成乱码。

## 当前架构硬边界

- GUI 只做交互、状态展示和任务编排。
- 长任务必须进入 `aitrain_worker`，不能直接堵塞 GUI 线程。
- 能力通过 `BuiltinCapabilityRegistry` 与 Worker/core 边界扩展，不把模型逻辑塞进 `MainWindow`。
- SQLite 元数据通过 `ProjectRepository` 管理，不在 UI 里手写散落 SQL。
- 当前生产训练入口只允许官方/上游后端：Ultralytics YOLO detection/segmentation/OBB、SMP semantic segmentation、Anomalib PatchCore/EfficientAD，以及 PaddleOCR Det/Rec 官方适配器。旧的 C++ tiny detector、segmentation/OCR scaffold 训练、小型 PaddleOCR Rec CTC 和 shipped `python_mock` 已从产品训练路径移除，不得重新作为产品后端描述。
- YOLO 产品边界是“官方 Ultralytics 训练/首次 ONNX 导出/`val()` 评估 + AITrain C++ runtime 推理、benchmark、受支持的部署验证、overlay 和交付报告”；OBB v1 仅承诺 ONNX Runtime 产品部署，NCNN 不属于 OBB v1。SMP 语义分割只承诺 ONNX Runtime 推理、overlay、benchmark 和部署验证，NCNN/TensorRT 导出不属于 SMP 范围。Anomaly v1 使用 Worker-managed Python/Anomalib artifacts，不声明 AITrain C++ ONNX/TensorRT/NCNN anomaly runtime。OCR 产品验收是 PaddleOCR Det/Rec/System 官方报告路径，不使用历史 C++ OCR ONNX wiring 作为当前验收。
- generated/public smoke 只证明接线和 artifact 生成，不证明客户域精度。clean Windows、package-root TensorRT、客户域 OCR 和 unsupported-hardware TensorRT 结论必须有对应外部或客户域证据。
- 仍为 scaffold、smoke、diagnostic 或 report-only 的能力必须清楚标注，不伪装成完整生产能力。

## Definition of Done

一次 vibe coding 任务完成至少满足：

- 代码能编译。
- 相关测试通过。
- 没有破坏已有功能入口。
- UI 文本没有乱码。
- 新增行为有明确验收方式。
- 如果只是 scaffold、smoke、diagnostic 或 report-only，必须清楚标注，不伪装成完整生产能力或客户域验收通过。
