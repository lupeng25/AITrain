# 实施检查清单

## 本轮界面迁移完成记录（2026-09-12）

- 三个日常入口、单层业务视图、数据版本与报告选择、历史训练配置复用及结果定位已迁移。
- 数据重开、两图像样本计数、质量终态、图片预览与产物清单读取状态已通过回归。
- 高级参数取消、Escape 返回、八后端入口与导航草稿保持已通过回归。
- 全量 Harness 35/35 组通过；原生 UI 回归 33 项通过；缩放验证的原生/离屏范围见 `../design/ui-workbench-implementation.md`。

## 追加收尾完成记录（2026-09-12）

- 全项目目录搜索及过滤游标隔离通过；覆盖首屏之外的数据、模型、任务与证据。
- 设置支持浅/深主题即时切换和持久化；英文静态文案、长导航及分类换行通过检查。
- 草稿跨进程、延迟建页、项目切换、重建、失效绑定、损坏格式及确认丢弃通过回归。
- 最终 Harness 35/35 组通过；最终缩放与英文补验、历史失败纠正见 `../design/ui-workbench-closeout.md`。

## 稳定化基线

- 新项目必须使用 Schema 13 单行 `project_meta`；Schema 12 只读识别后返回 `SchemaRebuildRequired`。
- 项目打开必须先取得 canonical root 的 Owner Lease；Worker 只能通过认证控制通道和 Worker Lease 使用 `openForWorkerChild()`。
- Product Contract 是 backend、Python Profile、Runtime route 和三条数据集转换路线的唯一编译期事实。
- Artifact rename 后不得合成业务失败；只允许完成 journal v2 固化的 completion action 或进入恢复。
- 当前无 Resume、任意转换 outputPath、任务级 Python 解释器和 Runtime 自动回退。

## 开始前

- 明确任务类型：UI、Core、Worker、Capability、Dataset、Training、Docs。
- 读取 `docs/harness/project-context.md`。
- 在 Windows PowerShell 中读取项目文本时使用 `-Encoding UTF8`；不要用默认编码读取中文文档后根据乱码输出做判断。
- 枚举 Git 路径时如涉及中文文件名，使用 `git -c core.quotepath=false ...`。
- 如果是 UI 任务，读取 `docs/harness/ui-guidelines.md`。
- 确认是否会修改公共接口：
  - `CapabilityRegistry.h`
  - `WorkerProtocol.h`（ 控制面）
  - `ProjectStore.h`（SQLite  元数据）
  - `WorkflowResult.h`（跨模块结构化结果）
- 确认是否需要新增测试。

## 开发中

- 保持变更范围小。
- 新增或修改文本文件保持 UTF-8，除非明确做编码迁移，否则保留原文件 BOM / no BOM 风格。
- 不把训练逻辑塞进 GUI。
- 不在 Worker 中直接依赖 GUI 类型。
- 不让能力后端绕过公共接口访问主窗口。
- 新增 UI 文本使用 `QStringLiteral`。
- 新增表格和状态必须有空状态。
- 新增长任务必须有日志、进度和失败信息。

## 完成前

必须运行：

```powershell
.\tools\harness-check.ps1
```

如果只改文档，可以不用编译，但需要说明未运行构建的原因。

如果只改状态/手册/验收文档，至少运行：

```powershell
git diff --check
.\tools\harness-context.ps1
```

如果本次任务涉及编码、中文乱码、文件名或文档读取问题，还需要用显式 UTF-8 读取至少一个相关中文文件确认显示正常，例如：

```powershell
.\tools\encoding-check.ps1
Get-Content -LiteralPath HARNESS.md -Encoding UTF8 -TotalCount 5
git -c core.quotepath=false ls-files | Select-String -Pattern 'AITrainStudio'
```

如果改动涉及当前项目状态、交付证据/验收、客户域 OCR、部署验证或样本复核，需要同步检查 `docs/harness/current-status.md`、`docs/harness/project-context.md`、`docs/acceptance-runbook.md`、`docs/product-roadmap-local-training-platform.md` 和 `docs/user-guide.md` 是否一致。

如果修改 UI 布局，还需运行 `aitrain_delivery_acceptance_ui_tests` 的 `workbenchViewsFitStandardWindows` 用例，检查 1280×820、1366×768 窗口中各业务模式的可见按钮边界。设置 `AITRAIN_CAPTURE_UI_DIR` 可输出窗口截图，并逐页检查字体、长文本、空态和主要操作。

应增加数据重新打开、目录分页、训练草稿与高级参数取消、任务终态和对应报告定位的定向回归。业务页面不使用整页滚动；长表、日志及报告在自身区域滚动。UI 验证不能替代硬件或客户域精度验收。

## 回答用户前

说明：

- 改了什么。
- 关键文件在哪里。
- 验证命令和结果。
- 没有完成的部分或刻意保留的 scaffold。
