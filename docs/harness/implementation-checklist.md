# Implementation Checklist

## 开始前

- 明确任务类型：UI、Core、Worker、Plugin、Dataset、Training、Docs。
- 读取 `docs/harness/project-context.md`。
- 在 Windows PowerShell 中读取项目文本时使用 `-Encoding UTF8`；不要用默认编码读取中文文档后根据乱码输出做判断。
- 枚举 Git 路径时如涉及中文文件名，使用 `git -c core.quotepath=false ...`。
- 如果是 UI 任务，读取 `docs/harness/ui-guidelines.md`。
- 确认是否会修改公共接口：
  - `PluginInterfaces.h`
  - `JsonProtocol.h`
  - `TaskModels.h`
  - `ProjectRepository.h`
- 确认是否需要新增测试。

## 开发中

- 保持变更范围小。
- 新增或修改文本文件保持 UTF-8，除非明确做编码迁移，否则保留原文件 BOM / no BOM 风格。
- 不把训练逻辑塞进 GUI。
- 不在 Worker 中直接依赖 GUI 类型。
- 不让插件绕过公共接口访问主窗口。
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
Get-Content -LiteralPath HARNESS.md -Encoding UTF8 -TotalCount 5
git -c core.quotepath=false ls-files | Select-String -Pattern 'AITrainStudio'
```

如果改动涉及当前项目状态、交付验收、客户域 OCR、部署验证或样本复核，需要同步检查 `docs/harness/current-status.md`、`docs/harness/project-context.md`、`docs/acceptance-runbook.md`、`docs/product-roadmap-local-training-platform.md` 和 `docs/user-guide.md` 是否一致。

如果修改 UI 布局，还需要至少执行一次非全屏 walkthrough：

```powershell
$pages = @('总览','项目','数据集','样本复核','训练实验','任务与产物','模型库','评估报告','模型导出','推理验证','交付验收','插件','环境','设置')
C:\Users\73200\.codex\skills\qt-gui-walkthrough\scripts\qt_walkthrough.ps1 `
  -AppPath .\build-vscode\bin\AITrainStudio.exe `
  -WorkingDirectory .\build-vscode\bin `
  -OutDir .\.deps\ui-walkthrough `
  -PageNames $pages `
  -Width 1280 `
  -Height 820
```

验收重点：关键操作按钮不能被非全屏首屏裁切；长路径、长状态文本和表格不能造成横向溢出；允许页面纵向滚动。

## 回答用户前

说明：

- 改了什么。
- 关键文件在哪里。
- 验证命令和结果。
- 没有完成的部分或刻意保留的 scaffold。
