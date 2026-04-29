# Implementation Checklist

## 开始前

- 明确任务类型：UI、Core、Worker、Plugin、Dataset、Training、Docs。
- 读取 `docs/harness/project-context.md`。
- 如果是 UI 任务，读取 `docs/harness/ui-guidelines.md`。
- 确认是否会修改公共接口：
  - `PluginInterfaces.h`
  - `JsonProtocol.h`
  - `TaskModels.h`
  - `ProjectRepository.h`
- 确认是否需要新增测试。

## 开发中

- 保持变更范围小。
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

## 回答用户前

说明：

- 改了什么。
- 关键文件在哪里。
- 验证命令和结果。
- 没有完成的部分或刻意保留的 scaffold。

