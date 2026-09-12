# 工作台改造收尾清单

日期：2026-09-12。用户已要求完成剩余收尾。本文件补充原方案和首轮实施记录，不用首轮测试通过代替本轮验收。

## 执行步骤

1. 完整目录搜索
   - 在 Storage Repository、ProjectStore、Query 和 Presenter 中贯通搜索条件，先过滤全项目元数据再分页。
   - 数据集支持名称和格式，模型支持模型族、任务及来源，训练与任务支持类型、状态和关联配置；明确搜索范围。
   - 搜索参数使用绑定值，搜索条件纳入游标身份；改变条件重置分页，返回目录保留条件和选择。
   - 覆盖首屏以外的结果、中文名称、空结果、字面通配符、不同搜索条件之间的游标误用。
2. 浅色与深色外观
   - 在 AppStyle 维护统一调色板和样式，提供设置入口并保存用户选择。
   - 同时覆盖表格、输入、菜单、禁用状态、报告阅读器、图表和弹窗，不只修改主背景。
   - 在两种外观中检查正文、选中行、按钮、焦点与状态颜色的可读性。
3. 跨应用重启的训练草稿
   - 按持久化 ProjectId 隔离保存基础字段、后端、高级参数和完整数据绑定；不把运行状态当作草稿。
   - 使用应用偏好持久化保存小型界面草稿；启动和项目切换后重新核对项目身份、快照身份及样本成员。
   - 明确恢复、丢弃和保存状态。取消高级编辑保留原草稿；损坏、旧版本或失效绑定不得自动启动任务。
   - 使用独立进程验证保存与恢复，并覆盖项目隔离、重建后的旧草稿、无效文件及不同后端。
4. 英文界面
   - 盘点工作台全部用户可见文案，接入已有 Qt 翻译链路并补齐英文资源。
   - 保持协议值、对象身份、模型名称、用户输入和原始后端日志的原意，不翻译机器标识。
   - 检查格式占位符和动态状态；以英文实际创建窗口和打开操作表单，检查截断和按钮溢出。
5. 验收与文档
   - 核对原方案验收矩阵，补充搜索、主题、进程重启和英文回归。
   - 运行受影响测试、全量 Harness、编码及差异检查；保留每次返回的真实结果。
   - 覆盖 100%、125%、150% 缩放；受实际显示器限制的大尺寸原生窗口单独记录，不冒称已经验证。
   - 更新当前状态、界面规范、用户手册、实施记录及本清单。

## 边界

继续保持 Qt 5.12、Qt Widgets 工作台、Schema 13、既有 Worker 协议和八个官方训练后端。草稿恢复不是 Resume，不改变训练、推理或客户域验收合同。客户数据精度及外部干净机器验收不以本次 UI 检查替代。

## 返回结果

批准方案及本轮收尾已完成。以下记录返回的验证结果。

| 收尾项 | 实现与回归 |
|---|---|
| 全项目目录搜索 | 数据集、模型、训练、任务和验收报告搜索贯通 Repository → Store → Workspace → Query → Presenter → 页面；参数绑定，查询条件绑定游标；覆盖首屏以外记录、中文、字面 `%`/`_` 和错误游标 |
| 主题 | 设置中保存浅色/深色选择，立即应用；正文、表格、输入、菜单、图表、状态和日志统一使用调色板；长导航提供省略及完整提示，设置分类可换行 |
| 训练草稿 | 以 ProjectId 保存版本化 JSON，最大 256 KB；自动保存、显式保存、恢复和确认丢弃；保存稳定参数值与四重绑定，未应用高级参数保留原值 |
| 恢复校验 | 数据版本、样本成员与后端合同重新核对；项目切换和延迟建页均恢复；失效绑定要求重选，损坏草稿保留默认参数，重建隔离旧身份；不自动启动任务 |
| 英文 | 工作台静态文案接入 Qt 翻译，资源包含 1384 条上下文译文；增加文案覆盖及占位符检查，英文双主题实际创建所有页面与模式 |

原方案中“第一批分页后再增加搜索”“跨重启不作为第一阶段承诺”等描述是分阶段安排，本次收尾已覆盖这些项目。正式后端、协议及外部验收边界保持原合同。

## 关键文件

- `src/core/include/aitrain/domain/Pagination.h`：目录筛选条件。
- `src/core/src/storage/StoragePagination.h` 与各目录 Repository：参数绑定、全目录筛选及游标隔离。
- `src/core/src/workflow/ProjectQueryService.cpp`：只读目录与项目身份查询。
- `src/app/src/AppStyle.cpp`、`StatusPill.cpp`、`MetricsWidget.cpp`、`SettingsPageController.cpp`：主题与设置。
- `src/app/src/TrainingPageControllerDraft.cpp`、`ApplicationSettingsService.cpp`：草稿持久化和恢复；`MainWindowActions.cpp`、`MainWindowSummaries.cpp`：避免项目激活或延迟建页覆盖草稿。
- `src/app/src/WorkbenchTranslation.h`、`LanguageSupport.cpp`、`src/app/translations/aitrain_en_US.ts`：翻译入口与英文资源。
- `tests/tst_delivery_acceptance_ui.cpp`、目录 Presenter 测试、`tst_application.cpp`、`test_workbench_translations.py`：集成验收。

## 本轮验证日志

日志根目录：`.deps/ui-closeout-20260912`。本轮沿用首轮记录中的 Qt 5.12.9、MSVC 14.44、隔离 Python 和 ONNX Runtime 构建环境。

- `new-tests-final.txt`：独立进程草稿及双主题英文定向回归，4 项通过，0 失败（含初始化/清理）。
- `harness-final.log`：最终全量 Harness 通过，35/35 组 CTest、0 失败，1001.05 秒。
- `ui-native-100.txt`：100% 原生全套界面回归，35 项通过，0 失败。
- `ui-native-125.txt`、`ui-native-150.txt`：分别 5 项通过，0 失败；前者覆盖三个尺寸，后者覆盖 1024×700。
- `ui-offscreen-150.txt`：150% 三尺寸中英文双主题，4 项通过，0 失败。
- `ui-last-layout-100.txt`、`ui-last-layout-125.txt`、`ui-last-layout-150.txt`：最终英文分类换行及两个主题的原生补验，分别 3 项通过，0 失败。
- `ui-last-layout-150-offscreen.txt`：最终大尺寸离屏补验，3 项通过，0 失败。
- `dialog-light.log`、`dialog-dark.log`：使用最终 AppStyle 对象额外检查浅/深主题原生确认弹窗，均正常打开、取消并以 0 退出。
- `harness-before-last-fixes.log` 保留修正前报告搜索按 TaskId 未匹配，以及 20,001 小文件快照用例超过默认 300 秒的结果。最终复跑使用隔离临时目录，并仅在测试进程设置 `QTEST_FUNCTION_TIMEOUT=900000`；测试数据规模及断言不变。构建和 CTest 顺序执行，避免 Windows 测试进程占用待链接文件。
- 编码、架构、英文覆盖及格式占位符检查通过，最终随 Harness 复验；收尾文档再次通过编码、差异及上下文检查。

复验命令沿用 `ui-workbench-implementation.md`；将日志目录替换为本轮目录。原生矩阵运行界面测试并设置 `QT_SCALE_FACTOR=1/1.25/1.5`；大尺寸离屏检查设置 `QT_QPA_PLATFORM=offscreen` 和 `QT_QPA_FONTDIR=C:\Windows\Fonts`。

本轮完整复跑另在当前测试进程中设置如下环境，不修改系统环境或产品超时：

```powershell
New-Item -ItemType Directory -Force .deps/ui-closeout-20260912/tmp | Out-Null
$env:TEMP = "$PWD\.deps\ui-closeout-20260912\tmp"
$env:TMP = $env:TEMP
$env:QTEST_FUNCTION_TIMEOUT = "900000"
```

## 验证边界

生成数据、登记记录及测试适配器用于证明界面、参数与持久化流程；没有新增客户域精度证据。原始后端日志、用户数据名称、模型标识及系统原生文件对话框遵循其原语言。未增加 TensorRT 推理解码器、异常检测 C++ Runtime、Resume、内置标注绘制器或干净 Windows 外部验收。

150% 下超出当前物理工作区的逻辑尺寸只记录离屏结果，不标记为原生显示器验收。草稿存放在当前用户设置中，不是可分享的项目产物；重建后旧身份草稿不会自动套用。
