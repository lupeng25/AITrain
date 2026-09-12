# 工作台界面重设计实施记录

日期：2026-09-12。依据：[已批准方案](ui-workbench-redesign.md)。状态：首轮界面迁移、定向回归及全量 Harness 检查完成；追加收尾及最终证据见[收尾记录](ui-workbench-closeout.md)。

## 实施范围

日常侧栏改为数据集、训练、模型。项目管理与概况放入顶栏项目菜单，任务记录为全局工具，验收报告从模型进入，环境与诊断及设置位于工具区。保留 Qt Widgets 壳层、独立 Worker 和现有状态所有权。

| 区域 | 已实施内容 |
|---|---|
| 外壳 | 中性侧栏、统一表格、收缩与省略长标题、返回工作区、活动任务反馈；移除常驻检查器 |
| 项目 | 最近项目、新建与打开、简短项目概况及原有重建确认；不展示目录结构作为主操作内容 |
| 数据集 | 目录、详情、导入、划分、转换、质量、复核、标注和技术模式；完整游标分页与版本选择 |
| 数据结果 | 查询已有审计名称，按安全成员预览图像；图像/样本数与文件数分开；质量终态读取对应版本的已提交报告 |
| 训练 | 记录目录、宽基础配置、单层高级参数分组、监控与日志；草稿保持、高级修改取消恢复、历史配置查看与复制、定位本次产出模型 |
| 模型 | 目录与详情、单独导入表单、已有模型说明的结构化编辑与完整 JSON 工具、分页、来源任务与最近独立验证 |
| 验证与交付 | 合并重复入口，通过模型与数据版本选择器生成现有六步命令，读取已提交可视化结果及报告 |
| 任务 | 单独目录与详情，产物/文件/指标/工作流/预览内容切换；报告定位和文件选择保持 |
| 验收与诊断 | OCR 导入与验收分别显示；数据版本与报告选择器替代 UUID 输入；保留外部证据导入及诊断采集 |
| 环境与设置 | 环境检查明细与顶栏状态同步；设置改用单层分类，保留偏好、能力摘要和许可信息 |

## 数据与运行边界

没有新增数据库 schema、训练算法、Worker 命令或运行时回退。Schema 13、Protocol V2、Artifact Journal V2，以及十四种 Worker 命令的业务边界保持不变。

新增 Query 投影用于数据名称、版本列表、按类别选择产物和模型最近独立验证。名称来自现有工作流审计参数；不单独引入名称写入协议。分页游标绑定查询类别及数据对象，拒绝交叉复用。

正式训练后端仍为 YOLO 检测/实例分割/OBB、SMP、PatchCore/EfficientAD、PaddleOCR Det/Rec。高级参数保留原映射；当前代码没有统一训练参数 schema，因此本次未声称“按 schema 自动生成全部表单”。新界面不启用 Resume。

图像与报告通过 Query 的已提交 Artifact 成员读取，异步结果检查对象身份与读取代次。预览遵守现有 4 MB 读取上限，图片另有解码尺寸限制；超限内容明确提示，不把截断文件显示成完整结果。异步图片/报告预览的完整清单和文件哈希校验已移入后台，GUI 只复制登记元数据。主界面不接收 committed 物理路径，HTML 阅读器不自动加载外部资源。

未读取的产物清单显示“文件清单尚未读取”，读取后统计当前已载入文件数；分页未完成时不冒充完整总量。目录中的模型合同校验与最近独立验证分开；任务完成和报告校验都不等同于客户域精度通过。

## 构建环境

- Qt 5.12.9，MSVC 14.44，C++20。
- 构建目录：`build-ui-analysis-msvc1444`。
- 测试解释器：隔离目录 `.deps/ui-redesign-testenv`，包含 pytest、numpy、Pillow、onnx 与 PyYAML。
- ONNX Runtime 1.24.3：从官方 NuGet 的 `Microsoft.ML.OnnxRuntime` 包提取 Windows x64 SDK，放在 `.deps/sdks/onnxruntime`。
- 下载包 SHA-256：`e4ab3b236c1b803e1e7167fe9a75caa1b82a2fd4ae82abbd130937c07517bcdb`。

初次全量检查因系统 Python 别名不可用及缺少 ONNX Runtime SDK 失败；进程树测试在隔离 Python 下已通过定向复验。上述依赖仅用于工作区构建与验证，没有替换用户的全局 Python 或部署环境。

## 验证记录

| 检查 | 本轮返回结果 | 证据位置 |
|---|---|---|
| 全量 Harness | 35/35 组通过、0 失败；CTest 总耗时 446.24 秒 | `.deps/ui-redesign-20260908/harness-final.log` |
| 原生 UI 全部用例 | 33 通过、0 失败 | `ui-final-native.txt` |
| Artifact Presenter | 9 通过、0 失败，包含异步完整性失败回调 | `presenter-final.txt` |
| 125% 原生 UI | 4 通过、0 失败，三个尺寸与重开项目 | `ui-scale-1.25.txt` |
| 150% 原生 UI | 4 通过、0 失败，1024×700 模式边界与重开项目 | `ui-scale-1.5.txt` |
| 150% 离屏布局 | 3 通过、0 失败，三个逻辑尺寸 | `ui-scale-1.5-offscreen.txt` |
| 编码及架构 | 通过 | `encoding-check.ps1 -IncludeUntracked`、`architecture-check.ps1` |

表中简写文件名均位于 `.deps/ui-redesign-20260908`。这里记录 QtTest 返回的通过数量，包含初始化和清理用例。

UI 回归通过正式 Session 入口重开磁盘项目，验证延迟建页即显示中文名称、两张 PNG 与质量终态；随后检查已提交报告文件选择保持、从已保存工作流复制训练参数、未登记模型时禁止跳转。另有导航往返、八后端合同匹配、高级参数取消与 Escape 返回等回归。版本和报告游标隔离、模型最近独立验证在各自 Query/Workflow 用例中验证。

测试项目使用生成的小图像及登记的训练工作流；Worker 后端流程测试使用测试夹具适配器。它们证明界面、命令和持久化流程，没有新增真实模型训练精度证据。

原生 100% 截图共 98 张，位于 `final-screenshots`，包含各页面模式与真实项目重开后的样本、历史草稿视图。已经查看真实项目、基础训练、高级参数和 OCR 导入等代表性截图，确认中文、字段、选择状态与主按钮可读。

当前显示器无法容纳 150% 下的 1280×820 原生逻辑窗口，Windows 会限制其高度。因此原生 150% 的模式遍历使用 1024×700；三个完整逻辑尺寸另用 Qt 离屏平台和 Windows 字体目录验证。离屏结果证明布局尺寸与绘制，不等同于当前显示器能容纳同样大的原生窗口。

复验入口：

```powershell
$env:PATH = "$PWD\.deps\ui-redesign-testenv\Scripts;$env:PATH"
$env:PYTHON = "$PWD\.deps\ui-redesign-testenv\Scripts\python.exe"
$env:AITRAIN_BUILD_DIR = "build-ui-analysis-msvc1444"
$env:AITRAIN_VCVARS64 = "$PWD\.deps\ui-redesign-20260908\vcvars-1444.cmd"
$env:AITRAIN_QT_ROOT = "D:\Qt\Qt5.12.9\5.12.9\msvc2017_64"
.\tools\harness-check.ps1
.\tools\encoding-check.ps1 -IncludeUntracked
.\tools\harness-context.ps1
```

构建后，设置 `AITRAIN_CAPTURE_UI_DIR` 并运行 `aitrain_delivery_acceptance_ui_tests.exe` 可生成截图；`QT_SCALE_FACTOR` 控制缩放，`AITRAIN_TEST_MINIMUM_WINDOW=1` 只遍历 1024×700。离屏大尺寸检查使用 `QT_QPA_PLATFORM=offscreen` 和 `QT_QPA_FONTDIR=C:\Windows\Fonts`。

## 本轮不扩展的能力

不新增内置标注绘制编辑器，不新增八种后端的真实客户数据精度证明，不将异常检测改成 C++ ONNX 部署，不实现 TensorRT 推理解码器，不增加 Schema 12 迁移或 Resume。英文静态文案、跨应用重启的草稿、全项目目录搜索及深色主题已在用户授权的追加收尾中完成，详见 `ui-workbench-closeout.md`。本节其余后端与外部验收边界继续有效。
