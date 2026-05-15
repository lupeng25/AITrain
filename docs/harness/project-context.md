# Project Context

## 项目定位

AITrain Studio 是一个 Windows + NVIDIA GPU 本地视觉训练平台。当前技术方向：

- C++20
- Qt Widgets
- CMake
- SQLite
- 独立 Worker 进程
- Qt 插件系统

目标能力：

- 训练 YOLO 风格检测模型。
- 训练 YOLO 风格分割模型。
- 训练 PaddleOCR Rec 风格字符识别模型。
- 管理数据集。
- 转换模型格式。
- 做推理验证。
- 做样本复核、评估、模型库、交付报告、部署验证、诊断包和客户域 OCR 验收。

训练实现方向：

- 真实模型训练优先通过 Worker 启动独立 Python 子进程实现。
- 若官方 Python 实现可用，优先适配官方实现，而不是自研训练框架。
- 检测和分割优先评估 Ultralytics YOLO；OCR Rec 优先评估 PaddleOCR / PaddlePaddle。
- 官方后端必须显式记录来源、版本和许可证约束。
- C++ 侧继续负责 GUI、Worker 编排、数据集校验、SQLite、ONNX Runtime/TensorRT 推理、打包和部署。
- 不把 Python 嵌入 `MainWindow` 或 GUI 进程。
- 现有 C++ tiny/scaffold 训练保留为 demo、回归测试和平台链路验证后端。

## 当前事实

阶段状态以 `docs/harness/current-status.md` 为准。不要只根据长期路线图末尾的历史“下一步建议”判断当前阶段。

截至 2026-05-15，官方 YOLO 检测/分割、PaddleOCR Det/Rec/System 官方工具链、ONNX Runtime 推理、RTX 4090 TensorRT 验收证据、本地插件 marketplace、环境 profile、数据集质量/快照、数据集格式转换 GUI、评估、benchmark、模型库、交付报告、样本复核、部署验证、诊断包和交付验收 GUI 闭环都已落地到当前本地代码。准确阶段边界仍以 `docs/harness/current-status.md` 为准。

已完成：

- `AITrainStudio.exe` Qt GUI。
- `aitrain_worker.exe` 独立任务进程。
- JSON Lines over `QLocalSocket` 通信。
- SQLite 项目、任务、指标存储。
- 插件接口和三个内置插件骨架。
- 数据集校验初版。
- 数据集转换 GUI 入口：已实现 COCO/VOC/YOLO 转换矩阵的 Worker 编排、表单预检、进度/日志/取消和结果展示；转换产物不自动登记为数据集。
- Tiny detection scaffold 训练、checkpoint、ONNX 导出和 ONNX Runtime 推理链路。
- YOLO 分割 scaffold/baseline 闭环：
  - `SegmentationDataset`
  - `SegmentationDataLoader`
  - polygon-to-mask
  - letterbox 对齐 mask
  - 多 polygon / 多 class mask
  - overlay preview
  - mask preview artifact
  - Worker 端 `maskLoss`、`maskCoverage`、`maskIoU`、`segmentationMap50`
  - scaffold checkpoint
- OCR recognition scaffold/baseline 闭环：
  - `OcrRecDataset`
  - 字符字典加载
  - label encode/decode
  - resize/pad batching
  - Worker 端 `ctcLoss`、`accuracy`、`editDistance`
  - scaffold checkpoint
  - preview artifact
- VSCode 构建、运行、调试配置。
- QtTest 基础覆盖。
- Worker-managed Python Trainer Adapter；`python_mock` 仅作为协议/scaffold fixture。
- 官方 Ultralytics YOLO detection / segmentation 训练、导出和 ONNX Runtime 推理 smoke。
- PaddlePaddle OCR Rec CTC 训练和 C++ ONNX greedy decode；官方 PaddleOCR Det/Rec/System 工具链 smoke。
- PaddleOCR Det DB-style ONNX probability-map 后处理和 Phase 47 真实导出 Det ONNX wiring smoke。
- TensorRT SDK-backed 导出/推理路径和 RTX 4090 D 验收证据；旧 GTX 1060 / SM 61 仍应为 `hardware-blocked`。
- Windows 打包、package smoke、release freeze handoff、离线授权和注册码生成器。
- 本地产品闭环：数据集质量报告、问题样本、X-AnyLabeling 复核清单、snapshot、训练 lineage、评估、benchmark、模型注册、pipeline、交付报告。
- Phase 49 交付闭环：样本复核页、交付验收页、客户域 OCR 验收向导、一键诊断包和导出后部署验证。

未完成：

- Clean Windows package acceptance 仍需要外部返回证据；不能只凭本机结果标记为通过。
- package-root TensorRT rerun 只有在重新打开外部验收时才执行；旧 GPU 的正确状态仍是 `hardware-blocked`。
- 客户域 OCR 生产声明必须使用真实客户/目标域数据和官方报告；public Total-Text、generated smoke、`.deps` 示例只能证明流程。
- NCNN v1 仍是产物存在校验，不是 runtime 推理验收。
- 插件签名、远程 marketplace、账号、支付、云调度和多人协作后置。
- 分类、姿态、OBB、异常检测、YOLO-World、YOLOE 等新算法后端后置。

## 源码地图

| 路径 | 职责 |
|---|---|
| `src/core` | 协议、插件接口、任务模型、SQLite 仓库、dataset/training scaffold、product workflow companion files |
| `src/app` | Qt Widgets GUI |
| `src/app/translations` | GUI 翻译源文件，构建时生成 `.qm` 并随应用安装 |
| `src/license_generator` | 内部 Qt 注册码生成器，用私钥签发绑定机器码的离线注册码 |
| `src/worker` | 长任务隔离进程 |
| `src/plugins/yolo_native` | YOLO 风格插件骨架 |
| `src/plugins/ocr_rec` | OCR Rec 插件骨架 |
| `src/plugins/dataset_interop` | 数据集互操作插件骨架，承载已实现的数据集转换能力 |
| `tests` | QtTest 测试 |
| `.vscode` | VSCode 构建、调试、任务配置 |
| `tools` | Harness 脚本 |
| `docs/harness` | AI 协作和工程护栏文档 |
| `docs/user-guide.md` | 终端用户 GUI 操作手册 |
| `docs/dataset-conversion.md` | 数据集格式转换矩阵、流程和边界 |
| `docs/delivery-evidence-index.md` | RC、RTX、package、OCR、诊断和外部验收证据索引 |
| `docs/operations-runbook.md` | 打包、安装、现场运维和验收命令 runbook |
| `docs/developer-architecture.md` | 架构边界、扩展入口和验证要求 |

## 当前源码组织说明

- `src/app/src/MainWindow.cpp` 已完成第一层 companion 拆分；Qt Widgets shell 仍保持 left sidebar、top status bar、central `QStackedWidget` 架构。
- `src/core/src/ProductWorkflow.cpp` 已完成第一层 companion 拆分，现在只作为 `ProductWorkflow.h` 的入口锚点；snapshot、quality、evaluation、benchmark、delivery、acceptance、pipeline 的实现分别位于同目录 `ProductWorkflow*.cpp` 文件，跨文件共享 helper 位于内部 `ProductWorkflowSupport.h/.cpp`。
- 这些拆分是行为保持型维护重构，不代表新的 Worker protocol、SQLite schema、插件接口、训练/推理/评估算法或报告字段变更。

## 构建环境

当前机器验证过的组合：

- MSVC 19.50
- Qt 5.12.9 `C:\Qt\Qt5.12.9\5.12.9\msvc2015_64`
- CMake NMake Makefiles
- 构建目录：`build-vscode`

标准检查命令：

```powershell
.\tools\harness-check.ps1
```

## 关键约束

- 保持 Qt 5.12+ 兼容，除非明确升级到 Qt 6。
- Windows 源码编译必须使用 `/utf-8`，避免中文乱码。
- UI 改动必须保留现有功能入口。
- GUI 可见文本优先保持中文源文案，通过 Qt 翻译资源和 `LanguageSupport` 派生英文界面；core/Worker/Python trainer 日志不要求在第一版全部翻译。
- 注册码系统使用离线签名 token；主应用只内置公钥，私钥文件必须本地保管，不进入客户包和源码提交。
- Worker 协议改动必须同步测试。
- 插件接口改动必须考虑已有三个内置插件。
- scaffold 能力必须明确标注，不要描述为真实 YOLO/OCR 训练。
