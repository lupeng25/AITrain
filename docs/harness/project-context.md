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

已完成：

- `AITrainStudio.exe` Qt GUI。
- `aitrain_worker.exe` 独立任务进程。
- JSON Lines over `QLocalSocket` 通信。
- SQLite 项目、任务、指标存储。
- 插件接口和三个内置插件骨架。
- 数据集校验初版。
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

未完成：

- Worker-managed Python Trainer Adapter。
- 官方 Python YOLO 检测真实训练。
- 官方 Python YOLO 分割真实训练。
- 真实 YOLO 分割 ONNX 后处理和 mask overlay。
- 官方 PaddleOCR Rec 真实训练。
- 完整 YOLO/OCR ONNX 导出和 TensorRT 真实导出。
- 完整 YOLO/OCR ONNX Runtime/TensorRT 后处理。
- 环境自检真实实现。
- Windows 打包。

## 源码地图

| 路径 | 职责 |
|---|---|
| `src/core` | 协议、插件接口、任务模型、SQLite 仓库、dataset/training scaffold |
| `src/app` | Qt Widgets GUI |
| `src/app/translations` | GUI 翻译源文件，构建时生成 `.qm` 并随应用安装 |
| `src/license_generator` | 内部 Qt 注册码生成器，用私钥签发绑定机器码的离线注册码 |
| `src/worker` | 长任务隔离进程 |
| `src/plugins/yolo_native` | YOLO 风格插件骨架 |
| `src/plugins/ocr_rec` | OCR Rec 插件骨架 |
| `src/plugins/dataset_interop` | 数据集互操作插件骨架 |
| `tests` | QtTest 测试 |
| `.vscode` | VSCode 构建、调试、任务配置 |
| `tools` | Harness 脚本 |
| `docs/harness` | AI 协作和工程护栏文档 |

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
