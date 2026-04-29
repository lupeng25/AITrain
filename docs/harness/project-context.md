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

## 当前事实

已完成：

- `AITrainStudio.exe` Qt GUI。
- `aitrain_worker.exe` 独立任务进程。
- JSON Lines over `QLocalSocket` 通信。
- SQLite 项目、任务、指标存储。
- 插件接口和三个内置插件骨架。
- 数据集校验初版。
- VSCode 构建、运行、调试配置。
- 第一轮 UI 重构。

未完成：

- 真实 LibTorch/CUDA YOLO 检测训练。
- 真实 YOLO 分割训练。
- 真实 OCR CTC 训练。
- 完整 YOLO/OCR ONNX 导出和 TensorRT 真实导出。
- 环境自检真实实现。
- Windows 打包。

## 源码地图

| 路径 | 职责 |
|---|---|
| `src/core` | 协议、插件接口、任务模型、SQLite 仓库 |
| `src/app` | Qt Widgets GUI |
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
- Worker 协议改动必须同步测试。
- 插件接口改动必须考虑已有三个内置插件。
