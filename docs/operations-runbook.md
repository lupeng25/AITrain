# AITrain Studio 运维交付 Runbook

最后更新：2026-05-15

本文面向打包、安装、现场运维和交付验收人员，说明 AITrain Studio 的包体边界、依赖准备、证据采集和常见处置。开发架构见 `docs/developer-architecture.md`，用户操作见 `docs/user-guide.md`。

## 包体边界

标准交付包应包含：

- `AITrainStudio.exe`
- `aitrain_worker.exe`
- 内置 Qt plugin DLL
- `plugins\models` 下的内置插件
- `translations` 下的 `.qm` 翻译文件
- 必要的 Qt runtime DLL
- `docs`、`examples`、`requirements` 和验收脚本

不得默认包含：

- 授权私钥或注册码生成器私钥。
- `.deps` 下的本地缓存、下载包、数据集、模型权重、ONNX、TensorRT engine、生成 ZIP。
- 未完成许可证审查的第三方训练框架、标注工具或二进制依赖。
- 客户数据、客户报告或包含客户信息的诊断包。

## 首次启动

1. 启动 `AITrainStudio.exe`。
2. 如果出现注册窗口，复制机器码给授权方。
3. 输入以 `AITRAIN1` 开头的离线注册码。
4. 验证通过后进入主界面。
5. 在“环境”页运行环境自检。

注册信息绑定当前机器。换机后需要重新签发注册码。主程序只应内置公钥；私钥只能保存在授权方内部环境。

## 运行环境

基础要求：

- Windows x64。
- NVIDIA GPU 工作站用于 GPU 训练、TensorRT 和 CUDA 相关验收；无兼容 GPU 时 TensorRT 应显示 `hardware-blocked`。
- 与包体匹配的 Qt runtime。
- 用于官方 YOLO / OCR 后端的独立 Python 环境。

常见外部依赖：

- Ultralytics / Torch / ONNX / ONNX Runtime：用于官方 YOLO 训练、导出和 smoke。
- PaddlePaddle / PaddleOCR / PaddleOCR 源码 checkout：用于官方 OCR 工具链。
- CUDA / cuDNN / TensorRT：用于 TensorRT engine build 和推理验收。
- NCNN 工具：用于 NCNN `.param/.bin` 导出；配置 NCNN SDK/runtime 后，部署验证可执行 YOLO 检测/分割 runtime 推理。
- X-AnyLabeling：作为外部标注工具。

常见配置：

- `AITRAIN_XANYLABELING_EXE`：指定 X-AnyLabeling 可执行文件。
- `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT`：指定 NCNN 转换工具；`AITRAIN_NCNN_ROOT` 同时用于启用 C++ NCNN runtime。
- Python、CUDA、TensorRT、PaddleOCR 等路径优先通过“环境”页自检和修复建议确认。

## 验收命令

上下文检查：

```powershell
.\tools\harness-context.ps1
```

源码级完整检查：

```powershell
.\tools\harness-check.ps1
```

本地 RC closeout：

```powershell
.\tools\local-rc-closeout.ps1
```

本地 package smoke：

```powershell
.\tools\package-smoke.ps1 -SkipBuild
.\tools\acceptance-smoke.ps1 -Package -SkipBuild
```

RTX / SM 75+ TensorRT 验收：

```powershell
.\tools\acceptance-smoke.ps1 -TensorRT
```

客户域 OCR 验收：

```powershell
.\tools\customer-ocr-validation.ps1
```

生成 release handoff：

```powershell
.\tools\release-freeze-handoff.ps1
```

## 外部验收收集

clean Windows package acceptance：

1. 在 clean Windows 机器上解压交付包。
2. 从 package root 运行 `.\tools\acceptance-smoke.ps1 -Package`。
3. 收集 `acceptance_summary.json`、完整控制台输出、包体布局证据。
4. 填写 `docs\acceptance-templates\clean-windows-acceptance-result.md`。
5. 只有 summary 为 `passed` 时才可标记通过。

package-root TensorRT rerun：

1. 在 RTX / SM 75+ 机器上从 package root 执行 TensorRT smoke。
2. 记录 GPU、驱动、CUDA runtime、TensorRT runtime、命令和输出。
3. 填写 `docs\acceptance-templates\tensorrt-acceptance-result.md`。
4. GTX 1060 / SM 61 等旧 GPU 返回 `hardware-blocked` 是正确结果，不应覆盖 RTX 4090 D 已通过证据。

客户域 OCR：

1. 使用客户/目标域 Det、Rec、System 数据。
2. 记录官方 PaddleOCR 报告和 AITrain 汇总报告。
3. Public Total-Text、generated smoke 和 `.deps` 示例只能证明流程，不证明客户域生产精度。

## 现场处置

| 现象 | 优先检查 |
|---|---|
| 只能看到注册窗口 | 机器码和注册码是否匹配；公钥是否正确编译进主程序。 |
| YOLO 后端启动失败 | Python 环境、Ultralytics、Torch、ONNX、ONNX Runtime。 |
| OCR 后端启动失败 | PaddlePaddle、PaddleOCR、源码 checkout、Python 环境隔离。 |
| TensorRT 为 `hardware-blocked` | GPU compute capability、驱动、CUDA、TensorRT runtime。旧 GPU 不应强行通过。 |
| NCNN 导出失败 | `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT` 是否指向有效工具。 |
| 插件禁用失败 | Windows 是否锁定 Qt plugin DLL；关闭相关任务或重启后重新扫描。 |
| 数据集转换后无法训练 | 是否手动选择转换输出目录并重新运行数据集校验。 |

## 证据保全

- 诊断包、验收输出和客户报告默认视为敏感材料，不提交源码控制。
- 每次外部验收都要保留命令、时间、机器、环境、输出摘要和模板。
- 同一验收 lane 重新运行时使用新目录，不覆盖旧证据。
- 对外结论必须能追溯到 `docs/delivery-evidence-index.md` 中的证据路径。
