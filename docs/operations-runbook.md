# AITrain Studio 运维交付 Runbook

最后更新：2026-06-21

本文面向打包、安装、现场运维和交付验收人员，说明 AITrain Studio 的包体边界、依赖准备、证据采集和常见处置。开发架构见 `docs/developer-architecture.md`，用户操作见 `docs/user-guide.md`。

## 包体边界

安装包分为三个：

- 产品本体安装包：当前项目编译生成的 `AITrainStudio.exe`、`aitrain_worker.exe`、内置能力注册表，以及产品随附的 `docs`、`examples`、`python_trainers`、`tools`、`translations`。
- Native 依赖项安装包：Qt/VC runtime、Qt runtime module folders、ONNX Runtime、NCNN、TensorRT 和其他运行时 DLL。
- Python AI 环境安装包：官方 YOLO/OBB、SMP、Anomalib 和 OCR 适配器使用的隔离 Python 环境，以及可选的 PaddleOCR 源码 checkout。默认安装到 `python_env`，不写系统 Python，不要求全局 `AITRAIN_PYTHON_EXECUTABLE`。

三个安装包默认安装到同一个 `AITrain Studio` 目录，但写入不同子树，避免互相覆盖。产品本体安装后需要 Native 依赖项安装包和 Python AI 环境安装包，或等价运行时，才能完整启用 GUI、Worker、YOLO/OBB/SMP/Anomalib/OCR 后端和部署验证。

产品本体安装包应包含：

- `AITrainStudio.exe`
- `aitrain_worker.exe`
- 内置 Qt runtime module DLL
- 编译期内置能力注册表
- `translations` 下的 `.qm` 翻译文件
- `docs`、`examples`、`requirements` 和验收脚本

Native 依赖项安装包应包含：

- 必要的 Qt runtime DLL 和 Qt runtime module folders。
- 必要的 MSVC runtime DLL。
- `runtimes\onnxruntime`、`runtimes\ncnn`、`runtimes\tensorrt`。
- 根目录下供 Worker 直接加载的 ONNX Runtime / NCNN 等 runtime DLL。

Python AI 环境安装包应包含：

- `python_env\python.exe` 或 `python_env\Scripts\python.exe`。
- Ultralytics、segmentation_models.pytorch、Anomalib、Torch、Lightning、timm、ONNX、ONNX Runtime、PaddlePaddle、PaddleOCR、NumPy、Pillow、OpenCV、PyYAML 等官方/上游适配器所需 Python 包。
- 可选 `python_env\PaddleOCR\tools\train.py`，用于 PaddleOCR Det/Rec/System 官方工具链。

安装顺序建议：

1. 产品本体安装包。
2. Native 依赖项安装包。
3. Python AI 环境安装包。
4. 启动 `AITrainStudio.exe` 并在“环境”页运行环境自检。

如果安装顺序不同也不应产生文件冲突；最终三个包必须落在同一个安装目录。Python AI 环境安装包不应包含训练数据、客户数据、模型权重、运行输出或验收证据。正式交付前应从可搬移的 staging Python 环境构建，不建议直接使用开发机 venv 作为客户包来源。

不得默认包含：

- 授权私钥、注册码生成器 `.aitrainkey` 受保护私钥文件或任何历史明文 `*aitrain-license-private-key*.json` 文件。
- `.deps` 下的本地缓存、下载包、数据集、模型权重、ONNX、TensorRT engine、生成 ZIP。
- 未完成许可证审查的第三方训练框架、标注工具或二进制依赖。
- 客户数据、客户报告或包含客户信息的诊断包。

## 首次启动

1. 启动 `AITrainStudio.exe`。
2. 如果出现注册窗口，复制机器码给授权方。
3. 输入以 `AITRAIN1` 开头的离线注册码。
4. 验证通过后进入主界面。
5. 在“环境”页运行环境自检。

注册信息绑定当前机器。换机后需要重新签发注册码。主程序只应内置公钥；私钥只能保存在授权方内部环境。生成器使用 Windows DPAPI 当前用户范围和仅当前用户 ACL 保存 `.aitrainkey`，默认不提供明文展示或导出。仓库内不得保存真实私钥；如果旧私钥曾进入源码或对外分发，应视为已泄漏，先 rotate 到新 key pair，再用新 `AITRAIN_LICENSE_PUBLIC_KEY` 构建客户包。是否清理 Git 历史应作为单独安全流程处理，不能用普通代码提交替代。

应用的可信 UTC 文件位于当前用户应用数据目录，由 DPAPI 保护。时钟回拨容差为 5 分钟；明显回拨或可信时间文件损坏会阻止授权通过。现场排障应先校正 Windows 时间并确认应用由原 Windows 用户运行，不得用删除或替换可信时间文件作为常规绕过方法。纯离线授权不能抵御管理员级完整程序和系统状态篡改。

## 运行环境

基础要求：

- Windows x64。
- NVIDIA GPU 工作站用于 GPU 训练、TensorRT 和 CUDA 相关验收；无兼容 GPU 时 TensorRT 应显示 `hardware-blocked`。
- 与包体匹配的 Qt runtime。
- 用于官方 YOLO/OBB、SMP、Anomalib 和 OCR 后端的独立 Python 环境。

常见外部依赖：

- Ultralytics / Torch / ONNX / ONNX Runtime：用于官方 YOLO detection/segmentation/OBB 训练、导出和 smoke。
- segmentation_models.pytorch / Torch / timm / ONNX / ONNX Runtime：用于 SMP 专用语义分割训练、评估、导出和 ONNX Runtime 部署验证。
- Anomalib / Torch / Lightning / timm / OpenCV / Pillow / NumPy：用于 PatchCore / EfficientAD 异常检测训练、评估、推理、heatmap/overlay/mask 和 benchmark；EfficientAD 还需要显式 `imagenetDir` 或等价本地数据。
- PaddlePaddle / PaddleOCR / PaddleOCR 源码 checkout：用于官方 OCR 工具链。
- CUDA / cuDNN / TensorRT：用于 TensorRT engine build 和推理验收。
- NCNN 工具和 SDK/runtime：用于 NCNN `.param/.bin` 导出和部署验证；配置 NCNN SDK/runtime 后，部署验证可执行 YOLO 检测/分割 runtime 推理。本机验证根目录为 `.deps\sdks\ncnn`，交付机器应使用等价 SDK/runtime 路径。
- X-AnyLabeling：作为外部标注工具和可选 CLI 转换工具，本地查找 `AITRAIN_XANYLABELING_EXE`、程序目录、`tools\x-anylabeling`、`.deps\tools\annotation-tools\X-AnyLabeling`、旧 `.deps\annotation-tools\X-AnyLabeling` 和 `PATH`。

常见配置：

- `AITRAIN_XANYLABELING_EXE`：指定 X-AnyLabeling 可执行文件。
- `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT`：指定 NCNN 转换工具；`AITRAIN_NCNN_ROOT` 同时用于启用 C++ NCNN runtime。
- Python、CUDA、TensorRT、PaddleOCR 等路径优先通过“环境”页自检和修复建议确认。

X-AnyLabeling 默认不进入产品本体包。若现场需要预装或随包分发，应先完成第三方许可证、体积和更新策略评审；当前 AITrain 只按本地外部依赖探测和调用。

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

GUI 1280x820 RC walkthrough：

```powershell
.\tools\ui-workbench-walkthrough.ps1
```

If the result summary reports `errorCode=license_required`, the app is blocked at offline registration. Configure a valid license token and `AITRAIN_LICENSE_PUBLIC_KEY`, then rerun the walkthrough before claiming a GUI pass.

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

NCNN 运行时只通过 `runRuntimeDeliveryWorkflowV2`，由 ModelPackageId、Manifest、tensor/blob/decoder 合同和 Artifact Store 解析模型。旧 NCNN smoke 脚本及 Worker 参数已删除；不完整合同统一返回结构化 `Unsupported`/`Blocked`，不得把裸 `.onnx`、`.param` 或样本路径传给 Worker。

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
| YOLO/OBB 后端启动失败 | Python 环境、Ultralytics、Torch、ONNX、ONNX Runtime。若使用 `device=0`，确认该 Python 安装的是 CUDA 版 PyTorch；CPU-only 环境只能用 `device=cpu`。 |
| SMP 后端启动失败 | Python 环境、segmentation_models.pytorch、Torch、timm、ONNX、ONNX Runtime；SMP 只承诺 ONNX Runtime 部署验证，不要求 NCNN/TensorRT。 |
| Anomaly 后端启动失败 | Python 环境、Anomalib、Torch、Lightning、timm、OpenCV、Pillow、NumPy；EfficientAD 缺少 Imagenette/ImageNet 辅助数据时应记录为 blocked。 |
| YOLO12 分割 `.pt` 训练立即失败 | 当前记录的 Ultralytics 8.3.171 环境无法解析 `yolo12n-seg.pt`，按 `blocked_missing_official_weight` 处理；改用 YOLO12 `-seg.yaml` 架构训练，或等待上游官方 `yolo12*-seg.pt` 权重可解析。 |
| YOLO26 共享环境失败，targeted 已过 | 共享 Ultralytics 8.3.171 环境没有可用的 YOLO26 configs/assets，且 nano `.pt` 权重与包代码不兼容；按 `blocked_model_unavailable` / `blocked_ultralytics_incompatible` 处理。隔离 targeted full 已在 2026-06-15 生成 20/20 训练、官方 ONNX、AITrain C++ ONNX 推理和 TensorRT 通过证据；YOLO26 NCNN 历史尝试 20/20 failed，当前产品不提供 YOLO26 NCNN 导出/转换。客户预检只能按 targeted summary 放行 YOLO26 训练/ONNX/TensorRT。 |
| 浏览器进度页 failed 数和历史失败数不一致 | 进度页按当前 `runId` 统计主 failed/planned/passed，并把旧 `row_summary.json` 计入 `historicalRowCount` / `historicalByStatus`。排查时先确认页面顶部 `runId`，不要把历史 failed 当成当前 run 失败。 |
| OCR 后端启动失败 | PaddlePaddle、PaddleOCR、源码 checkout、Python 环境隔离。 |
| TensorRT 为 `hardware-blocked` | GPU compute capability、驱动、CUDA、TensorRT runtime。旧 GPU 不应强行通过。 |
| NCNN 导出失败 | `AITRAIN_NCNN_ONNX2NCNN` 或 `AITRAIN_NCNN_ROOT` 是否指向有效工具。 |
| NCNN 部署验证返回 `sdk_missing` | 重新配置 `AITRAIN_NCNN_ROOT`，确认 `ncnn.dll` 位于 Worker 同目录或 `runtimes\ncnn`。 |
| NCNN 部署验证返回 `sample_missing` | 提供可读取的 `sampleImagePath`，否则只记录 blocked，不声明 runtime passed。 |
| NCNN 部署验证返回 `sidecar_missing` | 为外部 `.param/.bin` 提供 AITrain sidecar，或显式传入 `modelFamily`、`classNames`、`inputBlob`、`outputBlobs`、`decoder` 等配置。 |
| NCNN 部署验证失败并提示 unsupported `Shape` layer | 当前 `.param` 来自不兼容的 ONNX 转换；使用静态/兼容导出的 ONNX，或改用带 sidecar/config 的预转换 NCNN artifact 后运行 `--ncnn-param-smoke`。 |
| 能力注册表为空 | 检查 Worker `--builtin-capabilities` 输出和构建版本是否一致。 |
| 数据集转换后无法训练 | 是否手动选择转换输出目录并重新运行数据集校验。 |

## 证据保全

- 诊断包、验收输出和客户报告默认视为敏感材料，不提交源码控制。
- 每次外部验收都要保留命令、时间、机器、环境、输出摘要和模板。
- 同一验收 lane 重新运行时使用新目录，不覆盖旧证据。
- 对外结论必须能追溯到 `docs/delivery-evidence-index.md` 中的证据路径。
