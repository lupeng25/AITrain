# AITrain Studio 后续实施方案

生成日期：2026-04-29  
项目目录：`c:\Users\73200\Desktop\code\AITrain`  
目标：明确当前 C++/Qt 视觉模型训练平台的后续开发路线、阶段交付、验收标准和风险控制。

## 1. 当前状态

当前仓库已经完成平台骨架和 VSCode 开发配置，具备继续迭代的基础。

已完成内容：

- Qt/CMake 桌面主程序 `AITrainStudio`。
- 独立 Worker 进程 `aitrain_worker`。
- GUI 与 Worker 之间通过 JSON Lines over `QLocalSocket` 通信。
- SQLite 项目、任务、指标元数据存储。
- 核心插件接口：`IModelPlugin`、`ITrainer`、`IValidator`、`IExporter`、`IInferencer`、`IDatasetAdapter`。
- 三个内置插件骨架：
  - `YoloNativePlugin`
  - `OcrRecNativePlugin`
  - `DatasetInteropPlugin`
- 数据集格式校验的初版实现。
- 训练任务事件链路、日志输出、指标曲线、checkpoint 占位产物。
- YOLO 分割 scaffold：数据读取、SegmentationDataLoader、polygon-to-mask、letterbox 对齐 mask、多 polygon / 多 class mask、mask overlay 预览、mask preview 产物、Worker 端 maskLoss/maskCoverage/maskIoU/segmentationMap50 指标和 scaffold checkpoint。
- OCR recognition scaffold：PaddleOCR Rec 数据读取、字符字典加载、label encode/decode、resize/pad batching、Worker 端 ctcLoss/accuracy/editDistance 指标、scaffold checkpoint 和 preview artifact。
- VSCode 编译、运行、调试和测试配置。
- 基础 QtTest 测试通过。

当前未完成内容：

- 真实 YOLO 检测训练网络、loss、dataloader。
- 真实 YOLO 分割训练网络、真实 mask head、真实 mask loss。
- 真实 OCR CRNN/CTC 训练网络。
- 完整 YOLO/OCR ONNX 导出和 TensorRT 真实导出。
- 完整 YOLO/OCR ONNX Runtime/TensorRT 推理后处理。
- GPU/CUDA/TensorRT/LibTorch 环境自检。
- Windows 安装包和部署流程。

当前 Worker 已能执行 tiny detector 占位训练、导出 tiny detector ONNX，并通过 ONNX Runtime 做推理验证；也已具备 YOLO 分割训练 scaffold，可以读取 segmentation 数据、生成 letterbox 对齐 mask、输出 mask preview，并上报 maskLoss、maskCoverage、maskIoU 和 segmentationMap50；OCR recognition scaffold 可以读取 PaddleOCR Rec 数据、生成 resize/pad batch、输出 checkpoint 和 preview，并上报 ctcLoss、accuracy 和 editDistance。训练核心仍不是完整 YOLO/OCR。后续需要逐步替换为真实 C++/CUDA/LibTorch 训练实现，并扩展完整 YOLO/OCR 的 ONNX/TensorRT 后处理。

## 2. 总体实施原则

- 先稳定平台，再实现真实训练核心。
- 先完成检测训练闭环，再扩展分割和 OCR。
- 数据集校验必须先于训练。
- 所有长任务必须通过 Worker 隔离执行，GUI 不直接承载训练计算。
- 插件接口保持稳定，模型实现可以分阶段替换。
- 不在第一阶段同时推进 YOLO 检测、分割、OCR、TensorRT，避免问题定位困难。

## 3. 阶段路线图

| 阶段 | 目标 | 关键交付 | 验收标准 |
|---|---|---|---|
| 阶段 1 | 平台稳定化 | 任务状态机、Worker 协议、任务队列、环境自检、SQLite 表扩展 | Worker 崩溃 GUI 不崩溃；任务状态准确落库；重启后可恢复显示 |
| 阶段 2 | 数据集系统 | YOLO 检测/分割、PaddleOCR Rec 校验、划分、预览 | 非法数据集阻止训练；错误定位到文件和行号；GUI 可预览样本 |
| 阶段 3 | YOLO 检测训练 | DetectionDataset、DataLoader、模型、loss、mAP50、checkpoint | 小数据集可训练 1 epoch；loss 下降；生成 checkpoint |
| 阶段 4 | ONNX 导出与推理 | ONNX 导出、ONNX Runtime 推理、NMS、结果可视化 | 同图推理输出基本一致；GUI 显示检测框和耗时 |
| 阶段 5 | YOLO 分割训练 | SegmentationDataset、mask head、mask loss、mask 可视化 | 可读取 YOLO segmentation；训练输出 mask；GUI 叠加显示 |
| 阶段 6 | OCR 字符识别训练 | CRNN+CTC、字典、CTC loss、decode、accuracy | PaddleOCR Rec 数据可训练；accuracy 上升；可导出 ONNX |
| 阶段 7 | TensorRT 与部署 | TensorRT engine、FP16、动态 shape、Windows 打包 | 干净机器可启动；插件可加载；GPU 环境自检通过 |

## 4. 阶段 1：平台稳定化

目标：将现有骨架升级为稳定的任务平台，后续模型训练只需接入插件与 Worker。

### 4.1 任务状态机

需要实现严格状态迁移：

- `queued -> running`
- `running -> paused`
- `paused -> running`
- `running -> completed`
- `running -> failed`
- `running -> canceled`
- `queued -> canceled`

禁止无效迁移，例如：

- `completed -> running`
- `failed -> running`
- `canceled -> running`

需要落库字段：

- `task_id`
- `kind`
- `state`
- `plugin_id`
- `task_type`
- `work_dir`
- `message`
- `created_at`
- `updated_at`
- `started_at`
- `finished_at`

### 4.2 Worker 协议扩展

当前已有：

- `startTrain`
- `progress`
- `metric`
- `log`
- `artifact`
- `completed`
- `failed`
- `cancel`

需要新增：

- `pause`
- `resume`
- `heartbeat`
- `environmentCheck`
- `validateDataset`
- `exportModel`
- `infer`

建议统一消息格式：

```json
{
  "type": "metric",
  "requestId": "optional-request-id",
  "payload": {
    "taskId": "task-id",
    "name": "loss",
    "value": 0.42,
    "step": 10,
    "epoch": 1
  }
}
```

### 4.3 任务队列

第一版建议本地单机只允许同时运行一个训练任务。

需要支持：

- 添加任务到队列。
- 取消排队任务。
- 运行任务取消。
- Worker 异常退出后标记任务 `failed`。
- GUI 重启后显示历史任务。
- 当前运行任务可恢复日志查看。

### 4.4 SQLite 表扩展

建议新增表：

- `datasets`
- `dataset_versions`
- `artifacts`
- `exports`
- `plugin_configs`
- `environment_checks`

`artifacts` 用于记录：

- checkpoint
- ONNX
- TensorRT engine
- validation report
- inference preview
- logs

### 4.5 环境自检

GUI 增加“环境”页面，检查：

- NVIDIA Driver 是否存在。
- GPU 型号。
- GPU 显存。
- CUDA runtime 是否存在。
- cuDNN 是否存在。
- TensorRT DLL 是否存在。
- ONNX Runtime DLL 是否存在。
- LibTorch DLL 是否存在。
- 插件 DLL 是否加载成功。

验收标准：

- 无 GPU 时给出明确提示。
- 缺 DLL 时显示具体缺失项。
- 插件加载失败时显示错误原因。

## 5. 阶段 2：数据集系统

目标：让所有训练入口都经过标准化校验，保证训练失败时可以快速定位数据问题。

### 5.1 YOLO 检测数据集

支持结构：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

校验内容：

- `data.yaml` 是否存在。
- 类别数量是否正确。
- 图片是否存在。
- label 是否存在。
- class id 是否越界。
- bbox 是否为 5 列。
- bbox 坐标是否在 `[0,1]`。
- 空标注图片是否按配置允许。
- train/val 是否为空。

### 5.2 YOLO 分割数据集

支持 YOLO segmentation txt：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

校验内容：

- polygon 点数是否合法。
- 坐标是否归一化。
- mask 点数是否为偶数。
- polygon 面积是否过小。
- class id 是否越界。

### 5.3 PaddleOCR Rec 数据集

支持格式：

```text
image_path\tlabel_text
```

支持字符字典：

```text
dict.txt
```

校验内容：

- label 文件是否存在。
- 图片是否存在。
- 标签是否为空。
- 字符是否在字典中。
- 文本长度是否超过 `maxTextLength`。
- 是否存在重复样本。

### 5.4 数据集划分

GUI 支持：

- 按比例划分 train/val/test。
- 固定随机种子。
- 保留原始数据。
- 生成标准化数据目录。
- 保存划分配置。

### 5.5 数据集预览

需要支持：

- 检测框预览。
- 分割 polygon 预览。
- OCR 图片和文本预览。
- 错误样本列表跳转。
- 随机抽样检查。

阶段验收：

- 不合法数据集必须阻止训练。
- 错误信息能定位到文件和行号。
- GUI 能显示样本预览。

## 6. 阶段 3：YOLO 检测训练

目标：先做 detection，不先做 segmentation。检测训练链路更短，适合作为第一个真实训练闭环。

### 6.1 技术路线

- 使用 LibTorch C++ CUDA。
- 使用 OpenCV 读取图片和做基础增强。
- 模型权重保存为自定义 checkpoint 或 TorchScript。
- 导出 ONNX。

第一版不建议直接完整复刻 YOLOv8/YOLO11。建议先实现一个 YOLO-style 小模型：

- Backbone：轻量 CNN/CSP block。
- Neck：FPN/PAN 简化版。
- Head：anchor-free detection head。
- Loss：
  - bbox loss
  - objectness/classification loss
  - 简化 IoU loss

### 6.2 需要实现的模块

- `DetectionDataset`
- `DetectionDataLoader`
- `letterbox`
- `resize`
- `horizontal flip`
- `color jitter`
- 模型定义
- forward
- loss 计算
- optimizer
- checkpoint 保存
- checkpoint 恢复
- mAP50 初版评估
- 训练指标上报

### 6.3 GUI 展示

训练中实时展示：

- train loss
- val loss
- precision
- recall
- mAP50
- learning rate
- epoch
- step
- GPU 显存
- 最新 checkpoint
- best checkpoint

阶段验收：

- 小型检测数据集能训练 1 epoch。
- loss 能下降。
- 能生成 checkpoint。
- 能在 GUI 中显示预测框。
- 能导出 ONNX。

## 7. 阶段 4：ONNX 导出与推理

目标：让训练产物可以用于部署验证。

当前状态：tiny detector 已完成 ONNX 导出、ONNX Runtime 推理、checkpoint/ONNX 一致性测试、Worker 端到端推理、预测 JSON 和 overlay 输出。完整 YOLO/OCR ONNX 后处理和 TensorRT 仍属于后续阶段能力。

### 7.1 ONNX 导出

每个插件实现 `IExporter`：

- 输入 checkpoint。
- 输出 `.onnx`。
- 保存 export config。
- 校验输出 tensor shape。
- 记录导出产物到 SQLite。

### 7.2 ONNX Runtime 推理

实现：

- 图片预处理。
- 模型推理。
- 检测框解码。
- NMS。
- 置信度过滤。
- 类别映射。
- GUI 可视化。

### 7.3 一致性检查

同一张图片对比：

- checkpoint 推理输出。
- ONNX Runtime 推理输出。

验收标准：

- 输出 shape 一致。
- 检测框数量和置信度在合理误差内。
- GUI 能显示推理耗时和结果。

## 8. 阶段 5：YOLO 分割训练

目标：在检测训练稳定后扩展 segmentation。

需要新增：

- `SegmentationDataset`
- polygon 到 mask 的转换。
- mask head。
- mask loss。
- mask 可视化。
- segmentation mAP 初版评估。

建议复用检测模型的大部分结构，只增加 segmentation branch。

阶段验收：

- 能读取 YOLO segmentation 数据。
- 能训练并输出 mask。
- GUI 能叠加显示分割结果。

## 9. 阶段 6：OCR 字符识别训练

目标：实现 PaddleOCR rec 格式兼容的 C++ OCR 训练链路。

第一版建议实现 CRNN + CTC，不建议一开始做复杂 SVTR。

### 9.1 模型结构

- CNN backbone。
- sequence feature reshape。
- BiLSTM 或轻量 Transformer encoder。
- Linear classifier。
- CTC loss。

### 9.2 需要实现

- `OcrRecDataset`
- 图片 resize/pad。
- 字符字典加载。
- label encode/decode。
- CTC loss。
- greedy decode。
- accuracy。
- edit distance。
- checkpoint 保存。
- ONNX 导出。
- OCR 推理预览。

阶段验收：

- PaddleOCR rec 格式数据能训练。
- accuracy 能在小数据集上上升。
- GUI 能显示图片、真实文本、预测文本。
- 能导出 ONNX。

## 10. 阶段 7：TensorRT 与 Windows 打包

目标：做成普通用户可运行的软件包。

### 10.1 TensorRT

支持：

- FP32。
- FP16。
- workspace size。
- dynamic shape profile。
- engine cache。
- engine 加载与推理。

### 10.2 Windows 打包

安装目录结构建议：

```text
AITrainStudio/
  AITrainStudio.exe
  aitrain_worker.exe
  plugins/
    models/
  runtimes/
    onnxruntime/
    tensorrt/
  examples/
  docs/
```

需要做：

- 使用 `windeployqt` 部署 Qt 运行库。
- 拷贝插件目录。
- 拷贝 Worker。
- 拷贝 ONNX Runtime DLL。
- 检查 CUDA/TensorRT DLL。
- 增加 `cmake --install` 规则。

阶段验收：

- 在干净 Windows 机器上能启动。
- 插件能加载。
- 无 GPU 时能提示原因。
- 有 GPU 时能通过环境自检。

## 11. 风险与控制

| 风险 | 影响 | 控制方式 |
|---|---|---|
| 纯 C++ 复刻 YOLO/OCR 训练成本高 | 周期长，效果与官方 Python 生态可能不一致 | 先做 YOLO-style 闭环，再逐步提高模型复杂度 |
| 数据集质量不可控 | 训练失败或指标异常 | 训练前强制校验，错误定位到文件和行号 |
| CUDA/TensorRT 环境差异大 | 用户机器启动失败或导出失败 | 增加环境自检和明确缺失 DLL 提示 |
| GUI 与训练进程耦合 | 训练崩溃导致 GUI 崩溃 | 保持 Worker 隔离和心跳机制 |
| 插件接口频繁变化 | 后续插件维护成本高 | 阶段 1 固化协议和 manifest schema |
| 直接追求完整 YOLOv8/YOLO11 复刻 | 研发周期失控 | 第一版实现 YOLO-style 小模型，后续逐步兼容 |

## 12. 推荐优先级

| 优先级 | 事项 | 原因 |
|---|---|---|
| P0 | 平台稳定化 | 所有后续训练能力都依赖任务、插件、协议和存储稳定 |
| P0 | 数据集系统 | 数据入口不可靠会直接破坏训练结果 |
| P1 | YOLO 检测训练 | 检测闭环最短，适合作为第一个真实训练目标 |
| P1 | ONNX 导出和推理 | 训练产物必须能被部署验证 |
| P2 | YOLO 分割训练 | 在检测基础上扩展，复用较多 |
| P2 | OCR 识别训练 | 单独网络链路，放在检测闭环稳定之后 |
| P3 | TensorRT 和安装包 | 最后做产品化部署 |

## 13. 当前阶段状态与下一步建议

阶段状态的权威入口是 `docs/harness/current-status.md`。新 AI 对话必须先读取该文件，再参考本路线图。

当前状态：

- 阶段 1 平台稳定化：平台 scaffold 已完成。
- 阶段 2 数据集系统：初版已完成，YOLO 检测、YOLO 分割、PaddleOCR Rec 校验可用；数据集划分目前主要覆盖 YOLO 检测。
- 阶段 3 YOLO 检测训练：tiny linear detector scaffold 已完成，可训练小数据、输出指标、checkpoint、preview；不是真实 LibTorch/CUDA YOLO。
- 阶段 4 ONNX 导出与推理：tiny detector ONNX/ONNX Runtime/Worker 推理链路已完成；完整 YOLO/OCR 后处理和 TensorRT 未完成。
- 阶段 5 YOLO 分割训练：scaffold/baseline 闭环已完成，已有 `SegmentationDataset`、`SegmentationDataLoader`、polygon-to-mask、letterbox 对齐 mask、多 polygon / 多 class mask、overlay preview、mask preview、Worker 端 mask 指标、scaffold `segmentationMap50` 和 checkpoint；真实 mask head、真实 mask loss、CUDA 训练仍未完成。
- 阶段 6 OCR 字符识别训练：scaffold/baseline 闭环已完成，已有 `OcrRecDataset`、字典加载、label encode/decode、resize/pad batching、Worker 端 OCR 指标、scaffold checkpoint 和 preview；真实 CRNN/CTC loss、真实 OCR 训练和 OCR ONNX 导出仍未完成。

下一步建议进入阶段 7，先做 TensorRT 与打包 admission scaffold，具体任务：

1. 梳理 Windows 安装目录结构和 CMake install 规则。
2. 明确 TensorRT 当前 unsupported/export placeholder 行为。
3. 扩展环境自检的 runtime DLL 路径提示。
4. 增加打包路径和 TensorRT unsupported 行为测试。
5. 每完成一小步运行 `.\tools\harness-check.ps1`。
