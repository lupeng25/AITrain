# RTX4090 验证证据归档 - 2026-06-15

本文档用于在清理 `.deps/rtx4090-validation` 历史重产物前保留可审查证据。`python-ocr-gpu` 环境按用户要求保留，用于后续 OCR GPU 复跑。

## 证据来源

- 原始目录：`.deps/rtx4090-validation`
- 机器可读归档：`docs/validation/rtx4090-validation-evidence-20260615.json`
- 大日志尾部归档：`docs/validation/rtx4090-validation-log-tails-20260615.json`
- 保留环境：`.deps/rtx4090-validation/python-ocr-gpu`

## 归档结论

- RTX 4090 D TensorRT 验证线已有 passed 证据；旧 GTX 1060 / SM 61 等不支持硬件仍应记录为 `hardware-blocked`。
- 2026-06-05 follow-up 记录了 local baseline/package、GUI walkthrough、历史 Phase47 Det ONNX+CTest、修复后 CPU smoke、修复后 Phase45、TensorRT 和 public OCR GPU workflow 证据，并保留范围边界。
- 2026-05-13 closeout 仍是历史通过基线；public Total-Text OCR GPU Rec 历史结果记录为 `accuracy=0.73263886345003948`、`CER=0.13776229079138269`。
- 2026-06-05 public OCR GPU rerun 在当前 `accuracy>0.70` gate 下通过，记录为 `accuracy=0.71874997504340365`、`CER=0.13828554259854942`。
- Public Total-Text 只能证明公开数据工作流，不是客户域 OCR 生产精度证明。
- `python-ocr-gpu` 不删除；其他历史输出、旧虚拟环境、下载缓存和重型产物可清理。

## 清理策略

归档完成后，删除 `.deps/rtx4090-validation` 下除 `python-ocr-gpu` 以外的历史证据目录、旧环境、下载缓存、数据集和训练产物。项目文档中的历史证据路径改为引用本归档文件；需要复跑 OCR GPU 时继续使用 `.deps/rtx4090-validation/python-ocr-gpu`。

## 本次清理结果

- `.deps` 总大小：`37.63 GB` -> `8.83 GB`
- `.deps/rtx4090-validation`：`32.54 GB` -> `3.74 GB`
- 释放空间：`28.80 GB`
- 删除目标：33 个
- 保留目标：`.deps/rtx4090-validation/python-ocr-gpu`，大小 `3.74 GB`
- 清理报告：`.deps/Cleanup-Reports/cleanup-rtx4090-validation-20260615.json`
