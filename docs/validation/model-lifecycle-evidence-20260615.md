# 模型生命周期证据归档 - 2026-06-15

本文档用于在删除 `.deps` 重型模型产物前保留可审查证据。这里记录的是工程生命周期闭环状态，不是客户域精度证明，也不是通用 benchmark。

## 证据来源

- 全量生命周期行级证据原始来源：`.deps/full-model-lifecycle`（已清理）
- OCR 5 epoch 重跑证据原始来源：`.deps/ocr-rerun-5epoch/run-20260615-131044`（已清理）
- YOLO26 独立 targeted matrix：`.deps/phase-yolo26-model-matrix`
- 机器可读归档：`docs/validation/model-lifecycle-evidence-20260615.json`
- 大日志尾部归档：`docs/validation/model-lifecycle-log-tails-20260615.json`

## 总结

| 验证线 | 结果 |
|---|---:|
| 全量生命周期 row summary | 76 passed / 120 total rows |
| OCR 5 epoch rerun | 20 passed / 20 total rows |
| YOLO26 targeted 100 epoch matrix | 20 passed / 20 required rows |
| YOLO26 ONNX runtime validation | 20 passed / 20 rows |
| YOLO26 TensorRT deployment validation | 20 passed / 20 rows |
| YOLO26 NCNN validation | 20 failed / 20 rows |

## 产品口径结论

- YOLOv5u、YOLOv8、YOLO11，以及 YOLO12 `.yaml` / detection `.pt` 行在主线 full-lifecycle 中有 passed 行级证据。
- YOLO12 segmentation `.pt` 仍有 5 行 blocked，原因是验证环境无法解析官方 `yolo12*-seg.pt` 权重。
- 共享 full-lifecycle 环境中 YOLO26 20 行被 blocked；随后隔离 YOLO26 targeted matrix 完成 20/20 required rows，并通过训练、ONNX 导出、AITrain C++ ONNX 推理和 TensorRT 部署验证。
- YOLO26 NCNN 20/20 failed，因此产品侧应移除 YOLO26 NCNN 导出/部署选项。YOLO26 当前只按 ONNX 和 TensorRT 记录可运行部署证据。
- OCR Det/Rec/System 5 epoch rerun 完成 20/20 passed。`pp-ocrv4-mobile-det` 在 OCR rerun summary 中来自早期证据恢复；full-lifecycle 中仍有单独的 `ppocrv4-mobile-det` passed 行级证据。
- COCO128、COCO128-seg 和 Total-Text 只能证明公开数据上的工程闭环与产物生成，不可替代客户域 OCR 或客户域视觉生产精度验证。

## 未关闭或范围限定的问题

| 问题 | 状态 | 产品影响 |
|---|---|---|
| YOLO12 segmentation `.pt` 官方权重 | Open blocker | 在官方权重可解析前，YOLO12 segmentation `.pt` 行保持 blocked。 |
| YOLO26 NCNN | 已通过产品范围规避 | 产品不应暴露 YOLO26 NCNN export/deploy 选项。 |
| TensorRT runtime decoding | 范围限定 | 当前证据覆盖 engine export / deployment validation，不声明 GUI TensorRT 单图推理已通过。 |
| 公开数据指标 | 范围限定 | 不用本轮公开数据指标声明客户域生产精度。 |

## 本次归档后的清理策略

归档完成后，可以删除 `.deps` 中的重型生成产物，包括模型 checkpoint、ONNX、TensorRT engine、NCNN 中间文件、大训练日志、数据集压缩包和 cache。长期保留证据为本文档、`model-lifecycle-evidence-20260615.json` 和 `model-lifecycle-log-tails-20260615.json`。本次清理已删除原始 `.deps/full-model-lifecycle` 和 `.deps/ocr-rerun-5epoch` 产物目录。

运行时依赖不属于本次证据清理范围，除非后续明确要求，否则继续保留规范位置：`.deps/repos/PaddleOCR`、`.deps/sdks/onnxruntime`、`.deps/sdks/ncnn`。
