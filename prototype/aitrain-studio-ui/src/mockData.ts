import type {
  ArtifactRecord,
  DatasetState,
  EnvironmentCheck,
  MetricPoint,
  ModelRecord,
  ProjectState,
  TaskRecord,
  TrainingRun,
} from "./types";

export const projects: ProjectState[] = [
  {
    id: "bearing-defect",
    name: "轴承缺陷检测",
    task: "目标检测",
    backend: "Ultralytics YOLO Detection",
    root: "D:\\AITrain\\bearing-defect",
    updatedAt: "2026-07-10 10:42",
    datasets: 3,
    experiments: 12,
    models: 5,
  },
  {
    id: "surface-segmentation",
    name: "钢板表面分割",
    task: "语义分割",
    backend: "SMP Semantic Segmentation",
    root: "D:\\AITrain\\surface-segmentation",
    updatedAt: "2026-07-09 16:18",
    datasets: 2,
    experiments: 8,
    models: 3,
  },
  {
    id: "label-ocr",
    name: "铭牌字符识别",
    task: "OCR",
    backend: "PaddleOCR Det / Rec",
    root: "D:\\AITrain\\label-ocr",
    updatedAt: "2026-07-08 09:31",
    datasets: 4,
    experiments: 6,
    models: 2,
  },
];

export const datasets: DatasetState[] = [
  {
    id: "bearing-v3",
    name: "bearing-v3",
    split: "train / val / test",
    sampleCount: 2840,
    annotatedCount: 2816,
    classes: [
      { name: "outer-race scratch", count: 912, color: "#2563eb" },
      { name: "inner-race pit", count: 684, color: "#16a34a" },
      { name: "ball corrosion", count: 541, color: "#d97706" },
      { name: "normal", count: 703, color: "#64748b" },
    ],
    status: "healthy",
    updatedAt: "2026-07-10 09:46",
    path: "D:\\AITrain\\bearing-defect\\datasets\\bearing-v3",
  },
  {
    id: "bearing-v2",
    name: "bearing-v2",
    split: "train / val",
    sampleCount: 2156,
    annotatedCount: 2091,
    classes: [
      { name: "outer-race scratch", count: 721, color: "#2563eb" },
      { name: "inner-race pit", count: 596, color: "#16a34a" },
      { name: "normal", count: 839, color: "#64748b" },
    ],
    status: "warning",
    updatedAt: "2026-06-26 14:20",
    path: "D:\\AITrain\\bearing-defect\\datasets\\bearing-v2",
  },
];

export const initialTrainingRun: TrainingRun = {
  id: "run-20260710-0912",
  name: "yolo11n-bearing-v3",
  backend: "Ultralytics YOLO Detection",
  model: "yolo11n.pt",
  status: "running",
  epoch: 18,
  totalEpochs: 50,
  map50: 0.842,
  precision: 0.817,
  recall: 0.791,
  elapsed: "01:24:36",
  eta: "02:31:18",
  startedAt: "2026-07-10 09:12",
};

export const metrics: MetricPoint[] = Array.from({ length: 18 }, (_, index) => {
  const epoch = index + 1;
  return {
    epoch,
    map50: Number((0.42 + epoch * 0.023 + Math.sin(epoch / 2) * 0.024).toFixed(3)),
    precision: Number((0.47 + epoch * 0.019 + Math.sin(epoch / 3) * 0.018).toFixed(3)),
    recall: Number((0.43 + epoch * 0.02 + Math.cos(epoch / 2.5) * 0.017).toFixed(3)),
    loss: Number((1.78 * Math.exp(-epoch / 8) + 0.18).toFixed(3)),
  };
});

export const tasks: TaskRecord[] = [
  { id: "TR-1048", name: "yolo11n-bearing-v3", type: "训练", status: "running", progress: 36, startedAt: "2026-07-10 09:12", duration: "01:24:36" },
  { id: "VL-1047", name: "bearing-v3 数据校验", type: "校验", status: "completed", progress: 100, startedAt: "2026-07-10 08:46", duration: "00:02:18" },
  { id: "EX-1046", name: "bearing-v2 ONNX 导出", type: "导出", status: "completed", progress: 100, startedAt: "2026-07-09 18:32", duration: "00:01:04" },
  { id: "IV-1045", name: "RTX 4090 推理验证", type: "验证", status: "warning", progress: 100, startedAt: "2026-07-09 17:48", duration: "00:04:51" },
  { id: "TR-1044", name: "yolo11s-ablation", type: "训练", status: "blocked", progress: 67, startedAt: "2026-07-09 13:03", duration: "03:16:42" },
];

export const artifacts: ArtifactRecord[] = [
  { id: "a1", name: "best.pt", kind: "Checkpoint", format: "PyTorch", size: "6.2 MB", createdAt: "2026-07-10 10:35", path: "runs\\run-20260710-0912\\weights\\best.pt" },
  { id: "a2", name: "last.pt", kind: "Checkpoint", format: "PyTorch", size: "6.2 MB", createdAt: "2026-07-10 10:35", path: "runs\\run-20260710-0912\\weights\\last.pt" },
  { id: "a3", name: "metrics.csv", kind: "Metrics", format: "CSV", size: "18 KB", createdAt: "2026-07-10 10:36", path: "runs\\run-20260710-0912\\metrics.csv" },
  { id: "a4", name: "validation-report.html", kind: "Report", format: "HTML", size: "428 KB", createdAt: "2026-07-09 18:43", path: "reports\\validation-report.html" },
];

export const models: ModelRecord[] = [
  { id: "m5", version: "v1.5.0", backend: "Ultralytics YOLO Detection", map50: 0.842, precision: 0.817, recall: 0.791, latency: 8.6, size: "6.2 MB", status: "running", createdAt: "2026-07-10 10:35" },
  { id: "m4", version: "v1.4.0", backend: "Ultralytics YOLO Detection", map50: 0.826, precision: 0.803, recall: 0.784, latency: 8.4, size: "6.2 MB", status: "healthy", createdAt: "2026-07-06 17:18" },
  { id: "m3", version: "v1.3.1", backend: "Ultralytics YOLO Detection", map50: 0.798, precision: 0.776, recall: 0.762, latency: 8.1, size: "6.1 MB", status: "healthy", createdAt: "2026-06-28 11:42" },
  { id: "m2", version: "v1.2.0", backend: "Ultralytics YOLO Detection", map50: 0.771, precision: 0.758, recall: 0.736, latency: 7.9, size: "6.1 MB", status: "warning", createdAt: "2026-06-18 16:02" },
];

export const environmentChecks: EnvironmentCheck[] = [
  { id: "worker", name: "AITrain Worker", version: "1.8.0", status: "healthy", detail: "localhost:49720 · 24 ms" },
  { id: "cuda", name: "CUDA", version: "12.4", status: "healthy", detail: "NVIDIA GeForce RTX 4090 D · 23.4 GB" },
  { id: "onnx", name: "ONNX Runtime", version: "1.19.2", status: "healthy", detail: "CUDAExecutionProvider available" },
  { id: "tensorrt", name: "TensorRT", version: "10.3", status: "warning", detail: "Package-root rerun evidence not attached" },
  { id: "python", name: "Python", version: "3.11.9", status: "healthy", detail: "Official backend environment ready" },
  { id: "storage", name: "Project storage", version: "462 GB free", status: "healthy", detail: "D:\\AITrain" },
];

export const logs = [
  "10:34:02  Epoch 18/50  gpu_mem=4.82G  box_loss=0.512  cls_loss=0.284  dfl_loss=0.873",
  "10:34:18  128/178  24.6 it/s  img_size=640",
  "10:35:11  Validation  precision=0.817  recall=0.791  mAP50=0.842  mAP50-95=0.576",
  "10:35:12  Saved checkpoint: weights/best.pt",
  "10:35:13  Worker heartbeat OK · GPU temperature 68°C · utilization 91%",
  "10:35:25  Starting epoch 19/50",
];

export const capabilityRows = [
  ["Ultralytics YOLO Detection", "Train / Export / Validate", "ONNX · TensorRT · NCNN", "Ready"],
  ["Ultralytics YOLO Segmentation", "Train / Export / Validate", "ONNX · TensorRT · NCNN", "Ready"],
  ["Ultralytics YOLO OBB", "Train / Export / Validate", "ONNX Runtime only", "Ready"],
  ["SMP Semantic Segmentation", "Train / ONNX inference", "ONNX Runtime only", "Ready"],
  ["Anomalib PatchCore / EfficientAD", "Worker-managed Python", "Anomalib artifacts", "Ready"],
  ["PaddleOCR Det / Rec", "Official adapters / reports", "Official acceptance", "Ready"],
];
