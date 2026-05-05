#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginInterfaces.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QPushButton>
#include <QProcess>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QStandardPaths>
#include <QTableWidgetItem>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

namespace {

QLabel* mutedLabel(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("MutedText"));
    label->setWordWrap(true);
    return label;
}

QLabel* emptyStateLabel(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("EmptyState"));
    label->setWordWrap(true);
    label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    return label;
}

QLabel* inlineStatusLabel(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("InlineStatus"));
    label->setWordWrap(true);
    return label;
}

QPushButton* primaryButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("PrimaryButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

QPushButton* dangerButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("DangerButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

QString uiText(const char* source)
{
    return aitrain_app::translateText("MainWindow", QString::fromUtf8(source));
}

QString taskTypeLabel(const QString& taskType)
{
    if (taskType == QStringLiteral("detection")) {
        return uiText("检测");
    }
    if (taskType == QStringLiteral("segmentation")) {
        return uiText("分割");
    }
    if (taskType == QStringLiteral("ocr_detection")) {
        return uiText("OCR 检测");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return uiText("OCR 识别");
    }
    if (taskType == QStringLiteral("ocr")) {
        return uiText("OCR 端到端");
    }
    return taskType.isEmpty() ? uiText("未选择") : taskType;
}

void addComboItem(QComboBox* combo, const QString& displayText, const QString& value)
{
    if (!combo) {
        return;
    }
    combo->addItem(displayText, value);
}

QString backendLabel(const QString& backend)
{
    if (backend == QStringLiteral("ultralytics_yolo_detect")) {
        return uiText("Ultralytics YOLO 检测（官方）");
    }
    if (backend == QStringLiteral("ultralytics_yolo_segment")) {
        return uiText("Ultralytics YOLO 分割（官方）");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return uiText("PaddleOCR Det（官方/隔离环境）");
    }
    if (backend == QStringLiteral("paddleocr_rec")) {
        return QStringLiteral("PaddlePaddle OCR Rec CTC");
    }
    if (backend == QStringLiteral("paddleocr_rec_official")) {
        return uiText("PaddleOCR PP-OCRv4 Rec（官方/隔离环境）");
    }
    if (backend == QStringLiteral("paddleocr_system_official")) {
        return uiText("PaddleOCR System 推理（官方）");
    }
    if (backend == QStringLiteral("tiny_linear_detector")) {
        return uiText("Tiny detector（高级/占位）");
    }
    if (backend == QStringLiteral("python_mock")) {
        return uiText("Python mock（高级/协议测试）");
    }
    return backend;
}

QJsonObject readJsonObjectFile(const QString& path)
{
    if (path.isEmpty()) {
        return {};
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        return {};
    }
    return document.object();
}

QJsonObject compactEvaluationSummary(const QString& reportPath)
{
    const QJsonObject report = readJsonObjectFile(reportPath);
    if (report.isEmpty()) {
        return {};
    }
    QJsonObject summary;
    summary.insert(QStringLiteral("reportPath"), reportPath);
    summary.insert(QStringLiteral("taskType"), report.value(QStringLiteral("taskType")).toString());
    summary.insert(QStringLiteral("status"), report.value(QStringLiteral("status")).toString(QStringLiteral("completed")));
    summary.insert(QStringLiteral("scaffold"), report.value(QStringLiteral("scaffold")).toBool());
    summary.insert(QStringLiteral("metrics"), report.value(QStringLiteral("metrics")).toObject());
    summary.insert(QStringLiteral("limitations"), report.value(QStringLiteral("limitations")).toArray());
    return summary;
}

QJsonObject compactBenchmarkSummary(const QString& reportPath)
{
    const QJsonObject report = readJsonObjectFile(reportPath);
    if (report.isEmpty()) {
        return {};
    }
    QJsonObject summary;
    summary.insert(QStringLiteral("reportPath"), reportPath);
    summary.insert(QStringLiteral("runtime"), report.value(QStringLiteral("runtime")).toString());
    summary.insert(QStringLiteral("modelFamily"), report.value(QStringLiteral("modelFamily")).toString());
    summary.insert(QStringLiteral("runtimeStatus"), report.value(QStringLiteral("runtimeStatus")).toString(report.value(QStringLiteral("status")).toString()));
    summary.insert(QStringLiteral("deploymentConclusion"), report.value(QStringLiteral("deploymentConclusion")).toString());
    summary.insert(QStringLiteral("timedInference"), report.value(QStringLiteral("timedInference")).toBool());
    summary.insert(QStringLiteral("averageMs"), report.value(QStringLiteral("averageMs")).toDouble());
    summary.insert(QStringLiteral("p95Ms"), report.value(QStringLiteral("p95Ms")).toDouble());
    summary.insert(QStringLiteral("throughput"), report.value(QStringLiteral("throughput")).toDouble());
    summary.insert(QStringLiteral("failureCategory"), report.value(QStringLiteral("failureCategory")).toString());
    return summary;
}

QString metricValueText(const QJsonObject& metrics, const QStringList& keys)
{
    for (const QString& key : keys) {
        if (metrics.contains(key)) {
            return QStringLiteral("%1=%2").arg(key).arg(metrics.value(key).toDouble(), 0, 'f', 4);
        }
    }
    return {};
}

QString modelSummaryText(const QJsonObject& summary)
{
    QStringList parts;
    const QJsonObject evaluation = summary.value(QStringLiteral("evaluation")).toObject();
    const QJsonObject metrics = evaluation.value(QStringLiteral("metrics")).toObject();
    const QString metricText = metricValueText(metrics, {
        QStringLiteral("mAP50"),
        QStringLiteral("maskIoU"),
        QStringLiteral("accuracy"),
        QStringLiteral("cer")
    });
    if (!metricText.isEmpty()) {
        parts.append(metricText);
    }

    const QJsonObject benchmark = summary.value(QStringLiteral("benchmark")).toObject();
    if (!benchmark.isEmpty()) {
        const QString runtime = benchmark.value(QStringLiteral("runtime")).toString();
        const double p95 = benchmark.value(QStringLiteral("p95Ms")).toDouble();
        const double throughput = benchmark.value(QStringLiteral("throughput")).toDouble();
        if (benchmark.value(QStringLiteral("timedInference")).toBool()) {
            parts.append(QStringLiteral("%1 p95=%2 ms").arg(runtime.isEmpty() ? QStringLiteral("runtime") : runtime).arg(p95, 0, 'f', 2));
            parts.append(QStringLiteral("throughput=%1/s").arg(throughput, 0, 'f', 2));
        } else {
            const QString status = benchmark.value(QStringLiteral("runtimeStatus")).toString(QStringLiteral("limited"));
            parts.append(QStringLiteral("%1 %2").arg(runtime.isEmpty() ? QStringLiteral("runtime") : runtime, status));
        }
    }

    const QJsonArray limitations = summary.value(QStringLiteral("limitations")).toArray();
    if (!limitations.isEmpty()) {
        parts.append(uiText("限制 %1 项").arg(limitations.size()));
    }

    if (!parts.isEmpty()) {
        return parts.join(QStringLiteral(" | "));
    }

    QStringList fallback;
    for (auto it = summary.constBegin(); it != summary.constEnd() && fallback.size() < 4; ++it) {
        if (it.value().isDouble()) {
            fallback.append(QStringLiteral("%1=%2").arg(it.key()).arg(it.value().toDouble(), 0, 'f', 4));
        }
    }
    return fallback.isEmpty() ? QString::fromUtf8(QJsonDocument(summary).toJson(QJsonDocument::Compact)) : fallback.join(QStringLiteral(", "));
}

QString exportComboLabel(const QString& format)
{
    if (format == QStringLiteral("onnx")) {
        return uiText("ONNX 模型");
    }
    if (format == QStringLiteral("ncnn")) {
        return uiText("NCNN param/bin（onnx2ncnn）");
    }
    if (format == QStringLiteral("tiny_detector_json")) {
        return uiText("AITrain JSON（诊断）");
    }
    if (format == QStringLiteral("tensorrt")) {
        return uiText("TensorRT Engine（RTX / SM 75+ 外部验收）");
    }
    return format;
}

InfoPanel* createCompactSummaryCard(const QString& label, const QString& value, const QString& caption)
{
    auto* panel = new InfoPanel(label);
    panel->setObjectName(QStringLiteral("CompactMetricPanel"));
    panel->setMinimumWidth(0);
    panel->setMinimumHeight(78);
    panel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    if (auto* panelLayout = qobject_cast<QVBoxLayout*>(panel->layout())) {
        panelLayout->setContentsMargins(12, 10, 12, 10);
        panelLayout->setSpacing(5);
    }
    panel->bodyLayout()->setSpacing(2);
    auto* valueLabel = new QLabel(value);
    valueLabel->setObjectName(QStringLiteral("CompactMetricValue"));
    valueLabel->setWordWrap(true);
    valueLabel->setMinimumWidth(0);
    valueLabel->setMinimumHeight(22);
    valueLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    valueLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* captionLabel = new QLabel(caption);
    captionLabel->setObjectName(QStringLiteral("CompactMetricCaption"));
    captionLabel->setWordWrap(true);
    captionLabel->setMinimumWidth(0);
    captionLabel->setMinimumHeight(16);
    captionLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    panel->bodyLayout()->addWidget(valueLabel);
    panel->bodyLayout()->addWidget(captionLabel);
    panel->bodyLayout()->addStretch();
    return panel;
}

QString taskStateLabel(aitrain::TaskState state)
{
    switch (state) {
    case aitrain::TaskState::Queued: return uiText("排队中");
    case aitrain::TaskState::Running: return uiText("运行中");
    case aitrain::TaskState::Paused: return uiText("已暂停");
    case aitrain::TaskState::Completed: return uiText("已完成");
    case aitrain::TaskState::Failed: return uiText("失败");
    case aitrain::TaskState::Canceled: return uiText("已取消");
    }
    return uiText("未知");
}

QString taskKindLabel(aitrain::TaskKind kind)
{
    switch (kind) {
    case aitrain::TaskKind::Train: return uiText("训练");
    case aitrain::TaskKind::Validate: return uiText("校验");
    case aitrain::TaskKind::Export: return uiText("导出");
    case aitrain::TaskKind::Infer: return uiText("推理");
    case aitrain::TaskKind::Evaluate: return uiText("评估");
    case aitrain::TaskKind::Benchmark: return uiText("基准");
    case aitrain::TaskKind::Curate: return uiText("质检");
    case aitrain::TaskKind::Snapshot: return uiText("快照");
    case aitrain::TaskKind::Pipeline: return uiText("流水线");
    case aitrain::TaskKind::Report: return uiText("报告");
    }
    return uiText("任务");
}

QString environmentStatusLabel(const QString& status)
{
    if (status == QStringLiteral("ok")) {
        return uiText("通过");
    }
    if (status == QStringLiteral("hardware-blocked")) {
        return uiText("硬件受限");
    }
    if (status == QStringLiteral("warning")) {
        return uiText("警告");
    }
    if (status == QStringLiteral("missing")) {
        return uiText("缺失");
    }
    return uiText("未知");
}

QString issueSeverityLabel(const QString& severity)
{
    if (severity == QStringLiteral("error")) {
        return uiText("错误");
    }
    if (severity == QStringLiteral("warning")) {
        return uiText("警告");
    }
    return uiText("信息");
}

QString inferenceTaskTypeLabel(const QString& taskType)
{
    if (taskType == QStringLiteral("segmentation")) {
        return uiText("分割");
    }
    if (taskType == QStringLiteral("ocr_detection")) {
        return uiText("OCR 检测");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return uiText("OCR 识别");
    }
    if (taskType == QStringLiteral("ocr")) {
        return uiText("OCR 端到端");
    }
    return uiText("检测");
}

QString datasetFormatLabel(const QString& format)
{
    if (format == QStringLiteral("yolo_detection") || format == QStringLiteral("yolo_txt")) {
        return uiText("YOLO 检测");
    }
    if (format == QStringLiteral("yolo_segmentation")) {
        return uiText("YOLO 分割");
    }
    if (format == QStringLiteral("paddleocr_det")) {
        return QStringLiteral("PaddleOCR Det");
    }
    if (format == QStringLiteral("paddleocr_rec")) {
        return QStringLiteral("PaddleOCR Rec");
    }
    if (format == QStringLiteral("coco_json")) {
        return QStringLiteral("COCO JSON");
    }
    if (format == QStringLiteral("voc_xml")) {
        return QStringLiteral("VOC XML");
    }
    if (format == QStringLiteral("labelme_json")) {
        return QStringLiteral("LabelMe JSON");
    }
    return format.isEmpty() ? uiText("未选择") : format;
}

QString defaultBackendForTask(const QString& taskType)
{
    if (taskType == QStringLiteral("segmentation")) {
        return QStringLiteral("ultralytics_yolo_segment");
    }
    if (taskType == QStringLiteral("ocr_detection")) {
        return QStringLiteral("paddleocr_det_official");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return QStringLiteral("paddleocr_rec");
    }
    if (taskType == QStringLiteral("ocr")) {
        return QStringLiteral("paddleocr_system_official");
    }
    return QStringLiteral("ultralytics_yolo_detect");
}

QString defaultModelForBackend(const QString& backend)
{
    if (backend == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("yolov8n-seg.yaml");
    }
    if (backend == QStringLiteral("ultralytics_yolo_detect") || backend == QStringLiteral("ultralytics_yolo")) {
        return QStringLiteral("yolov8n.yaml");
    }
    if (backend == QStringLiteral("paddleocr_rec")) {
        return QStringLiteral("paddle_ctc_smoke");
    }
    if (backend == QStringLiteral("paddleocr_rec_official") || backend == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return QStringLiteral("PP-OCRv4_mobile_rec");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return QStringLiteral("PP-OCRv4_mobile_det");
    }
    if (backend == QStringLiteral("paddleocr_system_official")) {
        return QStringLiteral("PP-OCRv4_det_rec_system");
    }
    return QStringLiteral("diagnostic");
}

QString trainingBackendDescription(const QString& backend)
{
    if (backend == QStringLiteral("ultralytics_yolo_detect")) {
        return uiText("当前模型能力：官方 Ultralytics YOLO detection。适合 YOLO bbox 数据，输出 best.pt、ONNX、训练报告，可继续做 ONNX Runtime 推理和 overlay 验证。");
    }
    if (backend == QStringLiteral("ultralytics_yolo_segment")) {
        return uiText("当前模型能力：官方 Ultralytics YOLO segmentation。适合 YOLO polygon 数据，输出 mask 指标、best.pt、ONNX，并可生成 mask prediction JSON 与 overlay。");
    }
    if (backend == QStringLiteral("paddleocr_rec")) {
        return uiText("当前模型能力：PaddlePaddle CTC OCR Rec。适合 rec_gt.txt + dict.txt 小规模识别数据，可导出 ONNX 并走 C++ CTC greedy decode。");
    }
    if (backend == QStringLiteral("paddleocr_rec_official") || backend == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return uiText("当前模型能力：官方 PaddleOCR PP-OCRv4 Rec 适配器。适合隔离 OCR Python 环境，记录 train/export/predict 命令、checkpoint、inference model 和官方预测报告。");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return uiText("当前模型能力：官方 PaddleOCR PP-OCRv4 Det 适配器。适合 PaddleOCR 原生 det_gt.txt 数据，输出官方配置、checkpoint、inference model 和报告。");
    }
    if (backend == QStringLiteral("paddleocr_system_official")) {
        return uiText("当前模型能力：官方 PaddleOCR 端到端推理编排。使用已导出的 Det/Rec inference model 调用 predict_system.py；本阶段不做 C++ DB 后处理。");
    }
    if (backend == QStringLiteral("tiny_linear_detector")) {
        return uiText("高级/诊断：C++ tiny detector 占位训练，仅验证平台链路、checkpoint、ONNX 和回归测试，不代表真实 YOLO 能力。");
    }
    if (backend == QStringLiteral("python_mock")) {
        return uiText("高级/诊断：Python 协议测试后端，只验证 Worker JSON Lines 协议，不产生真实模型。");
    }
    return uiText("当前模型能力：通过 Worker 执行，产物、指标和失败原因会写入任务历史。");
}

QString exportFormatLabel(const QString& format)
{
    if (format == QStringLiteral("onnx")) {
        return uiText("ONNX 模型");
    }
    if (format == QStringLiteral("ncnn")) {
        return QStringLiteral("NCNN param/bin");
    }
    if (format == QStringLiteral("tensorrt")) {
        return QStringLiteral("TensorRT Engine");
    }
    return uiText("AITrain JSON（诊断）");
}

QString defaultExportFileName(const QString& format)
{
    if (format == QStringLiteral("onnx")) {
        return QStringLiteral("model.onnx");
    }
    if (format == QStringLiteral("ncnn")) {
        return QStringLiteral("model.param");
    }
    if (format == QStringLiteral("tensorrt")) {
        return QStringLiteral("model.engine");
    }
    return QStringLiteral("model.aitrain-export.json");
}

QString exportFileFilter(const QString& format)
{
    if (format == QStringLiteral("onnx")) {
        return QStringLiteral("ONNX model (*.onnx);;All files (*.*)");
    }
    if (format == QStringLiteral("ncnn")) {
        return QStringLiteral("NCNN param (*.param);;All files (*.*)");
    }
    if (format == QStringLiteral("tensorrt")) {
        return QStringLiteral("TensorRT engine (*.engine *.plan);;All files (*.*)");
    }
    return QStringLiteral("AITrain JSON (*.json *.aitrain);;All files (*.*)");
}

QString exportFormatNote(const QString& format)
{
    if (format == QStringLiteral("onnx")) {
        return uiText("主交付格式，可继续进入推理验证。");
    }
    if (format == QStringLiteral("ncnn")) {
        return uiText("需要配置 onnx2ncnn，输出 param/bin。");
    }
    if (format == QStringLiteral("tensorrt")) {
        return uiText("需要 RTX / SM 75+ 真机外部验收。");
    }
    return uiText("仅用于 tiny detector 诊断，不代表真实 YOLO/OCR。");
}

QString compactListSummary(const QStringList& values, int maxItems = 3)
{
    QStringList unique;
    for (const QString& value : values) {
        if (!value.trimmed().isEmpty() && !unique.contains(value)) {
            unique.append(value);
        }
    }
    unique.sort(Qt::CaseInsensitive);
    if (unique.isEmpty()) {
        return uiText("暂无");
    }
    const QString visible = unique.mid(0, maxItems).join(QStringLiteral(", "));
    const int remaining = unique.size() - qMin(unique.size(), maxItems);
    return remaining > 0
        ? uiText("%1 等 %2 项").arg(visible).arg(unique.size())
        : visible;
}

int uniqueStringCount(const QStringList& values)
{
    QStringList unique;
    for (const QString& value : values) {
        if (!value.trimmed().isEmpty() && !unique.contains(value)) {
            unique.append(value);
        }
    }
    return unique.size();
}

bool setComboCurrentData(QComboBox* combo, const QString& data)
{
    if (!combo || data.isEmpty()) {
        return false;
    }
    const int index = combo->findData(data);
    if (index < 0) {
        return false;
    }
    combo->setCurrentIndex(index);
    return true;
}

QString confidencePercent(double confidence)
{
    return QStringLiteral("%1%").arg(QString::number(confidence * 100.0, 'f', 1));
}

QStringList appImageNameFilters()
{
    return {
        QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),
        QStringLiteral("*.bmp"),
        QStringLiteral("*.tif"),
        QStringLiteral("*.tiff")
    };
}

QFileInfoList appImageFiles(const QDir& directory)
{
    QFileInfoList files;
    for (const QString& filter : appImageNameFilters()) {
        files.append(directory.entryInfoList({filter}, QDir::Files, QDir::Name));
    }
    return files;
}

QStringList xAnyLabelingCandidates()
{
    const QString envProgram = QString::fromLocal8Bit(qgetenv("AITRAIN_XANYLABELING_EXE")).trimmed();
    const QString appDir = QApplication::applicationDirPath();
    return {
        envProgram,
        QDir(appDir).filePath(QStringLiteral("X-AnyLabeling.exe")),
        QDir(appDir).filePath(QStringLiteral("xanylabeling.exe")),
        QDir(appDir).filePath(QStringLiteral("tools/x-anylabeling/X-AnyLabeling.exe")),
        QDir(QDir::currentPath()).filePath(QStringLiteral(".deps/annotation-tools/X-AnyLabeling/X-AnyLabeling.exe")),
        QStringLiteral("xanylabeling"),
        QStringLiteral("X-AnyLabeling.exe")
    };
}

QString resolveExecutableCandidate(const QString& candidate)
{
    const QString trimmed = candidate.trimmed();
    if (trimmed.isEmpty()) {
        return QString();
    }

    const QFileInfo info(trimmed);
    if (info.isAbsolute() || trimmed.contains(QLatin1Char('/')) || trimmed.contains(QLatin1Char('\\'))) {
        return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
    }

    const QString pathExecutable = QStandardPaths::findExecutable(trimmed);
    if (!pathExecutable.isEmpty()) {
        return pathExecutable;
    }
    return info.exists() && info.isFile() ? info.absoluteFilePath() : QString();
}

QString resolvedXAnyLabelingProgram()
{
    for (const QString& candidate : xAnyLabelingCandidates()) {
        const QString resolved = resolveExecutableCandidate(candidate);
        if (!resolved.isEmpty()) {
            return resolved;
        }
    }
    return QString();
}

QString xAnyLabelingStatusText()
{
    const QString program = resolvedXAnyLabelingProgram();
    if (program.isEmpty()) {
        return uiText("状态：未检测到 X-AnyLabeling。可放到 .deps/annotation-tools/X-AnyLabeling，或设置 AITRAIN_XANYLABELING_EXE。");
    }
    return uiText("状态：已安装 | %1").arg(QDir::toNativeSeparators(program));
}

QString detectDatasetFormatFromPath(const QString& path)
{
    const QDir root(path);
    if (QFileInfo::exists(root.filePath(QStringLiteral("det_gt.txt")))
        || QFileInfo::exists(root.filePath(QStringLiteral("det_gt_train.txt")))) {
        return QStringLiteral("paddleocr_det");
    }
    if (QFileInfo::exists(root.filePath(QStringLiteral("rec_gt.txt")))
        || QFileInfo::exists(root.filePath(QStringLiteral("rec_gt_train.txt")))) {
        if (QFileInfo::exists(root.filePath(QStringLiteral("dict.txt")))) {
            return QStringLiteral("paddleocr_rec");
        }
    }

    if (!QFileInfo::exists(root.filePath(QStringLiteral("data.yaml")))) {
        return QString();
    }
    for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
        const QDir imageDir(root.filePath(QStringLiteral("images/%1").arg(split)));
        const QDir labelDir(root.filePath(QStringLiteral("labels/%1").arg(split)));
        if (!imageDir.exists() || !labelDir.exists()) {
            continue;
        }
        const QFileInfoList images = appImageFiles(imageDir);
        for (const QFileInfo& imageInfo : images) {
            QFile labelFile(labelDir.filePath(imageInfo.completeBaseName() + QStringLiteral(".txt")));
            if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                continue;
            }
            while (!labelFile.atEnd()) {
                const QString line = QString::fromUtf8(labelFile.readLine()).trimmed();
                if (line.isEmpty()) {
                    continue;
                }
                const QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
                    QString::SkipEmptyParts
#else
                    Qt::SkipEmptyParts
#endif
                );
                if (parts.size() == 5) {
                    return QStringLiteral("yolo_detection");
                }
                if (parts.size() >= 7 && parts.size() % 2 == 1) {
                    return QStringLiteral("yolo_segmentation");
                }
            }
        }
    }
    return QStringLiteral("yolo_detection");
}

QString formatJsonTextForPreview(const QByteArray& data)
{
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError) {
        return QString::fromUtf8(data);
    }
    return QString::fromUtf8(document.toJson(QJsonDocument::Indented));
}

void addTaskTypeItems(QComboBox* combo, const QStringList& taskTypes)
{
    if (!combo) {
        return;
    }
    for (const QString& taskType : taskTypes) {
        combo->addItem(taskTypeLabel(taskType), taskType);
    }
}

QString comboCurrentDataOrText(const QComboBox* combo)
{
    if (!combo) {
        return QString();
    }
    const QString data = combo->currentData().toString();
    return data.isEmpty() ? combo->currentText() : data;
}

QString inferenceSummaryFromPredictions(const QString& predictionsPath, const QJsonObject& fallback = {})
{
    const QString nativePath = QDir::toNativeSeparators(predictionsPath);
    QFile file(predictionsPath);
    if (!file.open(QIODevice::ReadOnly)) {
        const QString taskType = fallback.value(QStringLiteral("taskType")).toString(QStringLiteral("detection"));
        return uiText("%1：%2 个结果，%3 ms\n结果文件：%4")
            .arg(inferenceTaskTypeLabel(taskType))
            .arg(fallback.value(QStringLiteral("predictionCount")).toInt())
            .arg(fallback.value(QStringLiteral("elapsedMs")).toInt())
            .arg(nativePath);
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return uiText("预测结果 JSON 无法解析：%1").arg(nativePath);
    }

    const QJsonObject root = document.object();
    const QString taskType = root.value(QStringLiteral("taskType")).toString(
        fallback.value(QStringLiteral("taskType")).toString(QStringLiteral("detection")));
    const QJsonArray predictions = root.value(QStringLiteral("predictions")).toArray();
    const int elapsedMs = root.value(QStringLiteral("elapsedMs")).toInt(fallback.value(QStringLiteral("elapsedMs")).toInt());
    QString detail;
    if (!predictions.isEmpty()) {
        const QJsonObject first = predictions.at(0).toObject();
        if (taskType == QStringLiteral("ocr_recognition")) {
            const QString text = first.value(QStringLiteral("text")).toString();
            detail = text.isEmpty()
                ? uiText("未识别出文本")
                : uiText("文本 \"%1\"").arg(text);
            if (first.contains(QStringLiteral("confidence"))) {
                detail.append(uiText("，置信度 %1").arg(confidencePercent(first.value(QStringLiteral("confidence")).toDouble())));
            }
        } else {
            const QString className = first.value(QStringLiteral("className")).toString(
                QStringLiteral("class %1").arg(first.value(QStringLiteral("classId")).toInt()));
            detail = uiText("首个 %1，置信度 %2")
                .arg(className)
                .arg(confidencePercent(first.value(QStringLiteral("confidence")).toDouble()));
            if (taskType == QStringLiteral("segmentation")) {
                detail.append(QStringLiteral("，mask area %1").arg(confidencePercent(first.value(QStringLiteral("maskArea")).toDouble())));
            }
        }
    } else {
        detail = uiText("无结果");
    }

    return uiText("%1：%2 个结果，%3，%4 ms\n结果文件：%5")
        .arg(inferenceTaskTypeLabel(taskType))
        .arg(predictions.size())
        .arg(detail)
        .arg(elapsedMs)
        .arg(nativePath);
}

} // namespace

MainWindow::MainWindow(const QString& licenseOwner, const QString& licenseExpiry, QWidget* parent)
    : QMainWindow(parent)
    , licenseOwner_(licenseOwner)
    , licenseExpiry_(licenseExpiry)
{
    setWindowTitle(QStringLiteral("AITrain Studio"));
    setMinimumSize(1180, 760);

    auto* central = new QWidget(this);
    auto* rootLayout = new QHBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    sidebar_ = new Sidebar;
    sidebar_->addSection(tr("工作台"));
    sidebar_->addItem(tr("总览"), DashboardPage);
    sidebar_->addItem(tr("项目"), ProjectPage);
    sidebar_->addSection(tr("数据与训练"));
    sidebar_->addItem(tr("数据集"), DatasetPage);
    sidebar_->addItem(tr("训练实验"), TrainingPage);
    sidebar_->addItem(tr("任务与产物"), TaskQueuePage);
    sidebar_->addSection(tr("模型交付"));
    sidebar_->addItem(tr("模型库"), ModelRegistryPage);
    sidebar_->addItem(tr("评估报告"), EvaluationReportsPage);
    sidebar_->addItem(tr("模型导出"), ConversionPage);
    sidebar_->addItem(tr("推理验证"), InferencePage);
    sidebar_->addSection(tr("系统"));
    sidebar_->addItem(tr("插件"), PluginsPage);
    sidebar_->addItem(tr("环境"), EnvironmentPage);
    rootLayout->addWidget(sidebar_);

    auto* content = new QWidget;
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(0);
    contentLayout->addWidget(buildTopBar());

    stack_ = new QStackedWidget;
    stack_->addWidget(buildDashboardPage());
    stack_->addWidget(buildProjectPage());
    stack_->addWidget(buildDatasetPage());
    stack_->addWidget(buildTrainingPage());
    stack_->addWidget(buildTaskQueuePage());
    stack_->addWidget(buildModelRegistryPage());
    stack_->addWidget(buildEvaluationReportsPage());
    stack_->addWidget(buildConversionPage());
    stack_->addWidget(buildInferencePage());
    stack_->addWidget(buildPluginsPage());
    stack_->addWidget(buildEnvironmentPage());
    contentLayout->addWidget(stack_, 1);

    rootLayout->addWidget(content, 1);
    setCentralWidget(central);

    statusBar()->showMessage(tr("就绪"));

    connect(sidebar_, &Sidebar::pageRequested, this, &MainWindow::showPage);
    connect(&worker_, &WorkerClient::messageReceived, this, &MainWindow::handleWorkerMessage);
    connect(&worker_, &WorkerClient::logLine, this, &MainWindow::appendLog);
    connect(&worker_, &WorkerClient::connected, this, [this]() {
        workerPill_->setStatus(tr("Worker 已连接"), StatusPill::Tone::Success);
    });
    connect(&worker_, &WorkerClient::idle, this, &MainWindow::startNextQueuedTask);
    connect(&worker_, &WorkerClient::finished, this, [this](bool ok, const QString& message) {
        progressBar_->setValue(ok ? 100 : progressBar_->value());
        workerPill_->setStatus(ok ? tr("任务完成") : tr("任务失败"),
            ok ? StatusPill::Tone::Success : StatusPill::Tone::Error);
        appendLog(ok ? tr("任务完成：%1").arg(message) : tr("任务失败：%1").arg(message));
        const QString kind;
        const QString path;
        if (!currentTaskId_.isEmpty()) {
            QString error;
            repository_.updateTaskState(currentTaskId_, ok ? aitrain::TaskState::Completed : aitrain::TaskState::Failed, message, &error);
            if (ok) {
                updateExperimentRunSummary(currentTaskId_);
            }
            currentTaskId_.clear();
            updateRecentTasks();
            updateSelectedTaskDetails();
            updateModelRegistry();
        } else if (kind == QStringLiteral("export") && exportResultLabel_) {
            exportResultLabel_->setText(tr("导出完成：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
            QPixmap overlay(path);
            if (!overlay.isNull()) {
                inferenceOverlayLabel_->setPixmap(overlay.scaled(
                    inferenceOverlayLabel_->size().boundedTo(QSize(520, 360)),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
            } else {
                inferenceOverlayLabel_->setText(tr("推理 overlay 加载失败"));
            }
        } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
        }
        startNextQueuedTask();
    });

    refreshPlugins();
    aitrain_app::translateWidgetTree(this);
    showPage(DashboardPage, tr("总览"));
    updateHeaderState();
    updateDashboardSummary();
}

QWidget* MainWindow::buildTopBar()
{
    auto* topBar = new QFrame;
    topBar->setObjectName(QStringLiteral("TopBar"));
    topBar->setFixedHeight(72);

    auto* layout = new QHBoxLayout(topBar);
    layout->setContentsMargins(22, 10, 22, 10);
    layout->setSpacing(16);

    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    pageTitle_ = new QLabel(tr("首页"));
    pageTitle_->setObjectName(QStringLiteral("PageTitle"));
    pageCaption_ = new QLabel;
    pageCaption_->setObjectName(QStringLiteral("PageCaption"));
    titleLayout->addWidget(pageTitle_);
    titleLayout->addWidget(pageCaption_);

    headerProjectLabel_ = new QLabel(tr("项目：未打开"));
    headerProjectLabel_->setObjectName(QStringLiteral("MutedText"));
    workerPill_ = new StatusPill;
    workerPill_->setStatus(tr("Worker 空闲"), StatusPill::Tone::Neutral);
    pluginPill_ = new StatusPill;
    gpuPill_ = new StatusPill;
    gpuPill_->setStatus(tr("GPU 未检测"), StatusPill::Tone::Warning);
    licensePill_ = new StatusPill;
    licensePill_->setStatus(licenseOwner_.isEmpty()
            ? tr("已注册")
            : tr("授权：%1").arg(licenseOwner_),
        StatusPill::Tone::Success);
    licensePill_->setToolTip(licenseExpiry_.isEmpty()
            ? tr("离线授权已验证")
            : tr("授权有效期：%1").arg(licenseExpiry_));
    auto* languageSwitch = new QFrame;
    languageSwitch->setObjectName(QStringLiteral("LanguageSwitch"));
    auto* languageLayout = new QHBoxLayout(languageSwitch);
    languageLayout->setContentsMargins(2, 2, 2, 2);
    languageLayout->setSpacing(0);
    auto* zhButton = new QToolButton;
    zhButton->setObjectName(QStringLiteral("LanguageSwitchButton"));
    zhButton->setText(QStringLiteral("中"));
    zhButton->setCheckable(true);
    zhButton->setCursor(Qt::PointingHandCursor);
    zhButton->setToolTip(tr("切换到中文，重启后生效"));
    auto* enButton = new QToolButton;
    enButton->setObjectName(QStringLiteral("LanguageSwitchButton"));
    enButton->setText(QStringLiteral("EN"));
    enButton->setCheckable(true);
    enButton->setCursor(Qt::PointingHandCursor);
    enButton->setToolTip(tr("Switch to English after restart"));
    const QString configuredLanguage = aitrain_app::configuredLanguageCode();
    zhButton->setChecked(configuredLanguage == QStringLiteral("zh_CN"));
    enButton->setChecked(configuredLanguage == QStringLiteral("en_US"));
    languageLayout->addWidget(zhButton);
    languageLayout->addWidget(enButton);
    const auto storeLanguage = [this, zhButton, enButton](const QString& selected) {
        const QString previous = aitrain_app::configuredLanguageCode();
        aitrain_app::storeLanguageCode(selected);
        zhButton->setChecked(selected == QStringLiteral("zh_CN"));
        enButton->setChecked(selected == QStringLiteral("en_US"));
        if (previous != selected) {
            QMessageBox::information(this, tr("界面语言"), tr("语言设置已保存，重启 AITrain Studio 后生效。"));
        }
    };
    connect(zhButton, &QToolButton::clicked, this, [storeLanguage]() {
        storeLanguage(QStringLiteral("zh_CN"));
    });
    connect(enButton, &QToolButton::clicked, this, [storeLanguage]() {
        storeLanguage(QStringLiteral("en_US"));
    });

    layout->addWidget(titleBlock, 1);
    layout->addWidget(headerProjectLabel_);
    layout->addWidget(workerPill_);
    layout->addWidget(pluginPill_);
    layout->addWidget(gpuPill_);
    layout->addWidget(licensePill_);
    layout->addWidget(languageSwitch);
    return topBar;
}

QWidget* MainWindow::buildDashboardPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    projectLabel_ = inlineStatusLabel(QStringLiteral("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    gpuLabel_ = inlineStatusLabel(QStringLiteral("GPU / 运行时：未执行环境自检"));

    auto* grid = new QGridLayout;
    grid->setSpacing(12);
    auto* projectCard = createMetricCard(QStringLiteral("项目"), QStringLiteral("未打开"), QStringLiteral("当前本地工作目录"));
    dashboardProjectValue_ = projectCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(projectCard, 0, 0);
    auto* datasetCard = createMetricCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral("已登记并校验的数据集"));
    dashboardDatasetValue_ = datasetCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(datasetCard, 0, 1);
    auto* taskCard = createMetricCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理记录"));
    dashboardTaskValue_ = taskCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(taskCard, 0, 2);
    auto* modelCard = createMetricCard(QStringLiteral("模型版本"), QStringLiteral("0"), QStringLiteral("模型库 / 导出产物"));
    dashboardModelValue_ = modelCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(modelCard, 0, 3);
    auto* pluginCard = createMetricCard(QStringLiteral("插件"), QStringLiteral("0"), QStringLiteral("已加载能力插件"));
    dashboardPluginValue_ = pluginCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(pluginCard, 1, 0);
    auto* environmentCard = createMetricCard(QStringLiteral("环境"), QStringLiteral("待检测"), QStringLiteral("CUDA / TensorRT / Worker"));
    dashboardEnvironmentValue_ = environmentCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    grid->addWidget(environmentCard, 1, 1);

    auto* bottom = new QSplitter(Qt::Horizontal);

    auto* workflowPanel = new InfoPanel(QStringLiteral("下一步"));
    dashboardNextStepLabel_ = emptyStateLabel(QStringLiteral("打开项目后，按 数据集 -> 训练实验 -> 任务与产物 -> 模型导出 -> 推理验证 的顺序完成本机训练闭环。"));
    workflowPanel->bodyLayout()->addWidget(dashboardNextStepLabel_);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(12, 12, 12, 12);
    actionLayout->setSpacing(10);
    auto* projectButton = primaryButton(QStringLiteral("打开项目"));
    auto* datasetButton = new QPushButton(QStringLiteral("导入 / 校验数据"));
    auto* trainingButton = new QPushButton(QStringLiteral("启动训练实验"));
    auto* artifactButton = new QPushButton(QStringLiteral("查看任务与产物"));
    auto* modelRegistryButton = new QPushButton(QStringLiteral("模型库"));
    auto* inferenceButton = new QPushButton(QStringLiteral("推理验证"));
    connect(projectButton, &QPushButton::clicked, this, [this]() { showPage(ProjectPage, uiText("项目")); });
    connect(datasetButton, &QPushButton::clicked, this, [this]() { showPage(DatasetPage, uiText("数据集")); });
    connect(trainingButton, &QPushButton::clicked, this, [this]() { showPage(TrainingPage, uiText("训练实验")); });
    connect(artifactButton, &QPushButton::clicked, this, [this]() { showPage(TaskQueuePage, uiText("任务与产物")); });
    connect(modelRegistryButton, &QPushButton::clicked, this, [this]() { showPage(ModelRegistryPage, uiText("模型库")); });
    connect(inferenceButton, &QPushButton::clicked, this, [this]() { showPage(InferencePage, uiText("推理验证")); });
    actionLayout->addWidget(projectButton, 0, 0);
    actionLayout->addWidget(datasetButton, 0, 1);
    actionLayout->addWidget(trainingButton, 1, 0);
    actionLayout->addWidget(artifactButton, 1, 1);
    actionLayout->addWidget(modelRegistryButton, 2, 0);
    actionLayout->addWidget(inferenceButton, 2, 1);
    workflowPanel->bodyLayout()->addWidget(actionStrip);
    workflowPanel->bodyLayout()->addStretch();

    auto* recentPanel = new InfoPanel(QStringLiteral("最近任务"));
    recentTasksTable_ = new QTableWidget(0, 5);
    recentTasksTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("插件")
        << QStringLiteral("类型")
        << QStringLiteral("状态")
        << QStringLiteral("消息"));
    configureTable(recentTasksTable_);
    recentPanel->bodyLayout()->addWidget(recentTasksTable_);

    bottom->addWidget(workflowPanel);
    bottom->addWidget(recentPanel);
    bottom->setStretchFactor(0, 3);
    bottom->setStretchFactor(1, 4);

    layout->addWidget(projectLabel_);
    layout->addWidget(gpuLabel_);
    layout->addLayout(grid);
    layout->addWidget(bottom, 1);
    return page;
}

QWidget* MainWindow::buildProjectPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerOpenButton = primaryButton(QStringLiteral("创建 / 打开项目"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("PROJECT CONSOLE"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("项目控制台"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("统一管理本机训练项目、SQLite 元数据、数据集目录、运行目录和模型产物目录。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerOpenButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* statusCaption = new QLabel(QStringLiteral("状态"));
    statusCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* policyCaption = new QLabel(QStringLiteral("目录"));
    policyCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    projectConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("未打开项目。"));
    projectConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* policyStatus = inlineStatusLabel(QStringLiteral("项目会生成 datasets、runs、models 和 project.sqlite。"));
    policyStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    headerGrid->addWidget(statusCaption, 0, 0);
    headerGrid->addWidget(projectConsoleStatusLabel_, 0, 1);
    headerGrid->addWidget(policyCaption, 1, 0);
    headerGrid->addWidget(policyStatus, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* formPanel = new InfoPanel(QStringLiteral("项目设置"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    projectNameEdit_ = new QLineEdit(uiText("本地训练项目"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(QDir::home().filePath(QStringLiteral("AITrainProjects/local_project"))));
    auto* browseButton = new QPushButton(QStringLiteral("选择目录"));
    auto* createButton = primaryButton(QStringLiteral("创建 / 打开项目"));

    connect(browseButton, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择项目目录"));
        if (!directory.isEmpty()) {
            projectRootEdit_->setText(QDir::toNativeSeparators(directory));
        }
    });
    connect(headerOpenButton, &QPushButton::clicked, this, &MainWindow::createProject);
    connect(createButton, &QPushButton::clicked, this, &MainWindow::createProject);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(projectRootEdit_);
    pathLayout->addWidget(browseButton);
    form->addRow(QStringLiteral("项目名称"), projectNameEdit_);
    form->addRow(QStringLiteral("项目目录"), pathRow);
    formPanel->bodyLayout()->addLayout(form);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    actionLayout->addWidget(mutedLabel(QStringLiteral("打开项目后，数据集、任务、导出记录和环境检查都会写入 project.sqlite。")), 1);
    actionLayout->addWidget(createButton);
    formPanel->bodyLayout()->addWidget(actionStrip);
    formPanel->bodyLayout()->addStretch();

    auto* summaryPanel = new InfoPanel(QStringLiteral("项目摘要"));
    auto* summaryList = new QVBoxLayout;
    summaryList->setSpacing(10);
    auto* pathCard = createCompactSummaryCard(QStringLiteral("项目路径"), QStringLiteral("未打开"), QStringLiteral("当前项目根目录"));
    projectPathSummaryLabel_ = pathCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* sqliteCard = createCompactSummaryCard(QStringLiteral("SQLite"), QStringLiteral("未连接"), QStringLiteral("项目元数据状态"));
    projectSqliteSummaryLabel_ = sqliteCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* datasetCard = createCompactSummaryCard(QStringLiteral("数据集"), QStringLiteral("0"), QStringLiteral("已登记数据集"));
    projectDatasetSummaryLabel_ = datasetCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* taskCard = createCompactSummaryCard(QStringLiteral("任务"), QStringLiteral("0"), QStringLiteral("训练、校验、导出、推理"));
    projectTaskSummaryLabel_ = taskCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    auto* exportCard = createCompactSummaryCard(QStringLiteral("模型导出"), QStringLiteral("0"), QStringLiteral("已记录导出产物"));
    projectExportSummaryLabel_ = exportCard->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
    summaryList->addWidget(pathCard);
    summaryList->addWidget(sqliteCard);
    summaryList->addWidget(datasetCard);
    summaryList->addWidget(taskCard);
    summaryList->addWidget(exportCard);
    summaryPanel->bodyLayout()->addLayout(summaryList);

    auto* structurePanel = new InfoPanel(QStringLiteral("标准目录结构"));
    auto* structure = new QPlainTextEdit;
    structure->setReadOnly(true);
    structure->setMaximumHeight(170);
    structure->setPlainText(QStringLiteral("datasets/\n  raw/\n  normalized/\nruns/\n  <task-id>/\nmodels/\n  exported/\nproject.sqlite"));
    structurePanel->bodyLayout()->addWidget(structure);
    structurePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("项目页只负责创建和打开工作区；训练、导出和推理仍通过 Worker 执行。")));
    summaryPanel->bodyLayout()->addWidget(structurePanel);

    layout->addWidget(headerPanel);
    layout->addWidget(formPanel);
    layout->addWidget(summaryPanel);
    layout->addStretch();
    page->setWidget(content);
    updateProjectSummary();
    return page;
}

QWidget* MainWindow::buildDatasetPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* inputPanel = new InfoPanel(QStringLiteral("数据集操作"));
    auto* form = new QFormLayout;
    datasetPathEdit_ = new QLineEdit;
    auto* browseButton = new QPushButton(QStringLiteral("选择数据集"));
    connect(browseButton, &QPushButton::clicked, this, &MainWindow::browseDataset);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(datasetPathEdit_);
    pathLayout->addWidget(browseButton);

    datasetFormatCombo_ = new QComboBox;
    connect(datasetFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        currentDatasetFormat_ = currentDatasetFormat();
        currentDatasetValid_ = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
    });
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QStringLiteral("格式"), datasetFormatCombo_);

    splitOutputEdit_ = new QLineEdit;
    splitOutputEdit_->setPlaceholderText(QStringLiteral("默认输出到当前项目 datasets/normalized"));
    splitTrainRatioEdit_ = new QLineEdit(QStringLiteral("0.8"));
    splitValRatioEdit_ = new QLineEdit(QStringLiteral("0.2"));
    splitTestRatioEdit_ = new QLineEdit(QStringLiteral("0.0"));
    splitSeedEdit_ = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked, this, &MainWindow::splitDataset);
    auto* curateButton = new QPushButton(QStringLiteral("生成质量报告"));
    connect(curateButton, &QPushButton::clicked, this, &MainWindow::curateDataset);
    auto* snapshotButton = new QPushButton(QStringLiteral("创建数据快照"));
    connect(snapshotButton, &QPushButton::clicked, this, &MainWindow::createDatasetSnapshot);
    auto* openFixListButton = new QPushButton(QStringLiteral("打开问题清单"));
    connect(openFixListButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityFixList);
    auto* fixWithXAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling 修复"));
    connect(fixWithXAnyButton, &QPushButton::clicked, this, &MainWindow::launchXAnyLabelingForQualityFix);
    auto* ratioRow = new QWidget;
    auto* ratioLayout = new QHBoxLayout(ratioRow);
    ratioLayout->setContentsMargins(0, 0, 0, 0);
    ratioLayout->addWidget(splitTrainRatioEdit_);
    ratioLayout->addWidget(splitValRatioEdit_);
    ratioLayout->addWidget(splitTestRatioEdit_);
    ratioLayout->addWidget(splitSeedEdit_);
    form->addRow(QStringLiteral("输出目录"), splitOutputEdit_);
    form->addRow(QStringLiteral("train / val / test / seed"), ratioRow);
    inputPanel->bodyLayout()->addLayout(form);
    auto* datasetActionRow = new QHBoxLayout;
    datasetActionRow->addWidget(validateButton);
    datasetActionRow->addWidget(splitButton);
    datasetActionRow->addWidget(curateButton);
    datasetActionRow->addWidget(snapshotButton);
    datasetActionRow->addWidget(openFixListButton);
    datasetActionRow->addWidget(fixWithXAnyButton);
    datasetActionRow->addStretch();
    inputPanel->bodyLayout()->addLayout(datasetActionRow);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("所选数据集详情"));
    datasetDetailLabel_ = inlineStatusLabel(QStringLiteral("选择或导入数据集后显示格式、样本数、校验状态和最近报告。"));
    validationSummaryLabel_ = mutedLabel(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    validationIssuesTable_ = new QTableWidget(0, 5);
    validationIssuesTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("级别")
        << QStringLiteral("代码")
        << QStringLiteral("文件")
        << QStringLiteral("行号")
        << QStringLiteral("说明"));
    configureTable(validationIssuesTable_);
    validationOutput_ = new QPlainTextEdit;
    validationOutput_->setReadOnly(true);
    validationOutput_->setPlainText(QStringLiteral("校验报告 JSON 会显示在这里。"));
    resultPanel->bodyLayout()->addWidget(datasetDetailLabel_);
    resultPanel->bodyLayout()->addWidget(validationSummaryLabel_);
    resultPanel->bodyLayout()->addWidget(validationIssuesTable_, 2);
    resultPanel->bodyLayout()->addWidget(validationOutput_);

    auto* toolsPanel = new InfoPanel(QStringLiteral("数据集库与样本预览"));
    datasetListTable_ = new QTableWidget(0, 5);
    datasetListTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("数据集")
        << QStringLiteral("格式")
        << QStringLiteral("状态")
        << QStringLiteral("样本")
        << QStringLiteral("路径"));
    configureTable(datasetListTable_);
    connect(datasetListTable_, &QTableWidget::itemSelectionChanged, this, [this]() {
        if (!datasetListTable_ || datasetListTable_->selectedItems().isEmpty()) {
            return;
        }
        const int row = datasetListTable_->selectedItems().first()->row();
        const QString path = datasetListTable_->item(row, 4) ? datasetListTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        const QString format = datasetListTable_->item(row, 1) ? datasetListTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
        if (!path.isEmpty()) {
            datasetPathEdit_->setText(QDir::toNativeSeparators(path));
            const int formatIndex = datasetFormatCombo_->findData(format);
            if (formatIndex >= 0) {
                datasetFormatCombo_->setCurrentIndex(formatIndex);
            }
            currentDatasetPath_ = path;
            currentDatasetFormat_ = format;
            currentDatasetValid_ = datasetListTable_->item(row, 2)
                && datasetListTable_->item(row, 2)->data(Qt::UserRole).toString() == QStringLiteral("valid");
            updateTrainingSelectionSummary();
            refreshTrainingDefaults();
        }
    });
    datasetPreviewTable_ = new QTableWidget(0, 2);
    datasetPreviewTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("样本") << QStringLiteral("标签 / 说明"));
    configureTable(datasetPreviewTable_);
    auto* annotationPanel = new QGroupBox(QStringLiteral("外部标注工具"));
    auto* annotationLayout = new QGridLayout(annotationPanel);
    annotationLayout->setContentsMargins(10, 12, 10, 10);
    annotationLayout->setHorizontalSpacing(10);
    annotationLayout->setVerticalSpacing(8);
    auto* annotationSummary = mutedLabel(QStringLiteral("默认使用 X-AnyLabeling。推荐导出：检测使用 YOLO bbox，分割使用 YOLO polygon；PaddleOCR Det 使用 det_gt.txt，PaddleOCR Rec 使用 rec_gt.txt + dict.txt。"));
    annotationToolStatusLabel_ = inlineStatusLabel(xAnyLabelingStatusText());
    auto* launchAnnotationToolButton = new QPushButton(QStringLiteral("启动 X-AnyLabeling"));
    auto* refreshAnnotationStatusButton = new QPushButton(QStringLiteral("检测状态"));
    auto* refreshDatasetAfterAnnotationButton = new QPushButton(QStringLiteral("标注后刷新 / 重新校验"));
    auto* openDatasetDirButton = new QPushButton(QStringLiteral("打开数据目录"));
    connect(refreshAnnotationStatusButton, &QPushButton::clicked, this, &MainWindow::updateAnnotationToolStatus);
    connect(refreshDatasetAfterAnnotationButton, &QPushButton::clicked, this, &MainWindow::refreshAfterAnnotation);
    connect(openDatasetDirButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    });
    connect(launchAnnotationToolButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        const QString program = resolvedXAnyLabelingProgram();
        if (program.isEmpty()) {
            updateAnnotationToolStatus();
            QMessageBox::warning(this,
                QStringLiteral("X-AnyLabeling"),
                uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
            return;
        }
        const QStringList arguments = {QStringLiteral("--filename"), datasetPath, QStringLiteral("--no-auto-update-check")};
        if (QProcess::startDetached(program, arguments)) {
            updateAnnotationToolStatus();
            statusBar()->showMessage(uiText("已启动 X-AnyLabeling：%1").arg(QDir::toNativeSeparators(datasetPath)), 4000);
            return;
        }
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
    });
    annotationLayout->addWidget(annotationSummary, 0, 0, 1, 4);
    annotationLayout->addWidget(annotationToolStatusLabel_, 1, 0, 1, 4);
    annotationLayout->addWidget(launchAnnotationToolButton, 2, 0);
    annotationLayout->addWidget(refreshAnnotationStatusButton, 2, 1);
    annotationLayout->addWidget(refreshDatasetAfterAnnotationButton, 2, 2);
    annotationLayout->addWidget(openDatasetDirButton, 2, 3);
    annotationLayout->setColumnStretch(4, 1);
    toolsPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("已登记数据集")));
    toolsPanel->bodyLayout()->addWidget(datasetListTable_, 1);
    toolsPanel->bodyLayout()->addWidget(annotationPanel);
    toolsPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("样本预览")));
    toolsPanel->bodyLayout()->addWidget(datasetPreviewTable_, 1);
    toolsPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec。")));
    toolsPanel->bodyLayout()->addStretch();

    splitter->addWidget(toolsPanel);
    splitter->addWidget(resultPanel);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);

    layout->addWidget(inputPanel);
    layout->addWidget(splitter, 1);
    return page;
}

QWidget* MainWindow::buildTrainingPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(12);

    pluginCombo_ = new QComboBox;
    taskTypeCombo_ = new QComboBox;
    trainingBackendCombo_ = new QComboBox;
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("ultralytics_yolo_detect")), QStringLiteral("ultralytics_yolo_detect"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("ultralytics_yolo_segment")), QStringLiteral("ultralytics_yolo_segment"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_det_official")), QStringLiteral("paddleocr_det_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec")), QStringLiteral("paddleocr_rec"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_rec_official")), QStringLiteral("paddleocr_rec_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("paddleocr_system_official")), QStringLiteral("paddleocr_system_official"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("tiny_linear_detector")), QStringLiteral("tiny_linear_detector"));
    trainingBackendCombo_->addItem(backendLabel(QStringLiteral("python_mock")), QStringLiteral("python_mock"));
    modelPresetCombo_ = new QComboBox;
    modelPresetCombo_->setEditable(true);
    modelPresetCombo_->addItems(QStringList()
        << QStringLiteral("yolov8n.yaml")
        << QStringLiteral("yolov8n-seg.yaml")
        << QStringLiteral("PP-OCRv4_mobile_det")
        << QStringLiteral("paddle_ctc_smoke")
        << QStringLiteral("PP-OCRv4_mobile_rec")
        << QStringLiteral("PP-OCRv4_det_rec_system")
        << QStringLiteral("diagnostic"));
    epochsEdit_ = new QLineEdit(QStringLiteral("20"));
    batchEdit_ = new QLineEdit(QStringLiteral("8"));
    imageSizeEdit_ = new QLineEdit(QStringLiteral("640"));
    gridSizeEdit_ = new QLineEdit(QStringLiteral("4"));
    resumeCheckpointEdit_ = new QLineEdit;
    resumeCheckpointEdit_->setPlaceholderText(QStringLiteral("可选：选择已有 checkpoint 继续训练"));
    horizontalFlipCheck_ = new QCheckBox(QStringLiteral("水平翻转增强"));
    colorJitterCheck_ = new QCheckBox(QStringLiteral("亮度扰动增强"));
    connect(pluginCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        taskTypeCombo_->clear();
        auto* plugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
        if (plugin) {
            addTaskTypeItems(taskTypeCombo_, plugin->manifest().taskTypes);
        }
        refreshTrainingDefaults();
    });
    connect(taskTypeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::refreshTrainingDefaults);
    connect(trainingBackendCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        if (modelPresetCombo_ && trainingBackendCombo_) {
            modelPresetCombo_->setCurrentText(defaultModelForBackend(trainingBackendCombo_->currentData().toString()));
        }
        updateTrainingSelectionSummary();
    });
    trainingDatasetSummaryLabel_ = inlineStatusLabel(QStringLiteral("当前数据集：未选择。请先在数据集页导入并通过校验。"));
    trainingDatasetSummaryLabel_->setMinimumHeight(34);
    trainingBackendHintLabel_ = mutedLabel(QStringLiteral("官方后端会由 Worker 启动独立 Python 进程；scaffold 后端只用于高级诊断。"));
    trainingRunSummaryLabel_ = inlineStatusLabel(QStringLiteral("等待配置训练实验。"));
    trainingRunSummaryLabel_->setMinimumHeight(42);

    auto* startButton = primaryButton(QStringLiteral("启动训练"));
    startButton->setObjectName(QStringLiteral("GreenButton"));
    auto* pauseButton = new QPushButton(QStringLiteral("暂停任务"));
    auto* resumeButton = new QPushButton(QStringLiteral("继续任务"));
    auto* cancelButton = dangerButton(QStringLiteral("取消任务"));
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(pauseButton, &QPushButton::clicked, &worker_, &WorkerClient::pause);
    connect(resumeButton, &QPushButton::clicked, &worker_, &WorkerClient::resume);
    connect(cancelButton, &QPushButton::clicked, &worker_, &WorkerClient::cancel);

    trainingDatasetSummaryLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    trainingRunSummaryLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("LOCAL TRAINING WORKBENCH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("训练实验"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("按数据集类型优先选择官方 YOLO / OCR 后端；运行结果沉淀到任务与产物。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(startButton);
    headerTop->addWidget(pauseButton);
    headerTop->addWidget(resumeButton);
    headerTop->addWidget(cancelButton);
    headerRoot->addLayout(headerTop);

    auto* headerLayout = new QGridLayout;
    headerLayout->setHorizontalSpacing(12);
    headerLayout->setVerticalSpacing(8);
    headerLayout->setColumnStretch(0, 0);
    headerLayout->setColumnStretch(1, 1);
    auto* datasetHeader = new QLabel(QStringLiteral("数据集"));
    datasetHeader->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* summaryHeader = new QLabel(QStringLiteral("摘要"));
    summaryHeader->setObjectName(QStringLiteral("ExperimentMeta"));
    headerLayout->addWidget(datasetHeader, 0, 0);
    headerLayout->addWidget(trainingDatasetSummaryLabel_, 0, 1);
    headerLayout->addWidget(summaryHeader, 1, 0);
    headerLayout->addWidget(trainingRunSummaryLabel_, 1, 1);
    headerRoot->addLayout(headerLayout);

    auto* setupPanel = new InfoPanel(QStringLiteral("实验参数"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("任务类型"), taskTypeCombo_);
    form->addRow(QStringLiteral("训练后端"), trainingBackendCombo_);
    form->addRow(QStringLiteral("模型预设"), modelPresetCombo_);
    form->addRow(QStringLiteral("Epochs"), epochsEdit_);
    form->addRow(QStringLiteral("Batch Size"), batchEdit_);
    form->addRow(QStringLiteral("Image Size"), imageSizeEdit_);
    setupPanel->bodyLayout()->addLayout(form);
    setupPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("当前模型能力说明")));
    setupPanel->bodyLayout()->addWidget(trainingBackendHintLabel_);

    auto* advancedGroup = new QGroupBox(QStringLiteral("高级 / 诊断后端"));
    auto* advancedForm = new QFormLayout(advancedGroup);
    advancedForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    advancedForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    advancedForm->setHorizontalSpacing(14);
    advancedForm->setVerticalSpacing(10);
    advancedForm->addRow(QStringLiteral("能力插件"), pluginCombo_);
    advancedForm->addRow(QStringLiteral("Grid Size"), gridSizeEdit_);
    advancedForm->addRow(QStringLiteral("Resume"), resumeCheckpointEdit_);
    auto* augmentRow = new QWidget;
    auto* augmentLayout = new QHBoxLayout(augmentRow);
    augmentLayout->setContentsMargins(0, 0, 0, 0);
    augmentLayout->setSpacing(14);
    augmentLayout->addWidget(horizontalFlipCheck_);
    augmentLayout->addWidget(colorJitterCheck_);
    augmentLayout->addStretch();
    advancedForm->addRow(QStringLiteral("Augment"), augmentRow);
    setupPanel->bodyLayout()->addWidget(advancedGroup);
    setupPanel->bodyLayout()->addStretch();

    auto* setupScroll = new QScrollArea;
    setupScroll->setWidget(setupPanel);
    setupScroll->setWidgetResizable(true);
    setupScroll->setFrameShape(QFrame::NoFrame);
    setupScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setupScroll->setMinimumWidth(390);

    auto* monitorPanel = new InfoPanel(QStringLiteral("训练监控"));
    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    metricsWidget_ = new MetricsWidget;
    monitorPanel->bodyLayout()->addWidget(progressBar_);
    monitorPanel->bodyLayout()->addWidget(metricsWidget_, 1);

    auto* artifactPanel = new InfoPanel(QStringLiteral("任务与产物"));
    artifactPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("运行后会记录 checkpoint、训练报告、ONNX、预览图和请求参数。完整产物浏览请进入“任务与产物”。")));
    artifactPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("主流程优先使用官方 YOLO / PaddleOCR 后端；PaddleOCR System 产物来自官方工具链，不代表 C++ DB 后处理已经接入。")));
    latestCheckpointLabel_ = mutedLabel(QStringLiteral("最新 checkpoint：暂无"));
    latestPreviewPathLabel_ = mutedLabel(QStringLiteral("最新预览：暂无"));
    latestPreviewImageLabel_ = new QLabel(QStringLiteral("暂无预览图"));
    latestPreviewImageLabel_->setObjectName(QStringLiteral("MutedText"));
    latestPreviewImageLabel_->setAlignment(Qt::AlignCenter);
    latestPreviewImageLabel_->setMinimumHeight(120);
    latestPreviewImageLabel_->setFrameShape(QFrame::StyledPanel);
    latestPreviewImageLabel_->setScaledContents(false);
    artifactPanel->bodyLayout()->addWidget(latestCheckpointLabel_);
    artifactPanel->bodyLayout()->addWidget(latestPreviewPathLabel_);
    artifactPanel->bodyLayout()->addWidget(latestPreviewImageLabel_);
    artifactPanel->bodyLayout()->addStretch();

    auto* logPanel = new InfoPanel(QStringLiteral("训练日志"));
    logEdit_ = new QTextEdit;
    logEdit_->setObjectName(QStringLiteral("LogView"));
    logEdit_->setReadOnly(true);
    logPanel->bodyLayout()->addWidget(logEdit_);

    auto* lowerSplitter = new QSplitter(Qt::Horizontal);
    lowerSplitter->addWidget(artifactPanel);
    lowerSplitter->addWidget(logPanel);
    lowerSplitter->setStretchFactor(0, 1);
    lowerSplitter->setStretchFactor(1, 2);

    auto* rightSplitter = new QSplitter(Qt::Vertical);
    rightSplitter->addWidget(monitorPanel);
    rightSplitter->addWidget(lowerSplitter);
    rightSplitter->setStretchFactor(0, 1);
    rightSplitter->setStretchFactor(1, 2);

    auto* bodySplitter = new QSplitter(Qt::Horizontal);
    bodySplitter->addWidget(setupScroll);
    bodySplitter->addWidget(rightSplitter);
    bodySplitter->setStretchFactor(0, 5);
    bodySplitter->setStretchFactor(1, 6);
    bodySplitter->setSizes(QList<int>() << 520 << 620);

    layout->addWidget(headerPanel);
    layout->addWidget(bodySplitter, 1);
    return page;
}

QWidget* MainWindow::buildTaskQueuePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("历史操作"));
    auto* row = new QHBoxLayout;
    auto* refreshButton = primaryButton(QStringLiteral("刷新历史"));
    auto* cancelButton = dangerButton(QStringLiteral("取消选中任务"));
    auto* reproduceButton = new QPushButton(QStringLiteral("复现实验"));
    taskKindFilterCombo_ = new QComboBox;
    taskKindFilterCombo_->addItem(uiText("全部类别"), QString());
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Train), QStringLiteral("train"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Validate), QStringLiteral("validate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Export), QStringLiteral("export"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Infer), QStringLiteral("infer"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Evaluate), QStringLiteral("evaluate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Benchmark), QStringLiteral("benchmark"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Curate), QStringLiteral("curate"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Snapshot), QStringLiteral("snapshot"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Pipeline), QStringLiteral("pipeline"));
    taskKindFilterCombo_->addItem(taskKindLabel(aitrain::TaskKind::Report), QStringLiteral("report"));
    taskStateFilterCombo_ = new QComboBox;
    taskStateFilterCombo_->addItem(uiText("全部状态"), QString());
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Queued), QStringLiteral("queued"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Running), QStringLiteral("running"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Completed), QStringLiteral("completed"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Failed), QStringLiteral("failed"));
    taskStateFilterCombo_->addItem(taskStateLabel(aitrain::TaskState::Canceled), QStringLiteral("canceled"));
    taskSearchEdit_ = new QLineEdit;
    taskSearchEdit_->setPlaceholderText(QStringLiteral("搜索任务、后端、消息"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::updateRecentTasks);
    connect(cancelButton, &QPushButton::clicked, this, &MainWindow::cancelSelectedTask);
    connect(reproduceButton, &QPushButton::clicked, this, &MainWindow::reproduceSelectedTrainingTask);
    connect(taskKindFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyTaskFilters);
    connect(taskStateFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyTaskFilters);
    connect(taskSearchEdit_, &QLineEdit::textChanged, this, &MainWindow::applyTaskFilters);
    row->addWidget(refreshButton);
    row->addWidget(cancelButton);
    row->addWidget(reproduceButton);
    row->addSpacing(12);
    row->addWidget(new QLabel(QStringLiteral("类别")));
    row->addWidget(taskKindFilterCombo_);
    row->addWidget(new QLabel(QStringLiteral("状态")));
    row->addWidget(taskStateFilterCombo_);
    row->addWidget(taskSearchEdit_, 1);
    row->addStretch();
    toolbar->bodyLayout()->addLayout(row);
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("这里统一追踪训练、校验、划分、导出和推理任务；运行产物在下方详情区集中查看。")));

    auto* tablePanel = new InfoPanel(QStringLiteral("任务历史"));
    taskQueueTable_ = new QTableWidget(0, 7);
    taskQueueTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("类别")
        << QStringLiteral("插件")
        << QStringLiteral("类型")
        << QStringLiteral("状态")
        << QStringLiteral("更新时间")
        << QStringLiteral("消息"));
    configureTable(taskQueueTable_);
    connect(taskQueueTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateSelectedTaskDetails);
    tablePanel->bodyLayout()->addWidget(taskQueueTable_);

    auto* detailPanel = new InfoPanel(QStringLiteral("任务详情与产物"));
    selectedTaskSummaryLabel_ = mutedLabel(QStringLiteral("请选择一个任务查看产物、指标和导出记录。"));
    taskArtifactTable_ = new QTableWidget(0, 4);
    taskArtifactTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("类型")
        << QStringLiteral("路径")
        << QStringLiteral("说明")
        << QStringLiteral("时间"));
    configureTable(taskArtifactTable_);
    connect(taskArtifactTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateArtifactPreviewFromSelection);

    taskMetricTable_ = new QTableWidget(0, 4);
    taskMetricTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("指标")
        << QStringLiteral("值")
        << QStringLiteral("Step")
        << QStringLiteral("Epoch"));
    configureTable(taskMetricTable_);

    taskExportTable_ = new QTableWidget(0, 3);
    taskExportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("路径")
        << QStringLiteral("时间"));
    configureTable(taskExportTable_);

    artifactImagePreviewLabel_ = new QLabel(QStringLiteral("暂无产物预览"));
    artifactImagePreviewLabel_->setObjectName(QStringLiteral("MutedText"));
    artifactImagePreviewLabel_->setAlignment(Qt::AlignCenter);
    artifactImagePreviewLabel_->setMinimumHeight(180);
    artifactImagePreviewLabel_->setFrameShape(QFrame::StyledPanel);
    artifactPreviewText_ = new QPlainTextEdit;
    artifactPreviewText_->setReadOnly(true);
    artifactPreviewText_->setPlainText(QStringLiteral("选择一个产物后显示摘要。"));
    auto* artifactDefaultPreview = new QWidget;
    auto* artifactDefaultLayout = new QVBoxLayout(artifactDefaultPreview);
    artifactDefaultLayout->setContentsMargins(0, 0, 0, 0);
    artifactDefaultLayout->setSpacing(8);
    artifactDefaultLayout->addWidget(artifactImagePreviewLabel_, 1);
    artifactDefaultLayout->addWidget(artifactPreviewText_, 2);
    artifactPreviewStack_ = new QStackedWidget;
    artifactPreviewStack_->addWidget(artifactDefaultPreview);
    artifactEvaluationReportView_ = new EvaluationReportView;
    auto* artifactEvaluationScroll = new QScrollArea;
    artifactEvaluationScroll->setWidget(artifactEvaluationReportView_);
    artifactEvaluationScroll->setWidgetResizable(true);
    artifactEvaluationScroll->setFrameShape(QFrame::NoFrame);
    artifactEvaluationScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    artifactEvaluationScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    artifactPreviewStack_->addWidget(artifactEvaluationScroll);

    auto* actionRow = new QHBoxLayout;
    auto* openDirButton = new QPushButton(QStringLiteral("打开目录"));
    auto* copyPathButton = new QPushButton(QStringLiteral("复制路径"));
    auto* useInferButton = new QPushButton(QStringLiteral("用作推理模型"));
    auto* useExportButton = new QPushButton(QStringLiteral("用作导出输入"));
    auto* registerModelButton = new QPushButton(QStringLiteral("注册模型版本"));
    auto* evaluateButton = new QPushButton(QStringLiteral("评估"));
    auto* benchmarkButton = new QPushButton(QStringLiteral("基准"));
    auto* reportButton = new QPushButton(QStringLiteral("交付报告"));
    connect(openDirButton, &QPushButton::clicked, this, &MainWindow::openSelectedArtifactDirectory);
    connect(copyPathButton, &QPushButton::clicked, this, &MainWindow::copySelectedArtifactPath);
    connect(useInferButton, &QPushButton::clicked, this, &MainWindow::useSelectedArtifactForInference);
    connect(useExportButton, &QPushButton::clicked, this, &MainWindow::useSelectedArtifactForExport);
    connect(registerModelButton, &QPushButton::clicked, this, &MainWindow::registerSelectedArtifactAsModelVersion);
    connect(evaluateButton, &QPushButton::clicked, this, &MainWindow::evaluateSelectedArtifact);
    connect(benchmarkButton, &QPushButton::clicked, this, &MainWindow::benchmarkSelectedArtifact);
    connect(reportButton, &QPushButton::clicked, this, &MainWindow::generateDeliveryReportFromSelectedArtifact);
    actionRow->addWidget(openDirButton);
    actionRow->addWidget(copyPathButton);
    actionRow->addWidget(useInferButton);
    actionRow->addWidget(useExportButton);
    actionRow->addWidget(registerModelButton);
    actionRow->addWidget(evaluateButton);
    actionRow->addWidget(benchmarkButton);
    actionRow->addWidget(reportButton);
    actionRow->addStretch();

    auto* detailSplitter = new QSplitter(Qt::Horizontal);
    auto* leftDetail = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftDetail);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->addWidget(selectedTaskSummaryLabel_);
    leftLayout->addWidget(taskArtifactTable_, 2);
    leftLayout->addWidget(taskMetricTable_, 1);
    leftLayout->addWidget(taskExportTable_, 1);
    auto* rightDetail = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightDetail);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->addLayout(actionRow);
    rightLayout->addWidget(artifactPreviewStack_, 1);
    detailSplitter->addWidget(leftDetail);
    detailSplitter->addWidget(rightDetail);
    detailSplitter->setStretchFactor(0, 2);
    detailSplitter->setStretchFactor(1, 2);
    detailPanel->bodyLayout()->addWidget(detailSplitter);

    auto* historySplitter = new QSplitter(Qt::Vertical);
    historySplitter->addWidget(tablePanel);
    historySplitter->addWidget(detailPanel);
    historySplitter->setStretchFactor(0, 2);
    historySplitter->setStretchFactor(1, 3);

    layout->addWidget(toolbar);
    layout->addWidget(historySplitter, 1);
    return page;
}

QWidget* MainWindow::buildModelRegistryPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("模型库"));
    auto* row = new QHBoxLayout;
    auto* refreshButton = primaryButton(QStringLiteral("刷新模型库"));
    auto* inferButton = new QPushButton(QStringLiteral("选中模型用于推理"));
    auto* exportButton = new QPushButton(QStringLiteral("选中模型用于导出"));
    auto* pipelineButton = new QPushButton(QStringLiteral("执行本地流水线"));
    auto* reportsButton = new QPushButton(QStringLiteral("查看评估报告"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);
    connect(inferButton, &QPushButton::clicked, this, [this]() {
        if (!modelVersionTable_ || modelVersionTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个模型版本。"));
            return;
        }
        const int row = modelVersionTable_->selectedItems().first()->row();
        QString path = modelVersionTable_->item(row, 4) ? modelVersionTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        if (path.isEmpty()) {
            path = modelVersionTable_->item(row, 3) ? modelVersionTable_->item(row, 3)->data(Qt::UserRole).toString() : QString();
        }
        if (path.isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中模型版本没有可用 checkpoint 或 ONNX 路径。"));
            return;
        }
        if (inferenceCheckpointEdit_) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        }
        showPage(InferencePage, uiText("推理验证"));
    });
    connect(exportButton, &QPushButton::clicked, this, [this]() {
        if (!modelVersionTable_ || modelVersionTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个模型版本。"));
            return;
        }
        const int row = modelVersionTable_->selectedItems().first()->row();
        QString path = modelVersionTable_->item(row, 3) ? modelVersionTable_->item(row, 3)->data(Qt::UserRole).toString() : QString();
        if (path.isEmpty()) {
            path = modelVersionTable_->item(row, 4) ? modelVersionTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        }
        if (path.isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中模型版本没有可用 checkpoint 或 ONNX 路径。"));
            return;
        }
        if (conversionCheckpointEdit_) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        }
        showPage(ConversionPage, uiText("模型导出"));
    });
    connect(pipelineButton, &QPushButton::clicked, this, &MainWindow::runLocalPipelinePlanFromCurrentDataset);
    connect(reportsButton, &QPushButton::clicked, this, &MainWindow::openEvaluationReportsPage);
    row->addWidget(refreshButton);
    row->addWidget(inferButton);
    row->addWidget(exportButton);
    row->addWidget(pipelineButton);
    row->addWidget(reportsButton);
    row->addStretch();
    modelRegistrySummaryLabel_ = mutedLabel(QStringLiteral("训练产物可从“任务与产物”注册为模型版本；评估报告已拆分到独立页面，模型库聚焦版本管理、导出和推理入口。"));
    toolbar->bodyLayout()->addLayout(row);
    toolbar->bodyLayout()->addWidget(modelRegistrySummaryLabel_);

    auto* splitter = new QSplitter(Qt::Vertical);

    auto* modelPanel = new InfoPanel(QStringLiteral("模型版本"));
    modelVersionTable_ = new QTableWidget(0, 8);
    modelVersionTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("模型")
        << QStringLiteral("版本")
        << QStringLiteral("状态")
        << QStringLiteral("Checkpoint")
        << QStringLiteral("ONNX")
        << QStringLiteral("来源任务")
        << QStringLiteral("交付摘要")
        << QStringLiteral("更新时间"));
    configureTable(modelVersionTable_);
    modelPanel->bodyLayout()->addWidget(modelVersionTable_);

    auto* pipelinePanel = new InfoPanel(QStringLiteral("流水线记录"));
    pipelineRunTable_ = new QTableWidget(0, 5);
    pipelineRunTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("名称")
        << QStringLiteral("模板")
        << QStringLiteral("状态")
        << QStringLiteral("摘要")
        << QStringLiteral("更新时间"));
    configureTable(pipelineRunTable_);
    pipelinePanel->bodyLayout()->addWidget(pipelineRunTable_);
    splitter->addWidget(modelPanel);
    splitter->addWidget(pipelinePanel);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 2);

    layout->addWidget(toolbar);
    layout->addWidget(splitter, 1);
    return page;
}

QWidget* MainWindow::buildEvaluationReportsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("评估报告"));
    auto* row = new QHBoxLayout;
    auto* refreshButton = primaryButton(QStringLiteral("刷新评估报告"));
    auto* backToModelsButton = new QPushButton(QStringLiteral("查看模型库"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);
    connect(backToModelsButton, &QPushButton::clicked, this, [this]() {
        showPage(ModelRegistryPage, uiText("模型库"));
    });
    row->addWidget(refreshButton);
    row->addWidget(backToModelsButton);
    row->addStretch();
    toolbar->bodyLayout()->addLayout(row);
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("集中查看最近评估报告、任务类型、报告路径和详细可视化结果；模型版本管理保留在“模型库”。")));

    auto* splitter = new QSplitter(Qt::Vertical);

    auto* reportPanel = new InfoPanel(QStringLiteral("最近评估报告"));
    evaluationReportTable_ = new QTableWidget(0, 5);
    evaluationReportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("任务")
        << QStringLiteral("类型")
        << QStringLiteral("模型")
        << QStringLiteral("报告")
        << QStringLiteral("时间"));
    configureTable(evaluationReportTable_);
    connect(evaluationReportTable_, &QTableWidget::itemSelectionChanged, this, &MainWindow::updateSelectedEvaluationReportDetails);
    reportPanel->bodyLayout()->addWidget(evaluationReportTable_);

    auto* reportDetailPanel = new InfoPanel(QStringLiteral("评估报告详情"));
    evaluationReportView_ = new EvaluationReportView;
    auto* evaluationReportScroll = new QScrollArea;
    evaluationReportScroll->setWidget(evaluationReportView_);
    evaluationReportScroll->setWidgetResizable(true);
    evaluationReportScroll->setFrameShape(QFrame::NoFrame);
    evaluationReportScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    evaluationReportScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    reportDetailPanel->bodyLayout()->addWidget(evaluationReportScroll);

    splitter->addWidget(reportPanel);
    splitter->addWidget(reportDetailPanel);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 3);

    layout->addWidget(toolbar);
    layout->addWidget(splitter, 1);
    return page;
}

QWidget* MainWindow::buildConversionPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerExportButton = primaryButton(QStringLiteral("开始导出"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("MODEL DELIVERY WORKBENCH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("模型导出"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("从训练产物生成 ONNX、NCNN、TensorRT 或诊断 JSON，并把报告写入任务与产物。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerExportButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* sourceLabel = new QLabel(QStringLiteral("输入"));
    sourceLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* policyLabel = new QLabel(QStringLiteral("策略"));
    policyLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* sourceStatus = inlineStatusLabel(QStringLiteral("优先从“任务与产物”选择 checkpoint、ONNX 或官方训练产物。"));
    sourceStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* policyStatus = inlineStatusLabel(QStringLiteral("导出只通过 Worker 执行；TensorRT 仍需 RTX / SM 75+ 外部验收。"));
    policyStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    headerGrid->addWidget(sourceLabel, 0, 0);
    headerGrid->addWidget(sourceStatus, 0, 1);
    headerGrid->addWidget(policyLabel, 1, 0);
    headerGrid->addWidget(policyStatus, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* setupPanel = new InfoPanel(QStringLiteral("导出设置"));
    conversionCheckpointEdit_ = new QLineEdit;
    conversionCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 checkpoint / ONNX / AITrain export"));
    auto* chooseCheckpointButton = new QPushButton(QStringLiteral("选择模型产物"));
    connect(chooseCheckpointButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型产物"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.pt *.pdparams *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });

    conversionFormatCombo_ = new QComboBox;
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("onnx")), QStringLiteral("onnx"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("ncnn")), QStringLiteral("ncnn"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tiny_detector_json")), QStringLiteral("tiny_detector_json"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tensorrt")), QStringLiteral("tensorrt"));

    conversionOutputEdit_ = new QLineEdit;
    conversionOutputEdit_->setPlaceholderText(QStringLiteral("留空则输出到项目 models/exported；未打开项目时使用输入同目录"));
    auto* chooseOutputButton = new QPushButton(QStringLiteral("选择输出"));
    connect(chooseOutputButton, &QPushButton::clicked, this, [this]() {
        const QString format = conversionFormatCombo_
            ? conversionFormatCombo_->currentData().toString()
            : QStringLiteral("onnx");
        const QString inputPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
        const QString defaultDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(inputPath).absolutePath();
        const QString selected = QFileDialog::getSaveFileName(
            this,
            uiText("选择导出路径"),
            QDir(defaultDir).filePath(defaultExportFileName(format)),
            exportFileFilter(format));
        if (!selected.isEmpty()) {
            conversionOutputEdit_->setText(QDir::toNativeSeparators(selected));
        }
    });
    auto* exportButton = primaryButton(QStringLiteral("开始导出"));
    connect(headerExportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);

    auto* inputRow = new QWidget;
    auto* inputLayout = new QHBoxLayout(inputRow);
    inputLayout->setContentsMargins(0, 0, 0, 0);
    inputLayout->setSpacing(8);
    inputLayout->addWidget(conversionCheckpointEdit_, 1);
    inputLayout->addWidget(chooseCheckpointButton);

    auto* outputRow = new QWidget;
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->setSpacing(8);
    outputLayout->addWidget(conversionOutputEdit_, 1);
    outputLayout->addWidget(chooseOutputButton);

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("模型输入"), inputRow);
    form->addRow(QStringLiteral("目标格式"), conversionFormatCombo_);
    form->addRow(QStringLiteral("输出路径"), outputRow);
    setupPanel->bodyLayout()->addLayout(form);

    auto* sourceHelp = emptyStateLabel(QStringLiteral("从“任务与产物”中选中 best.onnx、checkpoint 或官方导出目录后，可点击“用作导出输入”自动带入这里。"));
    setupPanel->bodyLayout()->addWidget(sourceHelp);

    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    actionLayout->addWidget(mutedLabel(QStringLiteral("导出任务会记录到任务历史，完成后可直接作为推理输入。")), 1);
    actionLayout->addWidget(exportButton);
    setupPanel->bodyLayout()->addWidget(actionStrip);
    setupPanel->bodyLayout()->addStretch();

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* matrixPanel = new InfoPanel(QStringLiteral("格式矩阵"));
    auto* matrixTable = new QTableWidget(4, 4);
    matrixTable->setWordWrap(true);
    matrixTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("输入")
        << QStringLiteral("产物")
        << QStringLiteral("状态"));
    configureTable(matrixTable);
    matrixTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    matrixTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    matrixTable->setMinimumHeight(170);
    matrixTable->verticalHeader()->setDefaultSectionSize(44);
    const QStringList formats = {
        QStringLiteral("onnx"),
        QStringLiteral("ncnn"),
        QStringLiteral("tiny_detector_json"),
        QStringLiteral("tensorrt")
    };
    for (int row = 0; row < formats.size(); ++row) {
        const QString format = formats.at(row);
        matrixTable->setItem(row, 0, new QTableWidgetItem(exportFormatLabel(format)));
        matrixTable->setItem(row, 1, new QTableWidgetItem(
            format == QStringLiteral("tiny_detector_json")
                ? QStringLiteral("tiny detector checkpoint")
                : uiText("checkpoint 或 ONNX")));
        matrixTable->setItem(row, 2, new QTableWidgetItem(defaultExportFileName(format)));
        matrixTable->setItem(row, 3, new QTableWidgetItem(exportFormatNote(format)));
    }
    matrixPanel->bodyLayout()->addWidget(matrixTable);

    auto* resultPanel = new InfoPanel(QStringLiteral("运行状态"));
    exportResultLabel_ = inlineStatusLabel(QStringLiteral("暂无导出任务。"));
    resultPanel->bodyLayout()->addWidget(exportResultLabel_);
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("ONNX 会写入 AITrain sidecar；NCNN 依赖本机 onnx2ncnn；TensorRT 在当前 GTX 1060 / SM 61 上保持 hardware-blocked。")));
    resultPanel->bodyLayout()->addStretch();

    rightLayout->addWidget(matrixPanel);
    rightLayout->addWidget(resultPanel, 1);

    mainSplitter->addWidget(setupPanel);
    mainSplitter->addWidget(rightStack);
    mainSplitter->setStretchFactor(0, 3);
    mainSplitter->setStretchFactor(1, 4);
    mainSplitter->setSizes(QList<int>() << 520 << 680);

    layout->addWidget(headerPanel);
    layout->addWidget(mainSplitter, 1);
    return page;
}

QWidget* MainWindow::buildInferencePage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("推理输入"));
    auto* inferForm = new QFormLayout;
    inferenceCheckpointEdit_ = new QLineEdit;
    inferenceImageEdit_ = new QLineEdit;
    inferenceOutputEdit_ = new QLineEdit;
    inferenceCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 ONNX / AITrain export / TensorRT engine"));
    inferenceImageEdit_->setPlaceholderText(QStringLiteral("选择验证图片"));
    inferenceOutputEdit_->setPlaceholderText(QStringLiteral("输出目录；留空则输出到模型同目录 inference"));
    auto* chooseModelButton = new QPushButton(QStringLiteral("选择模型文件"));
    auto* chooseImageButton = new QPushButton(QStringLiteral("选择图片"));
    auto* inferButton = primaryButton(QStringLiteral("开始推理"));
    connect(chooseModelButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型文件"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseImageButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择图片"), currentProjectPath_, QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceImageEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(inferButton, &QPushButton::clicked, this, &MainWindow::startInference);
    auto* modelRow = new QWidget;
    auto* modelLayout = new QHBoxLayout(modelRow);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    modelLayout->addWidget(inferenceCheckpointEdit_);
    modelLayout->addWidget(chooseModelButton);
    auto* imageRow = new QWidget;
    auto* imageLayout = new QHBoxLayout(imageRow);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->addWidget(inferenceImageEdit_);
    imageLayout->addWidget(chooseImageButton);
    inferForm->addRow(QStringLiteral("模型"), modelRow);
    inferForm->addRow(QStringLiteral("图片"), imageRow);
    inferForm->addRow(QStringLiteral("输出"), inferenceOutputEdit_);
    toolbar->bodyLayout()->addLayout(inferForm);
    toolbar->bodyLayout()->addWidget(mutedLabel(QStringLiteral("推理验证会根据 ONNX 模型族或 AITrain 导出信息选择 detection、segmentation 或 OCR Rec 后处理；完整 PaddleOCR System 推理通过官方工具链任务产物查看。")));
    toolbar->bodyLayout()->addWidget(inferButton);

    auto* preview = new QSplitter(Qt::Horizontal);
    auto* sourcePanel = new InfoPanel(QStringLiteral("输入预览"));
    sourcePanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("原图、视频帧或 OCR 输入图像。")));
    auto* resultPanel = new InfoPanel(QStringLiteral("结果预览"));
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("检测框、分割 mask、OCR 文本和耗时会显示在这里。")));
    inferenceResultLabel_ = mutedLabel(QStringLiteral("暂无推理结果。"));
    sourcePanel->bodyLayout()->addWidget(inferenceResultLabel_);
    inferenceOverlayLabel_ = new QLabel(QStringLiteral("暂无 overlay"));
    inferenceOverlayLabel_->setObjectName(QStringLiteral("MutedText"));
    inferenceOverlayLabel_->setAlignment(Qt::AlignCenter);
    inferenceOverlayLabel_->setMinimumHeight(240);
    inferenceOverlayLabel_->setFrameShape(QFrame::StyledPanel);
    resultPanel->bodyLayout()->addWidget(inferenceOverlayLabel_);
    preview->addWidget(sourcePanel);
    preview->addWidget(resultPanel);

    layout->addWidget(toolbar);
    layout->addWidget(preview, 1);
    return page;
}

QWidget* MainWindow::buildPluginsPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* refreshButton = primaryButton(QStringLiteral("重新扫描插件"));
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refreshPlugins);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("PLUGIN CAPABILITY MATRIX"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("插件能力矩阵"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("扫描模型、数据集、导出和推理扩展能力；插件仍只通过公共接口暴露能力。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(refreshButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* scanCaption = new QLabel(QStringLiteral("扫描"));
    scanCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* pathCaption = new QLabel(QStringLiteral("路径"));
    pathCaption->setObjectName(QStringLiteral("ExperimentMeta"));
    pluginConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("等待插件扫描。"));
    pluginConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    pluginSearchPathLabel_ = inlineStatusLabel(QStringLiteral("插件搜索路径：未初始化"));
    pluginSearchPathLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    headerGrid->addWidget(scanCaption, 0, 0);
    headerGrid->addWidget(pluginConsoleStatusLabel_, 0, 1);
    headerGrid->addWidget(pathCaption, 1, 0);
    headerGrid->addWidget(pluginSearchPathLabel_, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* pluginCountCard = createMetricCard(QStringLiteral("已加载插件"), QStringLiteral("0"), QStringLiteral("manifest 已加载"));
    pluginCountSummaryLabel_ = pluginCountCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* datasetFormatCard = createMetricCard(QStringLiteral("数据集格式"), QStringLiteral("0"), QStringLiteral("可识别 / 校验格式"));
    pluginDatasetFormatSummaryLabel_ = datasetFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* exportFormatCard = createMetricCard(QStringLiteral("导出格式"), QStringLiteral("0"), QStringLiteral("插件声明的导出目标"));
    pluginExportFormatSummaryLabel_ = exportFormatCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* gpuCard = createMetricCard(QStringLiteral("GPU 需求"), QStringLiteral("0"), QStringLiteral("声明需要 GPU 的插件"));
    pluginGpuSummaryLabel_ = gpuCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(pluginCountCard, 0, 0);
    summaryLayout->addWidget(datasetFormatCard, 0, 1);
    summaryLayout->addWidget(exportFormatCard, 0, 2);
    summaryLayout->addWidget(gpuCard, 0, 3);

    auto* tablePanel = new InfoPanel(QStringLiteral("已加载插件"));
    pluginTable_ = new QTableWidget(0, 7);
    pluginTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("ID")
        << QStringLiteral("名称")
        << QStringLiteral("版本")
        << QStringLiteral("任务")
        << QStringLiteral("数据集")
        << QStringLiteral("导出")
        << QStringLiteral("GPU"));
    configureTable(pluginTable_);
    pluginTable_->setWordWrap(true);
    pluginTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    pluginTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    pluginTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    pluginTable_->verticalHeader()->setDefaultSectionSize(42);
    tablePanel->bodyLayout()->addWidget(pluginTable_);

    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(tablePanel, 1);
    updatePluginSummary();
    return page;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* runButton = primaryButton(QStringLiteral("执行环境自检"));
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runEnvironmentCheck);

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("RUNTIME HEALTH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("运行时健康面板"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("检查 NVIDIA 驱动、CUDA、TensorRT、ONNX Runtime、Qt 插件和 Worker 可用性。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(runButton);
    headerRoot->addLayout(headerTop);

    environmentConsoleStatusLabel_ = inlineStatusLabel(QStringLiteral("尚未执行环境自检。"));
    environmentConsoleStatusLabel_->setObjectName(QStringLiteral("DarkInlineStatus"));
    headerRoot->addWidget(environmentConsoleStatusLabel_);

    auto* summaryStrip = new QFrame;
    summaryStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summaryStrip);
    summaryLayout->setContentsMargins(12, 12, 12, 12);
    summaryLayout->setHorizontalSpacing(12);
    summaryLayout->setVerticalSpacing(12);
    auto* okCard = createMetricCard(QStringLiteral("通过"), QStringLiteral("0"), QStringLiteral("可用依赖"));
    environmentOkSummaryLabel_ = okCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* warningCard = createMetricCard(QStringLiteral("警告"), QStringLiteral("0"), QStringLiteral("可继续但需关注"));
    environmentWarningSummaryLabel_ = warningCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* missingCard = createMetricCard(QStringLiteral("缺失"), QStringLiteral("0"), QStringLiteral("会阻塞相关能力"));
    environmentMissingSummaryLabel_ = missingCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    auto* uncheckedCard = createMetricCard(QStringLiteral("未检测"), QStringLiteral("0"), QStringLiteral("等待 Worker 自检"));
    environmentUncheckedSummaryLabel_ = uncheckedCard->findChild<QLabel*>(QStringLiteral("MetricValue"));
    summaryLayout->addWidget(okCard, 0, 0);
    summaryLayout->addWidget(warningCard, 0, 1);
    summaryLayout->addWidget(missingCard, 0, 2);
    summaryLayout->addWidget(uncheckedCard, 0, 3);

    auto* panel = new InfoPanel(QStringLiteral("检查明细"));
    environmentTable_ = new QTableWidget(0, 3);
    environmentTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("检查项") << QStringLiteral("状态") << QStringLiteral("说明"));
    configureTable(environmentTable_);
    environmentTable_->setWordWrap(true);
    environmentTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    environmentTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    environmentTable_->verticalHeader()->setDefaultSectionSize(42);
    const QStringList rows = {
        QStringLiteral("NVIDIA Driver"),
        QStringLiteral("CUDA Runtime"),
        QStringLiteral("cuDNN"),
        QStringLiteral("TensorRT"),
        QStringLiteral("ONNX Runtime"),
        QStringLiteral("LibTorch"),
        QStringLiteral("Qt Plugins"),
        QStringLiteral("AITrain Plugins"),
        QStringLiteral("Worker")
    };
    for (const QString& rowName : rows) {
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(rowName));
        environmentTable_->setItem(row, 1, new QTableWidgetItem(uiText("未检测")));
        environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("点击执行环境自检。")));
    }
    panel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("TensorRT 真机验收仍需要 RTX / SM 75+。当前 GTX 1060 / SM 61 只能记录为 hardware-blocked。")));
    panel->bodyLayout()->addWidget(environmentTable_);
    layout->addWidget(headerPanel);
    layout->addWidget(summaryStrip);
    layout->addWidget(panel);
    updateEnvironmentSummary();
    return page;
}

InfoPanel* MainWindow::createMetricCard(const QString& label, const QString& value, const QString& caption)
{
    auto* panel = new InfoPanel(label);
    auto* valueLabel = new QLabel(value);
    valueLabel->setObjectName(QStringLiteral("MetricValue"));
    auto* captionLabel = new QLabel(caption);
    captionLabel->setObjectName(QStringLiteral("MetricLabel"));
    captionLabel->setWordWrap(true);
    panel->bodyLayout()->addWidget(valueLabel);
    panel->bodyLayout()->addWidget(captionLabel);
    return panel;
}

QString MainWindow::pageCaption(int pageIndex) const
{
    switch (pageIndex) {
    case DashboardPage: return tr("本机项目、数据、训练、模型交付状态总览");
    case ProjectPage: return tr("创建或打开本地训练项目，统一管理数据、运行和模型产物");
    case DatasetPage: return tr("管理数据集库，完成导入、校验、划分和样本预览");
    case TrainingPage: return tr("启动官方后端优先的训练实验，并监控指标、日志和产物");
    case TaskQueuePage: return tr("追踪历史任务、指标、导出记录和所有 Worker 产物");
    case ModelRegistryPage: return tr("管理模型版本、来源 lineage，以及导出、推理和流水线入口");
    case EvaluationReportsPage: return tr("集中查看最近评估报告、任务类型、报告路径和详细可视化结果");
    case ConversionPage: return tr("将训练产物导出为 ONNX 或外部 TensorRT 验收目标");
    case InferencePage: return tr("选择模型与样本图，验证 detection、segmentation 或 OCR 推理结果");
    case PluginsPage: return tr("扫描和诊断模型插件");
    case EnvironmentPage: return tr("检查 GPU、CUDA、TensorRT 和运行时依赖");
    default: return {};
    }
}

void MainWindow::showPage(int pageIndex, const QString& title)
{
    stack_->setCurrentIndex(pageIndex);
    pageTitle_->setText(title);
    pageCaption_->setText(pageCaption(pageIndex));
    sidebar_->setCurrentIndex(pageIndex);
    if (pageIndex == ModelRegistryPage) {
        updateModelRegistry();
    }
    if (pageIndex == EvaluationReportsPage) {
        updateModelRegistry();
    }
}

void MainWindow::openEvaluationReportsPage()
{
    showPage(EvaluationReportsPage, uiText("评估报告"));
}

void MainWindow::createProject()
{
    currentProjectName_ = projectNameEdit_->text().trimmed();
    currentProjectPath_ = QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
    if (currentProjectName_.isEmpty() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, uiText("项目"), uiText("项目名称和目录不能为空。"));
        return;
    }

    ensureProjectSubdirs(currentProjectPath_);
    QString error;
    if (!repository_.open(QDir(currentProjectPath_).filePath(QStringLiteral("project.sqlite")), &error)
        || !repository_.upsertProject(currentProjectName_, currentProjectPath_, &error)) {
        QMessageBox::critical(this, uiText("项目"), error);
        return;
    }
    repository_.markInterruptedTasksFailed(uiText("上次会话结束时任务未正常完成，已标记为失败。"), &error);

    projectLabel_->setText(uiText("当前项目：%1").arg(currentProjectPath_));
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(currentProjectName_);
    }
    updateHeaderState();
    updateRecentTasks();
    updateDatasetList();
    updateModelRegistry();
    updateDashboardSummary();
    refreshTrainingDefaults();
    statusBar()->showMessage(uiText("项目已打开：%1").arg(currentProjectName_), 5000);
}

void MainWindow::browseDataset()
{
    const QString directory = QFileDialog::getExistingDirectory(this, uiText("选择数据集目录"));
    if (!directory.isEmpty()) {
        datasetPathEdit_->setText(QDir::toNativeSeparators(directory));
        const QString detectedFormat = detectDatasetFormatFromPath(directory);
        if (!detectedFormat.isEmpty() && datasetFormatCombo_) {
            const int index = datasetFormatCombo_->findData(detectedFormat);
            if (index >= 0) {
                datasetFormatCombo_->setCurrentIndex(index);
            }
        }
        if (splitOutputEdit_ && currentProjectPath_.isEmpty()) {
            splitOutputEdit_->setText(QDir::toNativeSeparators(QDir(directory).absoluteFilePath(QStringLiteral("../normalized"))));
        }
        currentDatasetPath_ = directory;
        const QString selectedFormat = currentDatasetFormat();
        currentDatasetFormat_ = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
        currentDatasetValid_ = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
    }
}

void MainWindow::validateDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集校验"), uiText("Worker 正在执行任务，稍后再校验数据集。"));
        return;
    }

    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (format.isEmpty() || path.isEmpty()) {
        validationSummaryLabel_->setText(uiText("请选择数据集目录和格式。"));
        return;
    }

    currentDatasetValid_ = false;
    currentDatasetPath_ = path;
    currentDatasetFormat_ = format;
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
    }
    validationSummaryLabel_->setText(uiText("正在通过 Worker 校验数据集。"));
    validationOutput_->setPlainText(uiText("等待校验结果。"));

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("maxFiles"), 5000);
    options.insert(QStringLiteral("allowEmptyLabels"), false);
    options.insert(QStringLiteral("maxTextLength"), 25);

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_validation"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集校验中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestDatasetValidation(workerExecutablePath(), path, format, options, &error, taskId, outputPath)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        validationSummaryLabel_->setText(uiText("无法启动数据集校验：%1").arg(error));
        QMessageBox::critical(this, uiText("数据集校验"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集校验中"), StatusPill::Tone::Info);
}

void MainWindow::splitDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("Worker 正在执行任务，稍后再划分数据集。"));
        return;
    }

    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_->text());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("请先选择数据集目录和格式。"));
        return;
    }
    if (format != QStringLiteral("yolo_detection") && format != QStringLiteral("yolo_txt")
        && format != QStringLiteral("yolo_segmentation")
        && format != QStringLiteral("paddleocr_det")
        && format != QStringLiteral("paddleocr_rec")) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("当前划分支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec 格式。"));
        return;
    }

    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == path && currentDatasetFormat_ == format;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(path, &error);
        datasetReady = dataset.rootPath == path
            && dataset.format == format
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, uiText("数据集划分"), uiText("请先通过当前格式的数据集校验。"));
        return;
    }

    QString outputPath = QDir::fromNativeSeparators(splitOutputEdit_->text().trimmed());
    if (outputPath.isEmpty()) {
        const QString datasetName = QFileInfo(path).fileName();
        const QString basePath = currentProjectPath_.isEmpty()
            ? QDir(path).absoluteFilePath(QStringLiteral("../normalized"))
            : QDir(currentProjectPath_).filePath(QStringLiteral("datasets/normalized/%1").arg(datasetName));
        outputPath = QDir::cleanPath(basePath);
        splitOutputEdit_->setText(QDir::toNativeSeparators(outputPath));
    }

    QJsonObject options;
    options.insert(QStringLiteral("trainRatio"), splitTrainRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("valRatio"), splitValRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("testRatio"), splitTestRatioEdit_->text().toDouble());
    options.insert(QStringLiteral("seed"), splitSeedEdit_->text().toInt());
    options.insert(QStringLiteral("maxIssues"), 200);
    options.insert(QStringLiteral("allowEmptyLabels"), false);

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Validate,
            QStringLiteral("dataset_split"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集划分中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestDatasetSplit(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集划分"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集划分中"), StatusPill::Tone::Info);
    statusBar()->showMessage(uiText("正在划分数据集"), 3000);
}

void MainWindow::curateDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("Worker 正在执行任务，稍后再生成数据质量报告。"));
        return;
    }
    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据质量报告"), uiText("请先选择数据集目录和格式。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Curate,
            QStringLiteral("dataset_quality"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据质量报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("maxIssues"), 500);
    options.insert(QStringLiteral("maxProblemSamples"), 500);
    options.insert(QStringLiteral("maxFiles"), 20000);
    options.insert(QStringLiteral("duplicateHashLimit"), 20000);
    options.insert(QStringLiteral("distributionWarningThreshold"), 0.25);
    options.insert(QStringLiteral("exportXAnyLabelingFixList"), true);

    QString error;
    if (!worker_.requestDatasetCuration(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据质量报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据质量报告生成中"), StatusPill::Tone::Info);
}

void MainWindow::openDatasetQualityFixList()
{
    if (latestQualityFixListPath_.isEmpty() || !QFileInfo::exists(latestQualityFixListPath_)) {
        QMessageBox::information(this, uiText("问题清单"), uiText("请先生成数据质量报告。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(latestQualityFixListPath_));
}

void MainWindow::launchXAnyLabelingForQualityFix()
{
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (datasetPath.isEmpty()) {
        QMessageBox::information(this, uiText("X-AnyLabeling 修复"), uiText("请先选择数据集目录。"));
        return;
    }
    if (!latestQualityFixListPath_.isEmpty()) {
        statusBar()->showMessage(uiText("问题清单：%1").arg(QDir::toNativeSeparators(latestQualityFixListPath_)), 6000);
    }
    const QString program = resolvedXAnyLabelingProgram();
    if (program.isEmpty()) {
        updateAnnotationToolStatus();
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
        return;
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    if (QProcess::startDetached(program, QStringList() << datasetPath)) {
        statusBar()->showMessage(uiText("已启动 X-AnyLabeling，请按问题清单修复样本。"), 5000);
    } else {
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
    }
}

void MainWindow::createDatasetSnapshot()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("数据集快照"), uiText("Worker 正在执行任务，稍后再创建数据集快照。"));
        return;
    }
    const QString format = currentDatasetFormat();
    const QString path = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (path.isEmpty() || format.isEmpty()) {
        QMessageBox::warning(this, uiText("数据集快照"), uiText("请先选择数据集目录和格式。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Snapshot,
            QStringLiteral("dataset_snapshot"),
            QStringLiteral("com.aitrain.plugins.dataset_interop"),
            outputPath,
            uiText("数据集快照创建中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);

    QString error;
    if (!worker_.requestDatasetSnapshot(workerExecutablePath(), path, outputPath, format, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("数据集快照"), error);
        return;
    }
    workerPill_->setStatus(uiText("数据集快照创建中"), StatusPill::Tone::Info);
}

void MainWindow::startModelExport()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("模型导出"), uiText("Worker 正在执行任务，稍后再导出模型。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty()) {
        QMessageBox::warning(this, uiText("模型导出"), uiText("请选择模型输入。"));
        return;
    }
    const QString format = conversionFormatCombo_
        ? conversionFormatCombo_->currentData().toString()
        : QStringLiteral("tiny_detector_json");
    QString outputPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (outputPath.isEmpty()) {
        const QString outputDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(checkpointPath).absoluteDir().absolutePath();
        QDir().mkpath(outputDir);
        outputPath = QDir(outputDir).filePath(defaultExportFileName(format));
    }

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Export,
            QStringLiteral("model_export"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            QFileInfo(outputPath).absolutePath(),
            uiText("模型导出中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestModelExport(workerExecutablePath(), checkpointPath, outputPath, format, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("模型导出"), error);
        return;
    }
    if (exportResultLabel_) {
        exportResultLabel_->setText(uiText("正在导出：%1").arg(QDir::toNativeSeparators(outputPath)));
    }
    workerPill_->setStatus(uiText("模型导出中"), StatusPill::Tone::Info);
}

void MainWindow::startInference()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("推理"), uiText("Worker 正在执行任务，稍后再推理。"));
        return;
    }
    const QString checkpointPath = QDir::fromNativeSeparators(inferenceCheckpointEdit_ ? inferenceCheckpointEdit_->text().trimmed() : QString());
    const QString imagePath = QDir::fromNativeSeparators(inferenceImageEdit_ ? inferenceImageEdit_->text().trimmed() : QString());
    QString outputPath = QDir::fromNativeSeparators(inferenceOutputEdit_ ? inferenceOutputEdit_->text().trimmed() : QString());
    if (checkpointPath.isEmpty() || imagePath.isEmpty()) {
        QMessageBox::warning(this, uiText("推理"), uiText("请选择模型文件和图片。"));
        return;
    }
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(checkpointPath).absoluteDir().filePath(QStringLiteral("inference"));
    }

    QString taskId;
    if (repository_.isOpen()) {
        taskId = createRepositoryTask(
            aitrain::TaskKind::Infer,
            QStringLiteral("inference"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            outputPath,
            uiText("推理中。"));
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestInference(workerExecutablePath(), checkpointPath, imagePath, outputPath, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("推理"), error);
        return;
    }
    if (inferenceResultLabel_) {
        inferenceResultLabel_->setText(uiText("正在推理：%1").arg(QDir::toNativeSeparators(imagePath)));
    }
    workerPill_->setStatus(uiText("推理中"), StatusPill::Tone::Info);
}

void MainWindow::startTraining()
{
    if (currentProjectPath_.isEmpty()) {
        createProject();
        if (currentProjectPath_.isEmpty()) {
            return;
        }
    }
    if (pluginCombo_->currentData().toString().isEmpty() || currentTaskType().isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请选择可用插件和任务类型。"));
        return;
    }
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_->text());
    const QString datasetFormat = currentDatasetFormat();
    if (datasetPath.isEmpty() || datasetFormat.isEmpty()) {
        QMessageBox::warning(this, uiText("训练"), uiText("请先选择并校验数据集。"));
        return;
    }
    auto* selectedPlugin = pluginManager_.pluginById(pluginCombo_->currentData().toString());
    if (!selectedPlugin || !selectedPlugin->datasetAdapter(datasetFormat)) {
        QMessageBox::warning(this, uiText("训练"), uiText("当前训练插件不支持所选数据集格式。"));
        return;
    }
    bool datasetReady = currentDatasetValid_ && currentDatasetPath_ == datasetPath && currentDatasetFormat_ == datasetFormat;
    if (!datasetReady && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        datasetReady = dataset.rootPath == datasetPath
            && dataset.format == datasetFormat
            && dataset.validationStatus == QStringLiteral("valid");
    }
    if (!datasetReady) {
        QMessageBox::warning(this, uiText("训练"), uiText("数据集未通过当前格式校验，不能启动训练。"));
        return;
    }

    const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
    QDir().mkpath(runDir);

    QJsonObject parameters;
    parameters.insert(QStringLiteral("epochs"), epochsEdit_->text().toInt());
    parameters.insert(QStringLiteral("batchSize"), batchEdit_->text().toInt());
    parameters.insert(QStringLiteral("imageSize"), imageSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("gridSize"), gridSizeEdit_->text().toInt());
    parameters.insert(QStringLiteral("seed"), 42);
    parameters.insert(QStringLiteral("resumeCheckpointPath"), QDir::fromNativeSeparators(resumeCheckpointEdit_->text().trimmed()));
    parameters.insert(QStringLiteral("horizontalFlip"), horizontalFlipCheck_ && horizontalFlipCheck_->isChecked());
    parameters.insert(QStringLiteral("colorJitter"), colorJitterCheck_ && colorJitterCheck_->isChecked());
    const QString trainingBackend = trainingBackendCombo_
        ? trainingBackendCombo_->currentData().toString().trimmed()
        : defaultBackendForTask(currentTaskType());
    const QString backendForRequest = trainingBackend.isEmpty() ? defaultBackendForTask(currentTaskType()) : trainingBackend;
    const QString modelPreset = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
    parameters.insert(QStringLiteral("trainingBackend"), backendForRequest);
    if (!modelPreset.isEmpty()) {
        parameters.insert(QStringLiteral("modelPreset"), modelPreset);
        if (backendForRequest.startsWith(QStringLiteral("ultralytics_yolo"))) {
            parameters.insert(QStringLiteral("model"), modelPreset);
        }
    }
    aitrain::TrainingRequest request;
    request.taskId = taskId;
    request.projectPath = currentProjectPath_;
    request.pluginId = pluginCombo_->currentData().toString();
    request.taskType = currentTaskType();
    request.datasetPath = datasetPath;
    request.outputPath = runDir;
    request.parameters = parameters;

    int datasetId = 0;
    bool needsSnapshot = true;
    if (repository_.isOpen()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &snapshotError);
        datasetId = dataset.id;
        needsSnapshot = !attachLatestSnapshotToRequest(request, datasetId, &snapshotError);
    }

    aitrain::TaskRecord record;
    record.id = taskId;
    record.projectName = currentProjectName_;
    record.pluginId = request.pluginId;
    record.taskType = request.taskType;
    record.kind = aitrain::TaskKind::Train;
    record.state = aitrain::TaskState::Queued;
    record.workDir = runDir;
    record.message = needsSnapshot
        ? uiText("等待自动创建数据快照。")
        : (worker_.isRunning() ? uiText("等待当前任务完成。") : uiText("等待 Worker 启动。"));
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    QString error;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return;
    }

    if (!needsSnapshot) {
        recordExperimentRunForRequest(request, datasetId, &error);
    }

    PendingTrainingTask pending{taskId, request, needsSnapshot, datasetId, datasetFormat};
    if (worker_.isRunning() || needsSnapshot) {
        pendingTrainingTasks_.append(pending);
        workerPill_->setStatus(uiText("任务已排队"), StatusPill::Tone::Info);
        appendLog(uiText("任务已加入队列：%1").arg(taskId));
        updateRecentTasks();
        startNextQueuedTask();
        return;
    }

    startQueuedTraining(taskId, request);
}

void MainWindow::cancelSelectedTask()
{
    if (!taskQueueTable_ || !repository_.isOpen()) {
        return;
    }

    const int row = taskQueueTable_->currentRow();
    if (row < 0 || !taskQueueTable_->item(row, 0)) {
        QMessageBox::information(this, uiText("任务队列"), uiText("请先选择一个任务。"));
        return;
    }

    const QString taskId = taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString();
    if (taskId.isEmpty()) {
        return;
    }

    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(200, &error);
    for (const aitrain::TaskRecord& task : tasks) {
        if (task.id != taskId) {
            continue;
        }

        if (task.state == aitrain::TaskState::Queued) {
            for (int index = 0; index < pendingTrainingTasks_.size(); ++index) {
                if (pendingTrainingTasks_.at(index).taskId == taskId) {
                    pendingTrainingTasks_.remove(index);
                    break;
                }
            }
            if (!repository_.updateTaskState(taskId, aitrain::TaskState::Canceled, uiText("用户取消排队任务。"), &error)) {
                QMessageBox::warning(this, uiText("任务队列"), error);
            }
            updateRecentTasks();
            return;
        }

        if ((task.state == aitrain::TaskState::Running || task.state == aitrain::TaskState::Paused) && taskId == currentTaskId_) {
            worker_.cancel();
            return;
        }

        QMessageBox::information(this, uiText("任务队列"), uiText("只能取消排队任务或当前 Worker 正在运行的任务。"));
        return;
    }
}

void MainWindow::reproduceSelectedTrainingTask()
{
    if (!repository_.isOpen()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先打开项目。"));
        return;
    }
    if (currentProjectPath_.isEmpty()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先打开或创建项目。"));
        return;
    }

    const QString sourceTaskId = selectedTaskId();
    if (sourceTaskId.isEmpty()) {
        QMessageBox::information(this, uiText("复现实验"), uiText("请先选择一个训练任务。"));
        return;
    }

    QString error;
    aitrain::TaskRecord sourceTask;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(500, &error);
    for (const aitrain::TaskRecord& task : tasks) {
        if (task.id == sourceTaskId) {
            sourceTask = task;
            break;
        }
    }
    if (sourceTask.id.isEmpty() || sourceTask.kind != aitrain::TaskKind::Train) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("只能复现历史训练任务。"));
        return;
    }

    const aitrain::ExperimentRunRecord sourceRun = repository_.experimentRunForTask(sourceTaskId, &error);
    if (sourceRun.id <= 0 || sourceRun.requestJson.trimmed().isEmpty()) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("该训练任务没有可复现的 request 记录。"));
        return;
    }

    QJsonParseError parseError;
    const QJsonDocument requestDoc = QJsonDocument::fromJson(sourceRun.requestJson.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !requestDoc.isObject()) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("原训练 request JSON 无法解析：%1").arg(parseError.errorString()));
        return;
    }

    aitrain::TrainingRequest request = aitrain::TrainingRequest::fromJson(requestDoc.object());
    const int snapshotId = sourceRun.datasetSnapshotId > 0
        ? sourceRun.datasetSnapshotId
        : request.parameters.value(QStringLiteral("datasetSnapshotId")).toInt();
    const aitrain::DatasetSnapshotRecord snapshot = repository_.datasetSnapshotById(snapshotId, &error);
    if (snapshot.id <= 0 || snapshot.manifestPath.isEmpty() || !QFileInfo::exists(snapshot.manifestPath)) {
        QMessageBox::warning(this, uiText("复现实验"), uiText("原实验的数据快照 manifest 缺失，无法按同一快照复现。请重新创建快照或选择其他训练任务。"));
        return;
    }

    const QString newTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString runDir = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(newTaskId));
    QDir().mkpath(runDir);

    request.taskId = newTaskId;
    request.projectPath = currentProjectPath_;
    request.outputPath = runDir;
    if (request.pluginId.isEmpty()) {
        request.pluginId = sourceTask.pluginId;
    }
    if (request.taskType.isEmpty()) {
        request.taskType = sourceTask.taskType;
    }
    request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
    request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
    request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
    if (!request.parameters.contains(QStringLiteral("seed"))) {
        request.parameters.insert(QStringLiteral("seed"), 42);
    }
    request.parameters.insert(QStringLiteral("reproducedFromTaskId"), sourceTaskId);
    request.parameters.insert(QStringLiteral("reproducedFromExperimentRunId"), sourceRun.id);
    request.parameters.insert(QStringLiteral("reproduceMode"), QStringLiteral("same_snapshot_same_params"));

    aitrain::TaskRecord record;
    record.id = newTaskId;
    record.projectName = currentProjectName_;
    record.pluginId = request.pluginId;
    record.taskType = request.taskType;
    record.kind = aitrain::TaskKind::Train;
    record.state = aitrain::TaskState::Queued;
    record.workDir = runDir;
    record.message = worker_.isRunning() ? uiText("复现实验已排队。") : uiText("复现实验等待 Worker 启动。");
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("复现实验"), error);
        return;
    }
    recordExperimentRunForRequest(request, snapshot.datasetId, &error);

    PendingTrainingTask pending{newTaskId, request, false, snapshot.datasetId, QString()};
    if (worker_.isRunning()) {
        pendingTrainingTasks_.append(pending);
        workerPill_->setStatus(uiText("复现实验已排队"), StatusPill::Tone::Info);
        updateRecentTasks();
        return;
    }

    updateRecentTasks();
    startQueuedTraining(newTaskId, request);
}

void MainWindow::runEnvironmentCheck()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("环境自检"), uiText("Worker 正在执行任务，稍后再运行环境自检。"));
        return;
    }

    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            auto* statusItem = new QTableWidgetItem(uiText("检测中"));
            environmentTable_->setItem(row, 1, statusItem);
            environmentTable_->setItem(row, 2, new QTableWidgetItem(uiText("等待 Worker 返回结果。")));
        }
    }
    updateEnvironmentSummary();

    QString error;
    if (!worker_.requestEnvironmentCheck(workerExecutablePath(), &error)) {
        QMessageBox::critical(this, uiText("环境自检"), error);
        return;
    }
    workerPill_->setStatus(uiText("环境自检中"), StatusPill::Tone::Info);
}

void MainWindow::handleWorkerMessage(const QString& type, const QJsonObject& payload)
{
    if (type == QStringLiteral("progress")) {
        if (!currentTaskId_.isEmpty()) {
            progressBar_->setValue(payload.value(QStringLiteral("percent")).toInt());
        }
        const QString message = payload.value(QStringLiteral("message")).toString();
        if (!message.isEmpty()) {
            statusBar()->showMessage(message, 3000);
        }
    } else if (type == QStringLiteral("metric")) {
        const QString name = payload.value(QStringLiteral("name")).toString();
        const double value = payload.value(QStringLiteral("value")).toDouble();
        metricsWidget_->addMetric(name, value);

        aitrain::MetricPoint point;
        point.taskId = currentTaskId_;
        point.name = name;
        point.value = value;
        point.step = payload.value(QStringLiteral("step")).toInt();
        point.epoch = payload.value(QStringLiteral("epoch")).toInt();
        point.createdAt = QDateTime::currentDateTimeUtc();
        QString error;
        repository_.insertMetric(point, &error);
    } else if (type == QStringLiteral("artifact")) {
        const QString path = payload.value(QStringLiteral("path")).toString();
        const QString kind = payload.value(QStringLiteral("kind")).toString();
        appendLog(uiText("产物：%1").arg(path));
        if (kind == QStringLiteral("checkpoint") && latestCheckpointLabel_) {
            latestCheckpointLabel_->setText(uiText("最新 checkpoint：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("preview") && latestPreviewPathLabel_) {
            latestPreviewPathLabel_->setText(uiText("最新预览：%1").arg(QDir::toNativeSeparators(path)));
            if (latestPreviewImageLabel_) {
                QPixmap preview(path);
                if (!preview.isNull()) {
                    latestPreviewImageLabel_->setPixmap(preview.scaled(
                        latestPreviewImageLabel_->size().boundedTo(QSize(320, 220)),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation));
                } else {
                    latestPreviewImageLabel_->setText(uiText("预览图加载失败"));
                }
            }
        }
        if (kind == QStringLiteral("export") && exportResultLabel_) {
            exportResultLabel_->setText(uiText("导出完成：%1").arg(QDir::toNativeSeparators(path)));
        } else if (kind == QStringLiteral("inference_overlay") && inferenceOverlayLabel_) {
            QPixmap overlay(path);
            if (!overlay.isNull()) {
                inferenceOverlayLabel_->setPixmap(overlay.scaled(
                    inferenceOverlayLabel_->size().boundedTo(QSize(520, 360)),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation));
            } else {
                inferenceOverlayLabel_->setText(uiText("推理 overlay 加载失败"));
            }
        } else if (kind == QStringLiteral("inference_predictions") && inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(path));
        }
        if (repository_.isOpen()) {
            aitrain::ArtifactRecord artifact;
            artifact.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            artifact.kind = kind;
            artifact.path = path;
            artifact.message = uiText("Worker 上报产物");
            artifact.createdAt = QDateTime::currentDateTimeUtc();
            QString error;
            repository_.insertArtifact(artifact, &error);
        }
    } else if (type == QStringLiteral("paused")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Paused, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(uiText("任务已暂停"), StatusPill::Tone::Warning);
        updateRecentTasks();
    } else if (type == QStringLiteral("resumed")) {
        QString error;
        repository_.updateTaskState(currentTaskId_, aitrain::TaskState::Running, payload.value(QStringLiteral("message")).toString(), &error);
        workerPill_->setStatus(uiText("训练运行中"), StatusPill::Tone::Info);
        updateRecentTasks();
    } else if (type == QStringLiteral("canceled")) {
        QString error;
        const QString canceledTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
        const QString canceledMessage = payload.value(QStringLiteral("message")).toString();
        repository_.updateTaskState(canceledTaskId, aitrain::TaskState::Canceled, canceledMessage, &error);
        if (hasActiveSnapshotTrainingTask_ && canceledTaskId == currentTaskId_) {
            repository_.updateTaskState(
                activeSnapshotTrainingTask_.taskId,
                aitrain::TaskState::Canceled,
                uiText("自动数据快照已取消，训练未启动。"),
                &error);
            hasActiveSnapshotTrainingTask_ = false;
            activeSnapshotTrainingTask_ = PendingTrainingTask();
        }
        workerPill_->setStatus(uiText("任务已取消"), StatusPill::Tone::Warning);
        appendLog(uiText("任务已取消：%1").arg(canceledMessage));
        currentTaskId_.clear();
        updateRecentTasks();
        startNextQueuedTask();
    } else if (type == QStringLiteral("failed")) {
        const QString failedTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
        const QString failedMessage = payload.value(QStringLiteral("message")).toString();
        QString error;
        repository_.updateTaskState(
            failedTaskId,
            aitrain::TaskState::Failed,
            failedMessage,
            &error);
        if (hasActiveSnapshotTrainingTask_ && failedTaskId == currentTaskId_) {
            repository_.updateTaskState(
                activeSnapshotTrainingTask_.taskId,
                aitrain::TaskState::Failed,
                uiText("自动数据快照失败：%1").arg(failedMessage),
                &error);
            hasActiveSnapshotTrainingTask_ = false;
            activeSnapshotTrainingTask_ = PendingTrainingTask();
        }
        updateRecentTasks();
        updateModelRegistry();
    } else if (type == QStringLiteral("environmentCheck")) {
        updateEnvironmentTable(payload);
    } else if (type == QStringLiteral("datasetValidation")) {
        updateDatasetValidationResult(payload);
    } else if (type == QStringLiteral("datasetSplit")) {
        updateDatasetSplitResult(payload);
    } else if (type == QStringLiteral("datasetQuality")) {
        latestQualityFixListPath_ = payload.value(QStringLiteral("xAnyLabelingFixListPath")).toString();
        if (validationSummaryLabel_) {
            const QJsonObject severityCounts = payload.value(QStringLiteral("severityCounts")).toObject();
            const QJsonObject summary = payload.value(QStringLiteral("summary")).toObject();
            validationSummaryLabel_->setText(uiText("质量报告完成：error %1 / warning %2 / info %3，问题样本 %4，重复图片 %5。")
                .arg(severityCounts.value(QStringLiteral("error")).toInt())
                .arg(severityCounts.value(QStringLiteral("warning")).toInt())
                .arg(severityCounts.value(QStringLiteral("info")).toInt())
                .arg(summary.value(QStringLiteral("problemSampleCount")).toInt())
                .arg(summary.value(QStringLiteral("duplicateImageCount")).toInt()));
        }
        if (validationOutput_) {
            validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
        }
        if (validationIssuesTable_) {
            validationIssuesTable_->setRowCount(0);
            const QJsonArray samples = !payload.value(QStringLiteral("problemSamples")).toArray().isEmpty()
                ? payload.value(QStringLiteral("problemSamples")).toArray()
                : payload.value(QStringLiteral("issues")).toArray();
            if (samples.isEmpty()) {
                validationIssuesTable_->insertRow(0);
                validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
                validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("ok")));
                validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString()));
                validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
                validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("未发现需要修复的问题样本。")));
            } else {
                for (const QJsonValue& value : samples) {
                    const QJsonObject issue = value.toObject();
                    const int row = validationIssuesTable_->rowCount();
                    validationIssuesTable_->insertRow(row);
                    validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(issueSeverityLabel(issue.value(QStringLiteral("severity")).toString())));
                    validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(issue.value(QStringLiteral("code")).toString()));
                    const QString issuePath = !issue.value(QStringLiteral("imagePath")).toString().isEmpty()
                        ? issue.value(QStringLiteral("imagePath")).toString()
                        : (!issue.value(QStringLiteral("labelPath")).toString().isEmpty()
                            ? issue.value(QStringLiteral("labelPath")).toString()
                            : issue.value(QStringLiteral("filePath")).toString());
                    validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(issuePath));
                    const int line = issue.value(QStringLiteral("line")).toInt();
                    validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(line > 0 ? QString::number(line) : QString()));
                    validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(issue.value(QStringLiteral("message")).toString()));
                }
            }
        }
        if (datasetDetailLabel_) {
            datasetDetailLabel_->setText(uiText("修复清单：%1")
                .arg(latestQualityFixListPath_.isEmpty() ? uiText("暂无") : QDir::toNativeSeparators(latestQualityFixListPath_)));
        }
        const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
        const QString format = payload.value(QStringLiteral("format")).toString();
        if (!datasetPath.isEmpty()) {
            currentDatasetPath_ = datasetPath;
            currentDatasetFormat_ = format;
            currentDatasetValid_ = payload.value(QStringLiteral("ok")).toBool();
        }
        if (repository_.isOpen() && !datasetPath.isEmpty()) {
            const QJsonObject summary = payload.value(QStringLiteral("summary")).toObject();
            aitrain::DatasetRecord dataset;
            dataset.name = QFileInfo(datasetPath).fileName();
            dataset.format = format;
            dataset.rootPath = datasetPath;
            dataset.validationStatus = payload.value(QStringLiteral("ok")).toBool() ? QStringLiteral("valid") : QStringLiteral("invalid");
            dataset.sampleCount = summary.value(QStringLiteral("sampleCount")).toInt();
            dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
            dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
            QString error;
            repository_.upsertDatasetValidation(dataset, &error);
            updateDatasetList();
        }
    } else if (type == QStringLiteral("datasetSnapshot")) {
        if (validationSummaryLabel_) {
            validationSummaryLabel_->setText(uiText("数据集快照完成：%1 个文件，hash %2。")
                .arg(payload.value(QStringLiteral("fileCount")).toInt())
                .arg(payload.value(QStringLiteral("contentHash")).toString().left(12)));
        }
        if (validationOutput_) {
            validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
        }
        if (repository_.isOpen()) {
            QString error;
            const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
            aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
            if (dataset.id <= 0 && !datasetPath.isEmpty()) {
                aitrain::DatasetRecord seed;
                seed.name = QFileInfo(datasetPath).fileName();
                seed.format = payload.value(QStringLiteral("format")).toString(currentDatasetFormat_);
                seed.rootPath = datasetPath;
                seed.validationStatus = currentDatasetValid_ ? QStringLiteral("valid") : QStringLiteral("snapshot");
                seed.sampleCount = payload.value(QStringLiteral("fileCount")).toInt();
                seed.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
                seed.lastValidatedAt = QDateTime::currentDateTimeUtc();
                repository_.upsertDatasetValidation(seed, &error);
                dataset = repository_.datasetByRootPath(datasetPath, &error);
            }
            if (dataset.id > 0) {
                aitrain::DatasetSnapshotRecord snapshot;
                snapshot.datasetId = dataset.id;
                snapshot.name = QFileInfo(datasetPath).fileName();
                snapshot.rootPath = datasetPath;
                snapshot.manifestPath = payload.value(QStringLiteral("manifestPath")).toString();
                snapshot.contentHash = payload.value(QStringLiteral("contentHash")).toString();
                snapshot.fileCount = payload.value(QStringLiteral("fileCount")).toInt();
                snapshot.totalBytes = payload.value(QStringLiteral("totalBytes")).toVariant().toLongLong();
                snapshot.metadataJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
                snapshot.createdAt = QDateTime::currentDateTimeUtc();
                const int snapshotId = repository_.insertDatasetSnapshot(snapshot, &error);
                if (hasActiveSnapshotTrainingTask_
                    && activeSnapshotTrainingTask_.request.datasetPath == datasetPath
                    && snapshotId > 0) {
                    snapshot.id = snapshotId;
                    activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
                    activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
                    activeSnapshotTrainingTask_.request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
                    activeSnapshotTrainingTask_.needsSnapshot = false;
                    recordExperimentRunForRequest(activeSnapshotTrainingTask_.request, activeSnapshotTrainingTask_.datasetId, &error);
                    pendingTrainingTasks_.prepend(activeSnapshotTrainingTask_);
                    hasActiveSnapshotTrainingTask_ = false;
                    activeSnapshotTrainingTask_ = PendingTrainingTask();
                }
            }
        }
        updateTrainingSelectionSummary();
        updateDatasetList();
    } else if (type == QStringLiteral("evaluationReport")) {
        if (repository_.isOpen()) {
            aitrain::EvaluationReportRecord report;
            report.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            report.modelPath = payload.value(QStringLiteral("modelPath")).toString();
            report.taskType = payload.value(QStringLiteral("taskType")).toString();
            report.datasetSnapshotId = payload.value(QStringLiteral("datasetSnapshotId")).toInt();
            report.reportPath = payload.value(QStringLiteral("reportPath")).toString();
            report.summaryJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
            report.createdAt = QDateTime::currentDateTimeUtc();
            QString error;
            repository_.insertEvaluationReport(report, &error);
            updateModelRegistry();
        }
    } else if (type == QStringLiteral("benchmarkReport")) {
        updateModelRegistry();
    } else if (type == QStringLiteral("pipelinePlan")) {
        if (repository_.isOpen()) {
            aitrain::PipelineRunRecord pipeline;
            pipeline.name = uiText("本地闭环流水线");
            pipeline.templateId = payload.value(QStringLiteral("templateId")).toString();
            QJsonArray taskIds = payload.value(QStringLiteral("taskIds")).toArray();
            if (taskIds.isEmpty()) {
                const QString fallbackTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
                if (!fallbackTaskId.isEmpty()) {
                    taskIds.append(fallbackTaskId);
                }
            }
            pipeline.taskIdsJson = QString::fromUtf8(QJsonDocument(taskIds).toJson(QJsonDocument::Compact));
            pipeline.state = payload.value(QStringLiteral("state")).toString(QStringLiteral("planned"));
            pipeline.summaryJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
            pipeline.createdAt = QDateTime::currentDateTimeUtc();
            pipeline.updatedAt = pipeline.createdAt;
            QString error;
            repository_.insertPipelineRun(pipeline, &error);
            registerPipelineModelVersion(payload);
            updateModelRegistry();
        }
    } else if (type == QStringLiteral("deliveryReport")) {
        updateModelRegistry();
    } else if (type == QStringLiteral("modelExport")) {
        if (exportResultLabel_) {
            const QString exportPath = payload.value(QStringLiteral("exportPath")).toString();
            const QString reportPath = payload.value(QStringLiteral("reportPath")).toString();
            exportResultLabel_->setText(reportPath.isEmpty()
                ? uiText("导出完成：%1").arg(QDir::toNativeSeparators(exportPath))
                : uiText("导出完成：%1；报告：%2").arg(QDir::toNativeSeparators(exportPath), QDir::toNativeSeparators(reportPath)));
        }
        if (repository_.isOpen()) {
            const QJsonObject config = payload.value(QStringLiteral("config")).toObject();
            aitrain::ExportRecord exportRecord;
            exportRecord.taskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
            exportRecord.sourceCheckpointPath = payload.value(QStringLiteral("checkpointPath")).toString();
            exportRecord.format = payload.value(QStringLiteral("format")).toString();
            exportRecord.path = payload.value(QStringLiteral("exportPath")).toString();
            exportRecord.configJson = QString::fromUtf8(QJsonDocument(config).toJson(QJsonDocument::Compact));
            exportRecord.inputShapeJson = QString::fromUtf8(QJsonDocument(config.value(QStringLiteral("input")).toObject()).toJson(QJsonDocument::Compact));
            exportRecord.outputShapeJson = QString::fromUtf8(QJsonDocument(QJsonObject{{QStringLiteral("outputs"), config.value(QStringLiteral("outputs")).toArray()}}).toJson(QJsonDocument::Compact));
            exportRecord.createdAt = QDateTime::currentDateTimeUtc();
            QString error;
            repository_.insertExport(exportRecord, &error);
        }
        updateModelRegistry();
    } else if (type == QStringLiteral("inferenceResult")) {
        if (inferenceResultLabel_) {
            inferenceResultLabel_->setText(inferenceSummaryFromPredictions(
                payload.value(QStringLiteral("predictionsPath")).toString(),
                payload));
        }
    }
}

void MainWindow::refreshPlugins()
{
    pluginManager_.scan(pluginSearchPaths());
    if (pluginTable_) {
        pluginTable_->setRowCount(0);
        const QVector<aitrain::IModelPlugin*> plugins = pluginManager_.plugins();
        if (plugins.isEmpty()) {
            pluginTable_->setRowCount(1);
            pluginTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无插件")));
            for (int column = 1; column < pluginTable_->columnCount(); ++column) {
                pluginTable_->setItem(0, column, new QTableWidgetItem(uiText("重新扫描或检查 plugins/models 目录。")));
            }
        }
        for (auto* plugin : plugins) {
            const aitrain::PluginManifest manifest = plugin->manifest();
            const int row = pluginTable_->rowCount();
            pluginTable_->insertRow(row);
            pluginTable_->setItem(row, 0, new QTableWidgetItem(manifest.id));
            pluginTable_->setItem(row, 1, new QTableWidgetItem(manifest.name));
            pluginTable_->setItem(row, 2, new QTableWidgetItem(manifest.version));
            pluginTable_->setItem(row, 3, new QTableWidgetItem(compactListSummary(manifest.taskTypes, 4)));
            pluginTable_->setItem(row, 4, new QTableWidgetItem(compactListSummary(manifest.datasetFormats, 4)));
            pluginTable_->setItem(row, 5, new QTableWidgetItem(compactListSummary(manifest.exportFormats, 4)));
            pluginTable_->setItem(row, 6, new QTableWidgetItem(manifest.requiresGpu ? uiText("需要") : uiText("否")));
        }
    }
    loadPluginCombos();
    updateHeaderState();
    updatePluginSummary();
    updateDashboardSummary();
    refreshTrainingDefaults();
}

QString MainWindow::workerExecutablePath() const
{
    const QString name =
#if defined(Q_OS_WIN)
        QStringLiteral("aitrain_worker.exe");
#else
        QStringLiteral("aitrain_worker");
#endif
    const QString appDir = QApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).filePath(name),
        QDir(appDir).filePath(QStringLiteral("../worker/") + name),
        QDir(appDir).filePath(QStringLiteral("../bin/") + name)
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QDir::cleanPath(candidate);
        }
    }
    return QDir(appDir).filePath(name);
}

QStringList MainWindow::pluginSearchPaths() const
{
    const QString appDir = QApplication::applicationDirPath();
    return {
        QDir(appDir).filePath(QStringLiteral("plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../plugins/models")),
        QDir(appDir).filePath(QStringLiteral("../../plugins/models"))
    };
}

void MainWindow::ensureProjectSubdirs(const QString& rootPath)
{
    QDir root(rootPath);
    root.mkpath(QStringLiteral("."));
    root.mkpath(QStringLiteral("datasets"));
    root.mkpath(QStringLiteral("runs"));
    root.mkpath(QStringLiteral("models"));
}

void MainWindow::appendLog(const QString& text)
{
    if (logEdit_) {
        logEdit_->append(QStringLiteral("[%1] %2").arg(QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), text));
    }
}

void MainWindow::loadPluginCombos()
{
    const QString currentPlugin = pluginCombo_ ? pluginCombo_->currentData().toString() : QString();
    if (pluginCombo_) {
        pluginCombo_->clear();
    }
    if (datasetFormatCombo_) {
        datasetFormatCombo_->clear();
    }

    QStringList formats;
    for (auto* plugin : pluginManager_.plugins()) {
        const aitrain::PluginManifest manifest = plugin->manifest();
        if (pluginCombo_) {
            pluginCombo_->addItem(manifest.name, manifest.id);
        }
        for (const QString& format : manifest.datasetFormats) {
            if (!formats.contains(format)) {
                formats.append(format);
            }
        }
    }
    if (datasetFormatCombo_) {
        for (const QString& format : formats) {
            datasetFormatCombo_->addItem(datasetFormatLabel(format), format);
        }
    }
    if (pluginCombo_ && !currentPlugin.isEmpty()) {
        const int index = pluginCombo_->findData(currentPlugin);
        if (index >= 0) {
            pluginCombo_->setCurrentIndex(index);
        }
    }
    refreshTrainingDefaults();
}

QString MainWindow::currentDatasetFormat() const
{
    return comboCurrentDataOrText(datasetFormatCombo_);
}

QString MainWindow::currentTaskType() const
{
    return comboCurrentDataOrText(taskTypeCombo_);
}

QString MainWindow::currentTaskKindFilter() const
{
    return comboCurrentDataOrText(taskKindFilterCombo_);
}

QString MainWindow::currentTaskStateFilter() const
{
    return comboCurrentDataOrText(taskStateFilterCombo_);
}

void MainWindow::updateRecentTasks()
{
    if (!repository_.isOpen()) {
        updateDashboardSummary();
        return;
    }
    QString error;
    const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(20, &error);
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(tasks.size()));
    }
    if (recentTasksTable_) {
        updateTaskTable(recentTasksTable_, tasks);
    }
    if (taskQueueTable_) {
        updateTaskTable(taskQueueTable_, tasks);
        applyTaskFilters();
    }
    updateDashboardSummary();
}

void MainWindow::updateDatasetList()
{
    if (!repository_.isOpen()) {
        updateDashboardSummary();
        return;
    }

    QString error;
    const QVector<aitrain::DatasetRecord> datasets = repository_.recentDatasets(50, &error);
    if (datasetListTable_) {
        datasetListTable_->setRowCount(0);
    }
    if (datasetListTable_ && datasets.isEmpty()) {
        datasetListTable_->insertRow(0);
        datasetListTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无数据集记录")));
        for (int column = 1; column < datasetListTable_->columnCount(); ++column) {
            datasetListTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        updateDashboardSummary();
        return;
    }

    if (datasetListTable_) {
        for (const aitrain::DatasetRecord& dataset : datasets) {
            const int row = datasetListTable_->rowCount();
            datasetListTable_->insertRow(row);
            auto* nameItem = new QTableWidgetItem(dataset.name);
            nameItem->setData(Qt::UserRole, dataset.id);
            datasetListTable_->setItem(row, 0, nameItem);
            auto* formatItem = new QTableWidgetItem(datasetFormatLabel(dataset.format));
            formatItem->setData(Qt::UserRole, dataset.format);
            datasetListTable_->setItem(row, 1, formatItem);
            auto* statusItem = new QTableWidgetItem(dataset.validationStatus == QStringLiteral("valid") ? uiText("通过") : uiText("未通过"));
            statusItem->setData(Qt::UserRole, dataset.validationStatus);
            datasetListTable_->setItem(row, 2, statusItem);
            datasetListTable_->setItem(row, 3, new QTableWidgetItem(QString::number(dataset.sampleCount)));
            auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(dataset.rootPath));
            pathItem->setData(Qt::UserRole, dataset.rootPath);
            datasetListTable_->setItem(row, 4, pathItem);
        }
    }
    updateDashboardSummary();
}

void MainWindow::updateAnnotationToolStatus()
{
    if (annotationToolStatusLabel_) {
        annotationToolStatusLabel_->setText(xAnyLabelingStatusText());
    }
}

void MainWindow::refreshAfterAnnotation()
{
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (datasetPath.isEmpty()) {
        QMessageBox::information(this, uiText("标注后刷新"), uiText("请先选择数据集目录。"));
        return;
    }

    const QString detectedFormat = detectDatasetFormatFromPath(datasetPath);
    if (!detectedFormat.isEmpty() && datasetFormatCombo_) {
        const int index = datasetFormatCombo_->findData(detectedFormat);
        if (index >= 0) {
            datasetFormatCombo_->setCurrentIndex(index);
        }
    }
    currentDatasetPath_ = datasetPath;
    const QString selectedFormat = currentDatasetFormat();
    currentDatasetFormat_ = selectedFormat.isEmpty() ? detectedFormat : selectedFormat;
    currentDatasetValid_ = false;
    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    validateDataset();
}

void MainWindow::applyTaskFilters()
{
    if (!taskQueueTable_ || taskQueueTable_->columnCount() < 7) {
        return;
    }

    const QString kind = currentTaskKindFilter();
    const QString state = currentTaskStateFilter();
    const QString query = taskSearchEdit_ ? taskSearchEdit_->text().trimmed() : QString();

    for (int row = 0; row < taskQueueTable_->rowCount(); ++row) {
        const QString rowKind = taskQueueTable_->item(row, 0)
            ? taskQueueTable_->item(row, 0)->data(Qt::UserRole + 1).toString()
            : QString();
        if (rowKind == QStringLiteral("empty")) {
            taskQueueTable_->setRowHidden(row, false);
            continue;
        }

        bool visible = true;
        if (!kind.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 1)
                && taskQueueTable_->item(row, 1)->data(Qt::UserRole).toString() == kind;
        }
        if (!state.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 4)
                && taskQueueTable_->item(row, 4)->data(Qt::UserRole).toString() == state;
        }
        if (!query.isEmpty()) {
            bool matched = false;
            for (int column = 0; column < taskQueueTable_->columnCount(); ++column) {
                auto* item = taskQueueTable_->item(row, column);
                if (item && item->text().contains(query, Qt::CaseInsensitive)) {
                    matched = true;
                    break;
                }
            }
            visible = visible && matched;
        }
        taskQueueTable_->setRowHidden(row, !visible);
    }
}

void MainWindow::updateTaskTable(QTableWidget* table, const QVector<aitrain::TaskRecord>& tasks)
{
    if (!table) {
        return;
    }

    table->setRowCount(0);
    if (tasks.isEmpty()) {
        table->insertRow(0);
        auto* item = new QTableWidgetItem(uiText("暂无任务记录"));
        item->setData(Qt::UserRole + 1, QStringLiteral("empty"));
        table->setItem(0, 0, item);
        for (int column = 1; column < table->columnCount(); ++column) {
            table->setItem(0, column, new QTableWidgetItem(QString()));
        }
        return;
    }

    for (const aitrain::TaskRecord& task : tasks) {
        const int row = table->rowCount();
        table->insertRow(row);
        auto* idItem = new QTableWidgetItem(task.id.left(8));
        idItem->setData(Qt::UserRole, task.id);
        table->setItem(row, 0, idItem);
        if (table->columnCount() == 5) {
            table->setItem(row, 1, new QTableWidgetItem(task.pluginId));
            auto* typeItem = new QTableWidgetItem(taskTypeLabel(task.taskType));
            typeItem->setData(Qt::UserRole, task.taskType);
            table->setItem(row, 2, typeItem);
            auto* stateItem = new QTableWidgetItem(taskStateLabel(task.state));
            stateItem->setData(Qt::UserRole, static_cast<int>(task.state));
            table->setItem(row, 3, stateItem);
            table->setItem(row, 4, new QTableWidgetItem(task.message));
        } else {
            auto* kindItem = new QTableWidgetItem(taskKindLabel(task.kind));
            QString kindData;
            switch (task.kind) {
            case aitrain::TaskKind::Train: kindData = QStringLiteral("train"); break;
            case aitrain::TaskKind::Validate: kindData = QStringLiteral("validate"); break;
            case aitrain::TaskKind::Export: kindData = QStringLiteral("export"); break;
            case aitrain::TaskKind::Infer: kindData = QStringLiteral("infer"); break;
            case aitrain::TaskKind::Evaluate: kindData = QStringLiteral("evaluate"); break;
            case aitrain::TaskKind::Benchmark: kindData = QStringLiteral("benchmark"); break;
            case aitrain::TaskKind::Curate: kindData = QStringLiteral("curate"); break;
            case aitrain::TaskKind::Snapshot: kindData = QStringLiteral("snapshot"); break;
            case aitrain::TaskKind::Pipeline: kindData = QStringLiteral("pipeline"); break;
            case aitrain::TaskKind::Report: kindData = QStringLiteral("report"); break;
            }
            kindItem->setData(Qt::UserRole, kindData);
            table->setItem(row, 1, kindItem);
            table->setItem(row, 2, new QTableWidgetItem(task.pluginId));
            auto* typeItem = new QTableWidgetItem(taskTypeLabel(task.taskType));
            typeItem->setData(Qt::UserRole, task.taskType);
            table->setItem(row, 3, typeItem);
            auto* stateItem = new QTableWidgetItem(taskStateLabel(task.state));
            QString stateData;
            switch (task.state) {
            case aitrain::TaskState::Queued: stateData = QStringLiteral("queued"); break;
            case aitrain::TaskState::Running: stateData = QStringLiteral("running"); break;
            case aitrain::TaskState::Paused: stateData = QStringLiteral("paused"); break;
            case aitrain::TaskState::Completed: stateData = QStringLiteral("completed"); break;
            case aitrain::TaskState::Failed: stateData = QStringLiteral("failed"); break;
            case aitrain::TaskState::Canceled: stateData = QStringLiteral("canceled"); break;
            }
            stateItem->setData(Qt::UserRole, stateData);
            table->setItem(row, 4, stateItem);
            table->setItem(row, 5, new QTableWidgetItem(task.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            table->setItem(row, 6, new QTableWidgetItem(task.message));
        }
    }
}

QString MainWindow::createRepositoryTask(aitrain::TaskKind kind,
    const QString& taskType,
    const QString& pluginId,
    const QString& workDir,
    const QString& message,
    const QString& requestedTaskId)
{
    if (!repository_.isOpen()) {
        return QString();
    }

    const QString taskId = requestedTaskId.isEmpty()
        ? QUuid::createUuid().toString(QUuid::WithoutBraces)
        : requestedTaskId;
    aitrain::TaskRecord record;
    record.id = taskId;
    record.projectName = currentProjectName_.isEmpty() ? QStringLiteral("manual") : currentProjectName_;
    record.pluginId = pluginId;
    record.taskType = taskType;
    record.kind = kind;
    record.state = aitrain::TaskState::Queued;
    record.workDir = workDir;
    record.message = message;
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;

    QString error;
    if (!repository_.insertTask(record, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return QString();
    }
    if (!repository_.updateTaskState(taskId, aitrain::TaskState::Running, message, &error)) {
        QMessageBox::critical(this, uiText("任务"), error);
        return QString();
    }
    currentTaskId_ = taskId;
    updateRecentTasks();
    return taskId;
}

bool MainWindow::attachLatestSnapshotToRequest(aitrain::TrainingRequest& request, int datasetId, QString* error)
{
    if (!repository_.isOpen() || datasetId <= 0) {
        return false;
    }

    const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(datasetId, error);
    if (snapshot.id <= 0 || snapshot.manifestPath.isEmpty() || !QFileInfo::exists(snapshot.manifestPath)) {
        return false;
    }

    request.parameters.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
    request.parameters.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
    request.parameters.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
    return true;
}

int MainWindow::recordExperimentRunForRequest(const aitrain::TrainingRequest& request, int datasetId, QString* error)
{
    if (!repository_.isOpen() || request.taskId.isEmpty()) {
        return 0;
    }

    const int snapshotId = request.parameters.value(QStringLiteral("datasetSnapshotId")).toInt();
    if (snapshotId <= 0) {
        return 0;
    }

    aitrain::ExperimentRecord experiment;
    const QString datasetName = QFileInfo(request.datasetPath).fileName().isEmpty()
        ? QStringLiteral("dataset")
        : QFileInfo(request.datasetPath).fileName();
    experiment.name = QStringLiteral("%1 / %2").arg(datasetName, taskTypeLabel(request.taskType));
    experiment.taskType = request.taskType;
    experiment.datasetId = datasetId;
    experiment.notes = uiText("Phase 35 自动记录的本地复现实验。");
    experiment.tagsJson = QString::fromUtf8(QJsonDocument(QJsonArray{
        QStringLiteral("phase35"),
        QStringLiteral("local-reproducible")
    }).toJson(QJsonDocument::Compact));
    experiment.createdAt = QDateTime::currentDateTimeUtc();
    experiment.updatedAt = experiment.createdAt;
    const int experimentId = repository_.upsertExperiment(experiment, error);
    if (experimentId <= 0) {
        return 0;
    }

    aitrain::ExperimentRunRecord run;
    run.experimentId = experimentId;
    run.taskId = request.taskId;
    run.trainingBackend = request.parameters.value(QStringLiteral("trainingBackend")).toString();
    run.modelPreset = request.parameters.value(QStringLiteral("modelPreset")).toString(
        request.parameters.value(QStringLiteral("model")).toString());
    run.datasetSnapshotId = snapshotId;
    run.requestJson = QString::fromUtf8(QJsonDocument(request.toJson()).toJson(QJsonDocument::Compact));
    run.environmentJson = QStringLiteral("{}");
    run.bestMetricsJson = QStringLiteral("{}");
    run.artifactSummaryJson = QStringLiteral("[]");
    run.createdAt = QDateTime::currentDateTimeUtc();
    run.updatedAt = run.createdAt;
    return repository_.insertExperimentRun(run, error);
}

void MainWindow::updateExperimentRunSummary(const QString& taskId)
{
    if (!repository_.isOpen() || taskId.isEmpty()) {
        return;
    }

    QString error;
    const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(taskId, &error);
    if (run.id <= 0) {
        return;
    }

    QJsonObject bestMetrics;
    const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
    for (const aitrain::MetricPoint& metric : metrics) {
        const QString key = metric.name;
        if (key.isEmpty()) {
            continue;
        }
        const QString lower = key.toLower();
        const bool minimize = lower.contains(QStringLiteral("loss"))
            || lower.contains(QStringLiteral("distance"))
            || lower == QStringLiteral("cer")
            || lower == QStringLiteral("wer");
        if (!bestMetrics.contains(key)
            || (minimize && metric.value < bestMetrics.value(key).toDouble())
            || (!minimize && metric.value > bestMetrics.value(key).toDouble())) {
            bestMetrics.insert(key, metric.value);
        }
    }

    QJsonArray artifactSummary;
    const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(taskId, &error);
    for (const aitrain::ArtifactRecord& artifact : artifacts) {
        artifactSummary.append(QJsonObject{
            {QStringLiteral("kind"), artifact.kind},
            {QStringLiteral("path"), artifact.path},
            {QStringLiteral("message"), artifact.message}
        });
    }

    repository_.updateExperimentRunSummary(
        taskId,
        QString::fromUtf8(QJsonDocument(bestMetrics).toJson(QJsonDocument::Compact)),
        QString::fromUtf8(QJsonDocument(artifactSummary).toJson(QJsonDocument::Compact)),
        &error);
}

void MainWindow::registerPipelineModelVersion(const QJsonObject& payload)
{
    if (!repository_.isOpen()) {
        return;
    }
    if (payload.value(QStringLiteral("state")).toString() != QStringLiteral("completed")) {
        return;
    }

    const QString sourceTaskId = payload.value(QStringLiteral("taskId")).toString(currentTaskId_);
    const QString exportPath = payload.value(QStringLiteral("exportPath")).toString();
    const QString modelPath = exportPath.isEmpty()
        ? payload.value(QStringLiteral("modelPath")).toString()
        : exportPath;
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        return;
    }

    const QJsonObject options = payload.value(QStringLiteral("options")).toObject();
    QString error;
    const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(sourceTaskId, &error);

    QJsonObject metricSummary;
    const QString evaluationReportPath = payload.value(QStringLiteral("evaluationReportPath")).toString();
    const QString benchmarkReportPath = payload.value(QStringLiteral("benchmarkReportPath")).toString();
    const QString deliveryReportPath = payload.value(QStringLiteral("deliveryReportPath")).toString();
    const QJsonObject evaluationSummary = compactEvaluationSummary(evaluationReportPath);
    const QJsonObject benchmarkSummary = compactBenchmarkSummary(benchmarkReportPath);
    if (!evaluationSummary.isEmpty()) {
        metricSummary.insert(QStringLiteral("evaluation"), evaluationSummary);
    }
    if (!benchmarkSummary.isEmpty()) {
        metricSummary.insert(QStringLiteral("benchmark"), benchmarkSummary);
    }
    metricSummary.insert(QStringLiteral("lineage"), QJsonObject{
        {QStringLiteral("sourceTaskId"), sourceTaskId},
        {QStringLiteral("datasetSnapshotId"), payload.value(QStringLiteral("datasetSnapshotId")).toInt()},
        {QStringLiteral("trainingBackend"), options.value(QStringLiteral("trainingBackend")).toString()},
        {QStringLiteral("modelPreset"), options.value(QStringLiteral("model")).toString()}
    });
    metricSummary.insert(QStringLiteral("artifacts"), QJsonObject{
        {QStringLiteral("modelPath"), modelPath},
        {QStringLiteral("checkpointPath"), payload.value(QStringLiteral("modelPath")).toString()},
        {QStringLiteral("exportPath"), exportPath},
        {QStringLiteral("evaluationReportPath"), evaluationReportPath},
        {QStringLiteral("benchmarkReportPath"), benchmarkReportPath},
        {QStringLiteral("deliveryReportPath"), deliveryReportPath}
    });
    QJsonArray limitations;
    const QString backendForLimit = options.value(QStringLiteral("trainingBackend")).toString();
    if (backendForLimit == QStringLiteral("tiny_linear_detector")
        || backendForLimit == QStringLiteral("python_mock")) {
        limitations.append(QStringLiteral("scaffold-or-diagnostic-backend"));
    }
    if (benchmarkSummary.value(QStringLiteral("failureCategory")).toString() == QStringLiteral("hardware-blocked")
        || benchmarkSummary.value(QStringLiteral("deploymentConclusion")).toString() == QStringLiteral("hardware-blocked")) {
        limitations.append(QStringLiteral("tensorrt-hardware-blocked"));
    }
    if (!limitations.isEmpty()) {
        metricSummary.insert(QStringLiteral("limitations"), limitations);
    }

    const QString backend = options.value(QStringLiteral("trainingBackend")).toString();
    const QString datasetName = QFileInfo(options.value(QStringLiteral("datasetPath")).toString()).fileName();
    const QString modelName = QStringLiteral("%1_%2_%3")
        .arg(currentProjectName_.isEmpty() ? QStringLiteral("pipeline_model") : currentProjectName_)
        .arg(datasetName.isEmpty() ? QStringLiteral("dataset") : datasetName)
        .arg(backend.isEmpty() ? QStringLiteral("pipeline") : backend);
    const QString version = QStringLiteral("pipeline_%1")
        .arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMddHHmmss")));

    aitrain::ModelVersionRecord record;
    record.modelName = modelName;
    record.version = version;
    record.sourceTaskId = sourceTaskId;
    record.experimentRunId = run.id;
    record.datasetSnapshotId = run.datasetSnapshotId > 0
        ? run.datasetSnapshotId
        : payload.value(QStringLiteral("datasetSnapshotId")).toInt();
    record.status = QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0
        ? QStringLiteral("exported")
        : QStringLiteral("validated");
    record.notes = uiText("由本地流水线自动注册。");
    record.metricsJson = QString::fromUtf8(QJsonDocument(metricSummary).toJson(QJsonDocument::Compact));
    if (QFileInfo(modelPath).suffix().compare(QStringLiteral("onnx"), Qt::CaseInsensitive) == 0) {
        record.onnxPath = modelPath;
        record.checkpointPath = payload.value(QStringLiteral("modelPath")).toString();
    } else {
        record.checkpointPath = modelPath;
    }
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;
    repository_.upsertModelVersion(record, &error);
}

void MainWindow::updateModelRegistry()
{
    if (!repository_.isOpen()) {
        if (modelVersionTable_) {
            modelVersionTable_->setRowCount(0);
            modelVersionTable_->insertRow(0);
            modelVersionTable_->setItem(0, 0, new QTableWidgetItem(uiText("请先打开项目")));
        }
        if (evaluationReportTable_) {
            evaluationReportTable_->setRowCount(0);
        }
        if (pipelineRunTable_) {
            pipelineRunTable_->setRowCount(0);
        }
        return;
    }

    QString error;
    const QVector<aitrain::ModelVersionRecord> models = repository_.recentModelVersions(200, &error);
    const QVector<aitrain::EvaluationReportRecord> reports = repository_.recentEvaluationReports(100, &error);
    const QVector<aitrain::PipelineRunRecord> pipelines = repository_.recentPipelineRuns(100, &error);

    if (modelRegistrySummaryLabel_) {
        modelRegistrySummaryLabel_->setText(uiText("模型版本：%1；评估报告：%2；流水线记录：%3。评估、基准和报告 v1 通过 Worker 生成 artifact，完整质量分析仍按 scaffold 标注。")
            .arg(models.size())
            .arg(reports.size())
            .arg(pipelines.size()));
    }

    if (modelVersionTable_) {
        modelVersionTable_->setRowCount(0);
        if (models.isEmpty()) {
            modelVersionTable_->insertRow(0);
            modelVersionTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无模型版本")));
            for (int column = 1; column < modelVersionTable_->columnCount(); ++column) {
                modelVersionTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ModelVersionRecord& model : models) {
                const int row = modelVersionTable_->rowCount();
                modelVersionTable_->insertRow(row);
                modelVersionTable_->setItem(row, 0, new QTableWidgetItem(model.modelName));
                modelVersionTable_->setItem(row, 1, new QTableWidgetItem(model.version));
                modelVersionTable_->setItem(row, 2, new QTableWidgetItem(model.status));
                auto* checkpointItem = new QTableWidgetItem(QDir::toNativeSeparators(model.checkpointPath));
                checkpointItem->setData(Qt::UserRole, model.checkpointPath);
                modelVersionTable_->setItem(row, 3, checkpointItem);
                auto* onnxItem = new QTableWidgetItem(QDir::toNativeSeparators(model.onnxPath));
                onnxItem->setData(Qt::UserRole, model.onnxPath);
                modelVersionTable_->setItem(row, 4, onnxItem);
                modelVersionTable_->setItem(row, 5, new QTableWidgetItem(model.sourceTaskId.left(8)));

                const QJsonObject metrics = QJsonDocument::fromJson(model.metricsJson.toUtf8()).object();
                const QString metricsText = metrics.isEmpty()
                    ? model.metricsJson
                    : modelSummaryText(metrics);
                modelVersionTable_->setItem(row, 6, new QTableWidgetItem(metricsText));
                modelVersionTable_->setItem(row, 7, new QTableWidgetItem(model.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (evaluationReportTable_) {
        evaluationReportTable_->setRowCount(0);
        if (reports.isEmpty()) {
            evaluationReportTable_->insertRow(0);
            evaluationReportTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无评估报告")));
            for (int column = 1; column < evaluationReportTable_->columnCount(); ++column) {
                evaluationReportTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::EvaluationReportRecord& report : reports) {
                const int row = evaluationReportTable_->rowCount();
                evaluationReportTable_->insertRow(row);
                evaluationReportTable_->setItem(row, 0, new QTableWidgetItem(report.taskId.left(8)));
                evaluationReportTable_->setItem(row, 1, new QTableWidgetItem(taskTypeLabel(report.taskType)));
                evaluationReportTable_->setItem(row, 2, new QTableWidgetItem(QDir::toNativeSeparators(report.modelPath)));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(report.reportPath));
                pathItem->setData(Qt::UserRole, report.reportPath);
                evaluationReportTable_->setItem(row, 3, pathItem);
                evaluationReportTable_->setItem(row, 4, new QTableWidgetItem(report.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
            if (evaluationReportTable_->rowCount() > 0) {
                evaluationReportTable_->selectRow(0);
            }
        }
    }
    updateSelectedEvaluationReportDetails();

    if (pipelineRunTable_) {
        pipelineRunTable_->setRowCount(0);
        if (pipelines.isEmpty()) {
            pipelineRunTable_->insertRow(0);
            pipelineRunTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无流水线记录")));
            for (int column = 1; column < pipelineRunTable_->columnCount(); ++column) {
                pipelineRunTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::PipelineRunRecord& pipeline : pipelines) {
                const int row = pipelineRunTable_->rowCount();
                pipelineRunTable_->insertRow(row);
                pipelineRunTable_->setItem(row, 0, new QTableWidgetItem(pipeline.name));
                pipelineRunTable_->setItem(row, 1, new QTableWidgetItem(pipeline.templateId));
                pipelineRunTable_->setItem(row, 2, new QTableWidgetItem(pipeline.state));
                pipelineRunTable_->setItem(row, 3, new QTableWidgetItem(pipeline.summaryJson.left(160)));
                pipelineRunTable_->setItem(row, 4, new QTableWidgetItem(pipeline.updatedAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }
}

QString MainWindow::selectedTaskId() const
{
    if (!taskQueueTable_ || taskQueueTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    auto* item = taskQueueTable_->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

QString MainWindow::selectedArtifactPath() const
{
    if (!taskArtifactTable_ || taskArtifactTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskArtifactTable_->selectedItems().first()->row();
    auto* item = taskArtifactTable_->item(row, 1);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

QString MainWindow::selectedEvaluationReportPath() const
{
    if (!evaluationReportTable_ || evaluationReportTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = evaluationReportTable_->selectedItems().first()->row();
    auto* item = evaluationReportTable_->item(row, 3);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void MainWindow::updateSelectedTaskDetails()
{
    if (!taskQueueTable_ || !repository_.isOpen()) {
        return;
    }
    if (taskQueueTable_->selectedItems().isEmpty()) {
        return;
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    const QString taskId = taskQueueTable_->item(row, 0)
        ? taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString()
        : QString();
    if (taskId.isEmpty()) {
        return;
    }
    const QString taskKind = taskQueueTable_->item(row, 1) ? taskQueueTable_->item(row, 1)->text() : QString();
    const QString taskBackend = taskQueueTable_->item(row, 2) ? taskQueueTable_->item(row, 2)->text() : QString();
    const QString taskType = taskQueueTable_->item(row, 3) ? taskQueueTable_->item(row, 3)->text() : QString();
    const QString taskState = taskQueueTable_->item(row, 4) ? taskQueueTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
    const QString taskMessage = taskQueueTable_->item(row, 6) ? taskQueueTable_->item(row, 6)->text() : QString();

    QString error;
    const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(taskId, &error);
    const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
    const QVector<aitrain::ExportRecord> exports = repository_.exportsForTask(taskId, &error);

    if (selectedTaskSummaryLabel_) {
        QString summary = uiText("任务 %1：%2 / %3 / %4，%5 个产物，%6 个指标点，%7 条导出记录")
            .arg(taskId.left(8))
            .arg(taskKind.isEmpty() ? uiText("任务") : taskKind)
            .arg(taskType.isEmpty() ? uiText("未记录类型") : taskType)
            .arg(taskBackend.isEmpty() ? uiText("未记录后端") : taskBackend)
            .arg(artifacts.size())
            .arg(metrics.size())
            .arg(exports.size());
        if (taskState == QStringLiteral("failed")) {
            summary.append(uiText("\n失败摘要：%1\n建议：优先查看 report/log 产物；若提示缺环境或缺 Python 包，进入“环境”页自检；若提示数据错误，回到“数据集”页重新校验。")
                .arg(taskMessage.isEmpty() ? uiText("Worker 未返回详细消息。") : taskMessage));
        } else if (!taskMessage.isEmpty()) {
            summary.append(uiText("\n最新消息：%1").arg(taskMessage));
        }
        selectedTaskSummaryLabel_->setText(summary);
    }

    if (taskArtifactTable_) {
        taskArtifactTable_->setRowCount(0);
        if (artifacts.isEmpty()) {
            taskArtifactTable_->insertRow(0);
            taskArtifactTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无产物")));
            for (int column = 1; column < taskArtifactTable_->columnCount(); ++column) {
                taskArtifactTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ArtifactRecord& artifact : artifacts) {
                const int artifactRow = taskArtifactTable_->rowCount();
                taskArtifactTable_->insertRow(artifactRow);
                taskArtifactTable_->setItem(artifactRow, 0, new QTableWidgetItem(artifact.kind));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(artifact.path));
                pathItem->setData(Qt::UserRole, artifact.path);
                taskArtifactTable_->setItem(artifactRow, 1, pathItem);
                taskArtifactTable_->setItem(artifactRow, 2, new QTableWidgetItem(artifact.message));
                taskArtifactTable_->setItem(artifactRow, 3, new QTableWidgetItem(artifact.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (taskMetricTable_) {
        taskMetricTable_->setRowCount(0);
        if (metrics.isEmpty()) {
            taskMetricTable_->insertRow(0);
            taskMetricTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无指标")));
            for (int column = 1; column < taskMetricTable_->columnCount(); ++column) {
                taskMetricTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::MetricPoint& metric : metrics) {
                const int metricRow = taskMetricTable_->rowCount();
                taskMetricTable_->insertRow(metricRow);
                taskMetricTable_->setItem(metricRow, 0, new QTableWidgetItem(metric.name));
                taskMetricTable_->setItem(metricRow, 1, new QTableWidgetItem(QString::number(metric.value, 'f', 6)));
                taskMetricTable_->setItem(metricRow, 2, new QTableWidgetItem(QString::number(metric.step)));
                taskMetricTable_->setItem(metricRow, 3, new QTableWidgetItem(QString::number(metric.epoch)));
            }
        }
    }

    if (taskExportTable_) {
        taskExportTable_->setRowCount(0);
        if (exports.isEmpty()) {
            taskExportTable_->insertRow(0);
            taskExportTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无导出")));
            for (int column = 1; column < taskExportTable_->columnCount(); ++column) {
                taskExportTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ExportRecord& exportRecord : exports) {
                const int exportRow = taskExportTable_->rowCount();
                taskExportTable_->insertRow(exportRow);
                taskExportTable_->setItem(exportRow, 0, new QTableWidgetItem(exportRecord.format));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(exportRecord.path));
                pathItem->setData(Qt::UserRole, exportRecord.path);
                taskExportTable_->setItem(exportRow, 1, pathItem);
                taskExportTable_->setItem(exportRow, 2, new QTableWidgetItem(exportRecord.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (!artifacts.isEmpty() && taskArtifactTable_) {
        int preferredRow = 0;
        for (int artifactRow = 0; artifactRow < taskArtifactTable_->rowCount(); ++artifactRow) {
            auto* kindItem = taskArtifactTable_->item(artifactRow, 0);
            if (kindItem && kindItem->text() == QStringLiteral("evaluation_report")) {
                preferredRow = artifactRow;
                break;
            }
        }
        taskArtifactTable_->selectRow(preferredRow);
    }
}

void MainWindow::previewArtifactPath(const QString& path)
{
    if (!artifactPreviewText_ || !artifactImagePreviewLabel_ || !artifactPreviewStack_) {
        return;
    }

    artifactPreviewStack_->setCurrentIndex(0);
    if (artifactEvaluationReportView_) {
        artifactEvaluationReportView_->clear();
    }
    artifactImagePreviewLabel_->clear();
    artifactImagePreviewLabel_->setText(uiText("暂无图片预览"));
    artifactPreviewText_->clear();
    if (path.isEmpty()) {
        artifactPreviewText_->setPlainText(uiText("请选择一个产物。"));
        return;
    }

    const QFileInfo info(path);
    if (!info.exists()) {
        artifactPreviewText_->setPlainText(uiText("产物不存在：%1").arg(QDir::toNativeSeparators(path)));
        return;
    }

    if (info.isDir()) {
        artifactPreviewText_->setPlainText(uiText("目录产物\n路径：%1\n修改时间：%2")
            .arg(QDir::toNativeSeparators(path), info.lastModified().toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
        return;
    }

    const QString suffix = info.suffix().toLower();
    if (QStringList{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("bmp")}.contains(suffix)) {
        QPixmap image(path);
        if (!image.isNull()) {
            artifactImagePreviewLabel_->setPixmap(image.scaled(
                artifactImagePreviewLabel_->size().boundedTo(QSize(520, 360)),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation));
        }
        artifactPreviewText_->setPlainText(uiText("图片产物\n路径：%1\n尺寸：%2 x %3\n大小：%4 bytes")
            .arg(QDir::toNativeSeparators(path))
            .arg(image.width())
            .arg(image.height())
            .arg(info.size()));
        return;
    }

    if (suffix == QStringLiteral("onnx")) {
        artifactPreviewText_->setPlainText(uiText("ONNX 模型\n路径：%1\n模型族：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), aitrain::inferOnnxModelFamily(path))
            .arg(info.size()));
        return;
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan") || suffix == QStringLiteral("pdparams")
        || suffix == QStringLiteral("aitrain") || suffix == QStringLiteral("param") || suffix == QStringLiteral("bin")) {
        artifactPreviewText_->setPlainText(uiText("模型产物\n路径：%1\n类型：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), suffix)
            .arg(info.size()));
        return;
    }

    if (QStringList{QStringLiteral("json"), QStringLiteral("yaml"), QStringLiteral("yml"), QStringLiteral("txt"), QStringLiteral("csv"), QStringLiteral("log")}.contains(suffix)) {
        if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("evaluation_report.json") && artifactEvaluationReportView_) {
            artifactEvaluationReportView_->loadReport(path);
            artifactPreviewStack_->setCurrentIndex(1);
            return;
        }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            artifactPreviewText_->setPlainText(uiText("无法读取文本产物：%1").arg(QDir::toNativeSeparators(path)));
            return;
        }
        const qint64 maxBytes = 256 * 1024;
        const QByteArray data = file.read(maxBytes);
        QString text = suffix == QStringLiteral("json") ? formatJsonTextForPreview(data) : QString::fromUtf8(data);
        if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("evaluation_report.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
                QStringList lines;
                lines << uiText("评估报告摘要");
                lines << uiText("任务类型：%1").arg(report.value(QStringLiteral("taskType")).toString());
                lines << uiText("真实评估：%1").arg(report.value(QStringLiteral("scaffold")).toBool() ? uiText("否，scaffold") : uiText("是"));
                lines << uiText("precision=%1 recall=%2 mAP50=%3")
                    .arg(metrics.value(QStringLiteral("precision")).toDouble(), 0, 'f', 4)
                    .arg(metrics.value(QStringLiteral("recall")).toDouble(), 0, 'f', 4)
                    .arg(metrics.value(QStringLiteral("mAP50")).toDouble(), 0, 'f', 4);
                lines << uiText("错误样本：%1；低置信样本：%2")
                    .arg(report.value(QStringLiteral("errorSamples")).toArray().size())
                    .arg(report.value(QStringLiteral("lowConfidenceSamples")).toArray().size());
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        } else if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("dataset_quality_report.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonObject severity = report.value(QStringLiteral("severityCounts")).toObject();
                const QJsonObject summary = report.value(QStringLiteral("summary")).toObject();
                QStringList lines;
                lines << uiText("数据质量报告摘要");
                lines << uiText("格式：%1；真实分析：%2")
                    .arg(report.value(QStringLiteral("format")).toString())
                    .arg(report.value(QStringLiteral("scaffold")).toBool() ? uiText("否，scaffold") : uiText("是"));
                lines << uiText("error=%1 warning=%2 info=%3 问题样本=%4 重复图片=%5")
                    .arg(severity.value(QStringLiteral("error")).toInt())
                    .arg(severity.value(QStringLiteral("warning")).toInt())
                    .arg(severity.value(QStringLiteral("info")).toInt())
                    .arg(summary.value(QStringLiteral("problemSampleCount")).toInt())
                    .arg(summary.value(QStringLiteral("duplicateImageCount")).toInt());
                lines << uiText("修复清单：%1").arg(QDir::toNativeSeparators(report.value(QStringLiteral("xAnyLabelingFixListPath")).toString()));
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        } else if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("problem_samples.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonArray samples = report.value(QStringLiteral("samples")).toArray();
                QStringList lines;
                lines << uiText("问题样本摘要");
                lines << uiText("问题样本数：%1").arg(samples.size());
                const int previewCount = qMin(5, samples.size());
                for (int index = 0; index < previewCount; ++index) {
                    const QJsonObject sample = samples.at(index).toObject();
                    lines << QStringLiteral("%1. %2 %3 %4")
                        .arg(index + 1)
                        .arg(sample.value(QStringLiteral("severity")).toString())
                        .arg(sample.value(QStringLiteral("code")).toString())
                        .arg(QDir::toNativeSeparators(sample.value(QStringLiteral("imagePath")).toString()));
                }
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        }
        if (info.size() > maxBytes) {
            text.append(uiText("\n\n[文件超过 256KB，仅显示前部内容]\n路径：%1").arg(QDir::toNativeSeparators(path)));
        }
        artifactPreviewText_->setPlainText(text);
        return;
    }

    artifactPreviewText_->setPlainText(uiText("不支持内联预览的产物\n路径：%1\n大小：%2 bytes")
        .arg(QDir::toNativeSeparators(path))
        .arg(info.size()));
}

void MainWindow::updateSelectedEvaluationReportDetails()
{
    if (!evaluationReportView_) {
        return;
    }
    const QString reportPath = selectedEvaluationReportPath();
    if (reportPath.isEmpty()) {
        evaluationReportView_->clear();
        return;
    }
    evaluationReportView_->loadReport(reportPath);
}

void MainWindow::updateArtifactPreviewFromSelection()
{
    previewArtifactPath(selectedArtifactPath());
}

void MainWindow::openSelectedArtifactDirectory()
{
    const QString path = selectedArtifactPath();
    if (path.isEmpty()) {
        return;
    }
    const QFileInfo info(path);
    const QString directory = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    QDesktopServices::openUrl(QUrl::fromLocalFile(directory));
}

void MainWindow::copySelectedArtifactPath()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty()) {
        QApplication::clipboard()->setText(QDir::toNativeSeparators(path));
        statusBar()->showMessage(uiText("产物路径已复制"), 3000);
    }
}

void MainWindow::useSelectedArtifactForInference()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty() && inferenceCheckpointEdit_) {
        inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        showPage(InferencePage, tr("推理验证"));
    }
}

void MainWindow::useSelectedArtifactForExport()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty() && conversionCheckpointEdit_) {
        conversionCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        showPage(ConversionPage, tr("模型导出"));
    }
}

void MainWindow::registerSelectedArtifactAsModelVersion()
{
    if (!repository_.isOpen()) {
        QMessageBox::information(this, uiText("模型注册"), uiText("请先打开项目。"));
        return;
    }
    const QString path = selectedArtifactPath();
    if (path.isEmpty()) {
        QMessageBox::information(this, uiText("模型注册"), uiText("请先选择一个 checkpoint、ONNX 或 engine 产物。"));
        return;
    }

    const QFileInfo info(path);
    const QString suffix = info.suffix().toLower();
    const QString defaultName = currentProjectName_.isEmpty() ? QStringLiteral("model") : currentProjectName_;
    bool ok = false;
    const QString modelName = QInputDialog::getText(this, uiText("模型注册"), uiText("模型名称"), QLineEdit::Normal, defaultName, &ok).trimmed();
    if (!ok || modelName.isEmpty()) {
        return;
    }

    const int existingCount = repository_.recentModelVersions(500).size();
    const QString defaultVersion = QStringLiteral("v%1").arg(existingCount + 1);
    const QString version = QInputDialog::getText(this, uiText("模型注册"), uiText("版本号"), QLineEdit::Normal, defaultVersion, &ok).trimmed();
    if (!ok || version.isEmpty()) {
        return;
    }

    QJsonObject metricSummary;
    const QString taskId = selectedTaskId();
    if (!taskId.isEmpty()) {
        QString error;
        const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
        for (const aitrain::MetricPoint& metric : metrics) {
            metricSummary.insert(metric.name, metric.value);
        }
    }

    aitrain::ModelVersionRecord record;
    record.modelName = modelName;
    record.version = version;
    record.sourceTaskId = taskId;
    if (!taskId.isEmpty()) {
        QString runError;
        const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(taskId, &runError);
        if (run.id > 0) {
            record.experimentRunId = run.id;
            record.datasetSnapshotId = run.datasetSnapshotId;
        }
    }
    record.status = suffix == QStringLiteral("onnx") ? QStringLiteral("exported") : QStringLiteral("draft");
    record.notes = uiText("从任务产物手动注册。");
    record.metricsJson = QString::fromUtf8(QJsonDocument(metricSummary).toJson(QJsonDocument::Compact));
    if (suffix == QStringLiteral("onnx")) {
        record.onnxPath = path;
    } else if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan")) {
        record.tensorRtEnginePath = path;
    } else {
        record.checkpointPath = path;
    }
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;

    QString error;
    const int id = repository_.upsertModelVersion(record, &error);
    if (id <= 0) {
        QMessageBox::critical(this, uiText("模型注册"), error);
        return;
    }
    updateModelRegistry();
    updateDashboardSummary();
    statusBar()->showMessage(uiText("已注册模型版本：%1:%2").arg(modelName, version), 4000);
    showPage(ModelRegistryPage, uiText("模型库"));
}

void MainWindow::evaluateSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("模型评估"), uiText("Worker 正在执行任务，稍后再评估模型。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    if (modelPath.isEmpty() || datasetPath.isEmpty()) {
        QMessageBox::warning(this, uiText("模型评估"), uiText("请先选择模型产物，并在数据集页选择评估数据集。"));
        return;
    }

    const QString taskType = currentTaskType().isEmpty() ? QStringLiteral("detection") : currentTaskType();
    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Evaluate,
            taskType,
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("模型评估报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("scaffoldAcknowledged"), true);
    if (repository_.isOpen()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &snapshotError);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &snapshotError);
        if (snapshot.id > 0) {
            options.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
            options.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
            options.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
        }
    }
    QString error;
    if (!worker_.requestModelEvaluation(workerExecutablePath(), modelPath, datasetPath, outputPath, taskType, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("模型评估"), error);
        return;
    }
    workerPill_->setStatus(uiText("模型评估中"), StatusPill::Tone::Info);
}

void MainWindow::benchmarkSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("部署基准"), uiText("Worker 正在执行任务，稍后再运行部署基准。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("部署基准"), uiText("请先选择一个模型产物。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Benchmark,
            QStringLiteral("model_benchmark"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("部署基准报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject options;
    options.insert(QStringLiteral("device"), QStringLiteral("cpu"));
    options.insert(QStringLiteral("batch"), 1);
    QString error;
    if (!worker_.requestModelBenchmark(workerExecutablePath(), modelPath, outputPath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("部署基准"), error);
        return;
    }
    workerPill_->setStatus(uiText("部署基准运行中"), StatusPill::Tone::Info);
}

void MainWindow::generateDeliveryReportFromSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("Worker 正在执行任务，稍后再生成交付报告。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("请先选择一个模型或报告产物。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("delivery_report"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("训练交付报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject context;
    context.insert(QStringLiteral("projectName"), currentProjectName_);
    context.insert(QStringLiteral("projectPath"), currentProjectPath_);
    context.insert(QStringLiteral("modelPath"), modelPath);
    context.insert(QStringLiteral("datasetPath"), currentDatasetPath_);
    context.insert(QStringLiteral("datasetFormat"), currentDatasetFormat_);
    context.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    QString error;
    if (!worker_.requestDeliveryReport(workerExecutablePath(), outputPath, context, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("交付报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("交付报告生成中"), StatusPill::Tone::Info);
}

void MainWindow::runLocalPipelinePlanFromCurrentDataset()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("本地流水线"), uiText("Worker 正在执行任务，稍后再执行流水线。"));
        return;
    }

    const QStringList templateLabels = {
        uiText("训练->评估->导出->注册->报告"),
        uiText("导出->推理->基准->报告")
    };
    bool ok = false;
    const QString selectedTemplate = QInputDialog::getItem(
        this,
        uiText("本地流水线"),
        uiText("选择流水线模板"),
        templateLabels,
        0,
        false,
        &ok);
    if (!ok || selectedTemplate.isEmpty()) {
        return;
    }
    const QString templateId = selectedTemplate == templateLabels.at(1)
        ? QStringLiteral("export-infer-benchmark-report")
        : QStringLiteral("train-evaluate-export-register");

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Pipeline,
            QStringLiteral("local_pipeline"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("本地流水线执行中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    int datasetId = 0;
    if (repository_.isOpen()) {
        QString repositoryError;
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &repositoryError);
        datasetId = dataset.id;
    }

    QJsonObject options;
    const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    options.insert(QStringLiteral("datasetId"), datasetId);
    options.insert(QStringLiteral("datasetPath"), datasetPath);
    options.insert(QStringLiteral("datasetFormat"), currentDatasetFormat());
    options.insert(QStringLiteral("taskType"), currentTaskType());
    options.insert(QStringLiteral("trainingBackend"), trainingBackendCombo_ ? trainingBackendCombo_->currentData().toString() : QString());
    options.insert(QStringLiteral("modelPreset"), modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString());
    options.insert(QStringLiteral("epochs"), epochsEdit_ ? epochsEdit_->text().toInt() : 1);
    options.insert(QStringLiteral("batchSize"), batchEdit_ ? batchEdit_->text().toInt() : 1);
    options.insert(QStringLiteral("imageSize"), imageSizeEdit_ ? imageSizeEdit_->text().toInt() : 640);
    options.insert(QStringLiteral("exportFormat"), QStringLiteral("onnx"));
    options.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    options.insert(QStringLiteral("modelPath"), selectedArtifactPath());
    options.insert(QStringLiteral("sampleImagePath"), inferenceImageEdit_ ? QDir::fromNativeSeparators(inferenceImageEdit_->text().trimmed()) : QString());

    QString error;
    if (!worker_.requestLocalPipeline(workerExecutablePath(), outputPath, templateId, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("本地流水线"), error);
        return;
    }
    workerPill_->setStatus(uiText("本地流水线执行中"), StatusPill::Tone::Info);
}

void MainWindow::refreshModelRegistry()
{
    updateModelRegistry();
}

void MainWindow::updateHeaderState()
{
    headerProjectLabel_->setText(currentProjectPath_.isEmpty()
        ? tr("项目：未打开")
        : tr("项目：%1").arg(currentProjectName_));
    const int pluginCount = pluginManager_.plugins().size();
    pluginPill_->setStatus(tr("插件 %1").arg(pluginCount), pluginCount > 0 ? StatusPill::Tone::Success : StatusPill::Tone::Warning);
    if (dashboardPluginValue_) {
        dashboardPluginValue_->setText(QString::number(pluginCount));
    }
    updatePluginSummary();
}

void MainWindow::updateEnvironmentTable(const QJsonObject& payload)
{
    const QJsonArray checks = payload.value(QStringLiteral("checks")).toArray();
    if (!environmentTable_) {
        return;
    }

    bool hasMissing = false;
    bool hasWarning = false;
    const auto markStatus = [&hasMissing, &hasWarning](const QString& status) {
        if (status == QStringLiteral("missing")) {
            hasMissing = true;
        } else if (status == QStringLiteral("warning") || status == QStringLiteral("hardware-blocked")) {
            hasWarning = true;
        }
    };
    const auto appendRow = [this, &payload, &markStatus](const QString& name, const QString& status, const QString& message, const QJsonObject& details = {}) {
        markStatus(status);
        const int row = environmentTable_->rowCount();
        environmentTable_->insertRow(row);
        environmentTable_->setItem(row, 0, new QTableWidgetItem(name));
        auto* statusItem = new QTableWidgetItem(environmentStatusLabel(status));
        statusItem->setData(Qt::UserRole, status);
        environmentTable_->setItem(row, 1, statusItem);
        environmentTable_->setItem(row, 2, new QTableWidgetItem(message));

        if (repository_.isOpen()) {
            aitrain::EnvironmentCheckRecord record;
            record.name = name;
            record.status = status;
            record.message = message;
            if (!details.isEmpty()) {
                record.detailsJson = QString::fromUtf8(QJsonDocument(details).toJson(QJsonDocument::Compact));
            }
            record.checkedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
            QString error;
            repository_.insertEnvironmentCheck(record, &error);
        }
    };

    environmentTable_->setRowCount(0);
    for (const QJsonValue& value : checks) {
        const QJsonObject check = value.toObject();
        const QString name = check.value(QStringLiteral("name")).toString();
        const QString status = check.value(QStringLiteral("status")).toString();
        const QString message = check.value(QStringLiteral("message")).toString();
        appendRow(name, status, message, check.value(QStringLiteral("details")).toObject());
    }

    const QJsonObject profiles = payload.value(QStringLiteral("profiles")).toObject();
    for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it) {
        const QJsonObject profile = it.value().toObject();
        const QString title = profile.value(QStringLiteral("title")).toString(it.key());
        const QString status = profile.value(QStringLiteral("status")).toString(QStringLiteral("warning"));
        const QJsonArray repairHints = profile.value(QStringLiteral("repairHints")).toArray();
        QStringList hintText;
        for (const QJsonValue& value : repairHints) {
            const QString text = value.toString().trimmed();
            if (!text.isEmpty()) {
                hintText.append(text);
            }
        }
        if (hintText.isEmpty()) {
            hintText.append(uiText("暂无修复建议。"));
        }
        appendRow(
            QStringLiteral("Profile / %1").arg(title),
            status,
            hintText.join(QStringLiteral(" | ")),
            profile);

        const QJsonArray profileChecks = profile.value(QStringLiteral("checks")).toArray();
        for (const QJsonValue& checkValue : profileChecks) {
            const QJsonObject check = checkValue.toObject();
            appendRow(
                QStringLiteral("%1 / %2").arg(title, check.value(QStringLiteral("name")).toString()),
                check.value(QStringLiteral("status")).toString(QStringLiteral("warning")),
                check.value(QStringLiteral("message")).toString(),
                check.value(QStringLiteral("details")).toObject());
        }
    }

    {
        const int pluginCount = pluginManager_.plugins().size();
        const QString status = pluginCount > 0 ? QStringLiteral("ok") : QStringLiteral("warning");
        const QString message = pluginCount > 0
            ? uiText("已加载 %1 个 AITrain 插件。").arg(pluginCount)
            : uiText("未加载 AITrain 插件，请检查 plugins/models 目录。");
        appendRow(uiText("AITrain Plugins"), status, message);
    }

    const StatusPill::Tone tone = hasMissing ? StatusPill::Tone::Error : (hasWarning ? StatusPill::Tone::Warning : StatusPill::Tone::Success);
    const QString text = hasMissing ? uiText("环境缺失") : (hasWarning ? uiText("环境警告") : uiText("环境通过"));
    gpuPill_->setStatus(text, tone);
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    gpuLabel_->setText(uiText("GPU / 运行时：%1").arg(text));
    if (dashboardEnvironmentValue_) {
        dashboardEnvironmentValue_->setText(hasMissing ? uiText("缺失") : (hasWarning ? uiText("警告") : uiText("通过")));
    }
    updateEnvironmentSummary();
    updateDashboardSummary();
    statusBar()->showMessage(uiText("环境自检完成"), 5000);
}

void MainWindow::updateDatasetValidationResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const int sampleCount = payload.value(QStringLiteral("sampleCount")).toInt();
    const QString datasetPath = payload.value(QStringLiteral("datasetPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString();
    const QJsonArray issues = payload.value(QStringLiteral("issues")).toArray();

    currentDatasetPath_ = datasetPath;
    currentDatasetFormat_ = format;
    currentDatasetValid_ = ok;

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? uiText("校验通过：%1 个样本。").arg(sampleCount)
            : uiText("校验失败：发现 %1 个问题，训练已被阻止。").arg(issues.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (datasetPreviewTable_) {
        datasetPreviewTable_->setRowCount(0);
        const QJsonArray previewSamples = payload.value(QStringLiteral("previewSamples")).toArray();
        if (previewSamples.isEmpty()) {
            datasetPreviewTable_->insertRow(0);
            datasetPreviewTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无样本预览")));
            datasetPreviewTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        } else {
            for (const QJsonValue& sampleValue : previewSamples) {
                const QString sample = sampleValue.toString();
                const QStringList parts = sample.split(QLatin1Char('\t'));
                const int row = datasetPreviewTable_->rowCount();
                datasetPreviewTable_->insertRow(row);
                datasetPreviewTable_->setItem(row, 0, new QTableWidgetItem(parts.value(0)));
                datasetPreviewTable_->setItem(row, 1, new QTableWidgetItem(parts.size() > 1 ? parts.mid(1).join(QStringLiteral("\t")) : uiText("标注文件已匹配")));
            }
        }
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (issues.isEmpty()) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(datasetPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("未发现数据集问题。")));
        } else {
            for (const QJsonValue& value : issues) {
                const QJsonObject issue = value.toObject();
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(issueSeverityLabel(issue.value(QStringLiteral("severity")).toString())));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(issue.value(QStringLiteral("code")).toString()));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(issue.value(QStringLiteral("filePath")).toString()));
                const int line = issue.value(QStringLiteral("line")).toInt();
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(line > 0 ? QString::number(line) : QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(issue.value(QStringLiteral("message")).toString()));
            }
        }
    }

    if (repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(datasetPath).fileName();
        dataset.format = format;
        dataset.rootPath = datasetPath;
        dataset.validationStatus = ok ? QStringLiteral("valid") : QStringLiteral("invalid");
        dataset.sampleCount = sampleCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        updateDatasetList();
    }

    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    updateDashboardSummary();
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? uiText("数据集校验通过") : uiText("数据集校验失败"), 5000);
}

void MainWindow::updateDatasetSplitResult(const QJsonObject& payload)
{
    const bool ok = payload.value(QStringLiteral("ok")).toBool();
    const QString outputPath = payload.value(QStringLiteral("outputPath")).toString();
    const QString format = payload.value(QStringLiteral("format")).toString(QStringLiteral("yolo_detection"));
    const int trainCount = payload.value(QStringLiteral("trainCount")).toInt();
    const int valCount = payload.value(QStringLiteral("valCount")).toInt();
    const int testCount = payload.value(QStringLiteral("testCount")).toInt();
    const QJsonArray errors = payload.value(QStringLiteral("errors")).toArray();

    if (validationSummaryLabel_) {
        validationSummaryLabel_->setText(ok
            ? uiText("划分完成：train %1 / val %2 / test %3。").arg(trainCount).arg(valCount).arg(testCount)
            : uiText("划分失败：%1 个错误。").arg(errors.size()));
    }
    if (validationOutput_) {
        validationOutput_->setPlainText(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    }
    if (validationIssuesTable_) {
        validationIssuesTable_->setRowCount(0);
        if (ok) {
            validationIssuesTable_->insertRow(0);
            validationIssuesTable_->setItem(0, 0, new QTableWidgetItem(uiText("通过")));
            validationIssuesTable_->setItem(0, 1, new QTableWidgetItem(QStringLiteral("split_ok")));
            validationIssuesTable_->setItem(0, 2, new QTableWidgetItem(outputPath));
            validationIssuesTable_->setItem(0, 3, new QTableWidgetItem(QString()));
            validationIssuesTable_->setItem(0, 4, new QTableWidgetItem(uiText("数据集已复制到标准划分目录。")));
        } else {
            for (const QJsonValue& value : errors) {
                const int row = validationIssuesTable_->rowCount();
                validationIssuesTable_->insertRow(row);
                validationIssuesTable_->setItem(row, 0, new QTableWidgetItem(uiText("错误")));
                validationIssuesTable_->setItem(row, 1, new QTableWidgetItem(QStringLiteral("split_error")));
                validationIssuesTable_->setItem(row, 2, new QTableWidgetItem(outputPath));
                validationIssuesTable_->setItem(row, 3, new QTableWidgetItem(QString()));
                validationIssuesTable_->setItem(row, 4, new QTableWidgetItem(value.toString()));
            }
        }
    }

    if (ok && repository_.isOpen()) {
        aitrain::DatasetRecord dataset;
        dataset.name = QFileInfo(outputPath).fileName();
        dataset.format = format;
        dataset.rootPath = outputPath;
        dataset.validationStatus = QStringLiteral("valid");
        dataset.sampleCount = trainCount + valCount + testCount;
        dataset.lastReportJson = QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Compact));
        dataset.lastValidatedAt = QDateTime::fromString(payload.value(QStringLiteral("checkedAt")).toString(), Qt::ISODateWithMs);
        QString error;
        repository_.upsertDatasetValidation(dataset, &error);
        updateDatasetList();
        datasetPathEdit_->setText(QDir::toNativeSeparators(outputPath));
        setComboCurrentData(datasetFormatCombo_, format);
        currentDatasetPath_ = outputPath;
        currentDatasetFormat_ = format;
        currentDatasetValid_ = true;
    }

    updateTrainingSelectionSummary();
    refreshTrainingDefaults();
    updateDashboardSummary();
    workerPill_->setStatus(uiText("Worker 空闲"), StatusPill::Tone::Neutral);
    statusBar()->showMessage(ok ? uiText("数据集划分完成") : uiText("数据集划分失败"), 5000);
}

void MainWindow::startQueuedTraining(const QString& taskId, const aitrain::TrainingRequest& request)
{
    metricsWidget_->clear();
    logEdit_->clear();
    progressBar_->setValue(0);
    currentTaskId_ = taskId;

    QString error;
    if (!worker_.startTraining(workerExecutablePath(), request, &error)) {
        repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, nullptr);
        currentTaskId_.clear();
        updateRecentTasks();
        QMessageBox::critical(this, QStringLiteral("Worker"), error);
        startNextQueuedTask();
        return;
    }

    repository_.updateTaskState(taskId, aitrain::TaskState::Running, QStringLiteral("started"), &error);
    workerPill_->setStatus(uiText("训练运行中"), StatusPill::Tone::Info);
    appendLog(uiText("任务已启动：%1").arg(taskId));
    updateRecentTasks();
}

void MainWindow::startNextQueuedTask()
{
    if (worker_.isRunning() || pendingTrainingTasks_.isEmpty()) {
        return;
    }

    const PendingTrainingTask next = pendingTrainingTasks_.takeFirst();
    if (next.needsSnapshot) {
        startSnapshotForQueuedTraining(next);
        return;
    }

    QString error;
    recordExperimentRunForRequest(next.request, next.datasetId, &error);
    startQueuedTraining(next.taskId, next.request);
}

void MainWindow::startSnapshotForQueuedTraining(const PendingTrainingTask& pending)
{
    if (!repository_.isOpen()) {
        return;
    }

    if (pending.datasetId <= 0 || pending.request.datasetPath.isEmpty()) {
        QString error;
        repository_.updateTaskState(pending.taskId, aitrain::TaskState::Failed, uiText("无法为训练创建数据快照：数据集记录缺失。"), &error);
        updateRecentTasks();
        return;
    }

    const QString snapshotTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QString outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(snapshotTaskId));
    const QString createdTaskId = createRepositoryTask(
        aitrain::TaskKind::Snapshot,
        QStringLiteral("dataset_snapshot"),
        QStringLiteral("com.aitrain.plugins.dataset_interop"),
        outputPath,
        uiText("为训练自动创建数据快照。"),
        snapshotTaskId);
    if (createdTaskId.isEmpty()) {
        return;
    }

    hasActiveSnapshotTrainingTask_ = true;
    activeSnapshotTrainingTask_ = pending;

    QJsonObject options;
    options.insert(QStringLiteral("maxFiles"), 20000);

    QString error;
    if (!worker_.requestDatasetSnapshot(workerExecutablePath(), pending.request.datasetPath, outputPath, pending.datasetFormat, options, &error, createdTaskId)) {
        repository_.updateTaskState(createdTaskId, aitrain::TaskState::Failed, error, nullptr);
        repository_.updateTaskState(pending.taskId, aitrain::TaskState::Failed, uiText("自动数据快照失败：%1").arg(error), nullptr);
        hasActiveSnapshotTrainingTask_ = false;
        activeSnapshotTrainingTask_ = PendingTrainingTask();
        currentTaskId_.clear();
        updateRecentTasks();
        QMessageBox::critical(this, uiText("数据快照"), error);
        return;
    }

    workerPill_->setStatus(uiText("自动数据快照创建中"), StatusPill::Tone::Info);
    appendLog(uiText("训练任务 %1 正在等待自动数据快照 %2。").arg(pending.taskId.left(8), createdTaskId.left(8)));
    updateRecentTasks();
}

void MainWindow::configureTable(QTableWidget* table) const
{
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->setShowGrid(false);
}

void MainWindow::updateProjectSummary()
{
    const bool hasProject = !currentProjectPath_.isEmpty() && repository_.isOpen();
    if (projectConsoleStatusLabel_) {
        projectConsoleStatusLabel_->setText(hasProject
            ? uiText("已打开：%1").arg(currentProjectName_)
            : uiText("未打开项目。"));
    }
    if (projectPathSummaryLabel_) {
        projectPathSummaryLabel_->setText(hasProject
            ? QDir::toNativeSeparators(currentProjectPath_)
            : uiText("未打开"));
    }
    if (projectSqliteSummaryLabel_) {
        projectSqliteSummaryLabel_->setText(hasProject ? uiText("已连接") : uiText("未连接"));
    }

    int datasetCount = 0;
    int taskCount = 0;
    int exportCount = 0;
    if (hasProject) {
        QString error;
        datasetCount = repository_.recentDatasets(200, &error).size();
        taskCount = repository_.recentTasks(200, &error).size();
        exportCount = repository_.recentExports(200, &error).size();
    }
    if (projectDatasetSummaryLabel_) {
        projectDatasetSummaryLabel_->setText(QString::number(datasetCount));
    }
    if (projectTaskSummaryLabel_) {
        projectTaskSummaryLabel_->setText(QString::number(taskCount));
    }
    if (projectExportSummaryLabel_) {
        projectExportSummaryLabel_->setText(QString::number(exportCount));
    }
}

void MainWindow::updatePluginSummary()
{
    const QVector<aitrain::IModelPlugin*> plugins = pluginManager_.plugins();
    QStringList datasetFormats;
    QStringList exportFormats;
    int gpuPlugins = 0;
    for (auto* plugin : plugins) {
        const aitrain::PluginManifest manifest = plugin->manifest();
        if (manifest.requiresGpu) {
            ++gpuPlugins;
        }
        datasetFormats.append(manifest.datasetFormats);
        exportFormats.append(manifest.exportFormats);
    }

    if (pluginConsoleStatusLabel_) {
        pluginConsoleStatusLabel_->setText(plugins.isEmpty()
            ? uiText("未加载插件，请检查插件目录。")
            : uiText("已加载 %1 个插件。").arg(plugins.size()));
    }
    if (pluginSearchPathLabel_) {
        pluginSearchPathLabel_->setText(uiText("插件搜索路径：%1").arg(pluginSearchPaths().join(QStringLiteral(" | "))));
    }
    if (pluginCountSummaryLabel_) {
        pluginCountSummaryLabel_->setText(QString::number(plugins.size()));
    }
    if (pluginDatasetFormatSummaryLabel_) {
        pluginDatasetFormatSummaryLabel_->setText(QString::number(uniqueStringCount(datasetFormats)));
        pluginDatasetFormatSummaryLabel_->setToolTip(compactListSummary(datasetFormats, 12));
    }
    if (pluginExportFormatSummaryLabel_) {
        pluginExportFormatSummaryLabel_->setText(QString::number(uniqueStringCount(exportFormats)));
        pluginExportFormatSummaryLabel_->setToolTip(compactListSummary(exportFormats, 12));
    }
    if (pluginGpuSummaryLabel_) {
        pluginGpuSummaryLabel_->setText(QString::number(gpuPlugins));
    }
}

void MainWindow::updateEnvironmentSummary()
{
    int ok = 0;
    int warning = 0;
    int missing = 0;
    int blocked = 0;
    int unchecked = 0;
    if (environmentTable_) {
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            const QString state = environmentTable_->item(row, 1) ? environmentTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
            if (state == QStringLiteral("ok")) {
                ++ok;
            } else if (state == QStringLiteral("hardware-blocked")) {
                ++warning;
                ++blocked;
            } else if (state == QStringLiteral("warning")) {
                ++warning;
            } else if (state == QStringLiteral("missing")) {
                ++missing;
            } else {
                ++unchecked;
            }
        }
    }
    if (environmentOkSummaryLabel_) {
        environmentOkSummaryLabel_->setText(QString::number(ok));
    }
    if (environmentWarningSummaryLabel_) {
        environmentWarningSummaryLabel_->setText(QString::number(warning));
    }
    if (environmentMissingSummaryLabel_) {
        environmentMissingSummaryLabel_->setText(QString::number(missing));
    }
    if (environmentUncheckedSummaryLabel_) {
        environmentUncheckedSummaryLabel_->setText(QString::number(unchecked));
    }
    if (environmentConsoleStatusLabel_) {
        if (missing > 0) {
            environmentConsoleStatusLabel_->setText(uiText("发现 %1 项缺失，相关能力会被阻塞。").arg(missing));
        } else if (warning > 0) {
            environmentConsoleStatusLabel_->setText(
                blocked > 0
                    ? uiText("发现 %1 项警告（其中 %2 项硬件受限），可继续但需要关注。").arg(warning).arg(blocked)
                    : uiText("发现 %1 项警告，可继续但需要关注。").arg(warning));
        } else if (unchecked > 0) {
            environmentConsoleStatusLabel_->setText(uiText("尚有 %1 项未检测。").arg(unchecked));
        } else {
            environmentConsoleStatusLabel_->setText(uiText("环境自检通过。"));
        }
    }
}

void MainWindow::updateDashboardSummary()
{
    const bool hasProject = !currentProjectPath_.isEmpty() && repository_.isOpen();
    if (dashboardProjectValue_) {
        dashboardProjectValue_->setText(hasProject ? currentProjectName_ : uiText("未打开"));
    }
    if (projectLabel_) {
        projectLabel_->setText(hasProject
            ? uiText("当前项目：%1").arg(QDir::toNativeSeparators(currentProjectPath_))
            : uiText("未打开项目。先创建或打开本地项目，后续数据集、任务和模型产物都会写入项目目录。"));
    }

    int datasetCount = 0;
    int validDatasetCount = 0;
    int taskCount = 0;
    int exportCount = 0;
    int modelVersionCount = 0;
    if (hasProject) {
        QString error;
        const QVector<aitrain::DatasetRecord> datasets = repository_.recentDatasets(200, &error);
        datasetCount = datasets.size();
        for (const aitrain::DatasetRecord& dataset : datasets) {
            if (dataset.validationStatus == QStringLiteral("valid")) {
                ++validDatasetCount;
            }
        }
        taskCount = repository_.recentTasks(200, &error).size();
        exportCount = repository_.recentExports(200, &error).size();
        modelVersionCount = repository_.recentModelVersions(200, &error).size();
    }

    if (dashboardDatasetValue_) {
        dashboardDatasetValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(validDatasetCount).arg(datasetCount)
            : QStringLiteral("0"));
    }
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(taskCount));
    }
    if (dashboardModelValue_) {
        dashboardModelValue_->setText(hasProject
            ? QStringLiteral("%1 / %2").arg(modelVersionCount).arg(exportCount)
            : QStringLiteral("0"));
    }
    if (dashboardPluginValue_) {
        dashboardPluginValue_->setText(QString::number(pluginManager_.plugins().size()));
    }

    QString environmentText = uiText("待检测");
    if (environmentTable_ && environmentTable_->rowCount() > 0) {
        bool hasMissing = false;
        bool hasWarning = false;
        bool hasChecked = false;
        for (int row = 0; row < environmentTable_->rowCount(); ++row) {
            const QString state = environmentTable_->item(row, 1) ? environmentTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
            hasChecked = hasChecked || !state.isEmpty();
            hasMissing = hasMissing || state == QStringLiteral("missing");
            hasWarning = hasWarning || state == QStringLiteral("warning") || state == QStringLiteral("hardware-blocked");
        }
        if (hasChecked) {
            environmentText = hasMissing ? uiText("缺失")
                : (hasWarning ? uiText("警告") : uiText("通过"));
        }
    }
    if (dashboardEnvironmentValue_) {
        dashboardEnvironmentValue_->setText(environmentText);
    }
    updateProjectSummary();
    updatePluginSummary();
    updateEnvironmentSummary();

    if (dashboardNextStepLabel_) {
        QString nextStep;
        if (!hasProject) {
            nextStep = uiText("先创建或打开一个本地项目。项目目录会集中保存数据集索引、任务历史、训练报告和模型产物。");
        } else if (validDatasetCount == 0) {
            nextStep = uiText("下一步：导入并校验 detection、segmentation 或 OCR Rec 数据集。只有通过校验的数据集会进入训练主流程。");
        } else if (taskCount == 0) {
            nextStep = uiText("下一步：进入训练实验，选择已校验数据集。平台会按任务类型优先选择官方 YOLO / OCR 后端。");
        } else if (exportCount == 0) {
            nextStep = uiText("下一步：在任务与产物中查看 checkpoint / report / ONNX，注册到模型库后再做评估、基准、导出或推理验证。");
        } else {
            nextStep = uiText("项目已具备可复验闭环：数据集、任务历史和模型导出均已记录。可继续运行推理验证或追加实验。");
        }
        dashboardNextStepLabel_->setText(nextStep);
    }
}

void MainWindow::updateTrainingSelectionSummary()
{
    const QString datasetPath = !currentDatasetPath_.isEmpty()
        ? currentDatasetPath_
        : QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
    const QString datasetFormat = !currentDatasetFormat_.isEmpty()
        ? currentDatasetFormat_
        : currentDatasetFormat();
    const QString state = currentDatasetValid_ ? uiText("已校验") : uiText("待校验");
    const QString pathText = datasetPath.isEmpty() ? uiText("未选择") : QDir::toNativeSeparators(datasetPath);
    QString snapshotText = uiText("快照：未选择数据集");
    if (!datasetPath.isEmpty() && repository_.isOpen()) {
        QString error;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(datasetPath, &error);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &error);
        snapshotText = snapshot.id > 0
            ? uiText("快照：#%1 | %2 文件 | hash %3 | %4")
                .arg(snapshot.id)
                .arg(snapshot.fileCount)
                .arg(snapshot.contentHash.left(12))
                .arg(QDir::toNativeSeparators(snapshot.manifestPath))
            : uiText("快照：暂无，启动训练时将自动创建。");
    }

    if (trainingDatasetSummaryLabel_) {
        trainingDatasetSummaryLabel_->setText(datasetPath.isEmpty()
            ? uiText("当前数据集：未选择。请先在数据集页导入并通过校验。")
            : uiText("当前数据集：%1 | %2 | %3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state, pathText, snapshotText));
    }
    if (datasetDetailLabel_) {
        datasetDetailLabel_->setText(datasetPath.isEmpty()
            ? uiText("选择或导入数据集后显示格式、样本数、校验状态和最近报告。")
            : uiText("格式：%1 | 状态：%2 | 路径：%3\n%4")
                .arg(datasetFormatLabel(datasetFormat), state, pathText, snapshotText));
    }
    if (trainingBackendHintLabel_ && trainingBackendCombo_) {
        trainingBackendHintLabel_->setText(trainingBackendDescription(trainingBackendCombo_->currentData().toString()));
    }
    if (trainingRunSummaryLabel_) {
        const QString backend = trainingBackendCombo_
            ? trainingBackendCombo_->currentData().toString()
            : defaultBackendForTask(currentTaskType());
        const QString model = modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString();
        trainingRunSummaryLabel_->setText(uiText("运行摘要：%1 | 后端 %2 | 模型 %3 | epoch %4 / batch %5 / image %6")
            .arg(taskTypeLabel(currentTaskType()),
                backend.isEmpty() ? uiText("未选择") : backend,
                model.isEmpty() ? uiText("默认") : model,
                epochsEdit_ ? epochsEdit_->text() : QStringLiteral("-"),
                batchEdit_ ? batchEdit_->text() : QStringLiteral("-"),
                imageSizeEdit_ ? imageSizeEdit_->text() : QStringLiteral("-")));
    }
}

void MainWindow::refreshTrainingDefaults()
{
    if (!trainingBackendCombo_ || !modelPresetCombo_) {
        updateTrainingSelectionSummary();
        return;
    }

    const QString datasetFormat = !currentDatasetFormat_.isEmpty()
        ? currentDatasetFormat_
        : currentDatasetFormat();
    QString preferredPlugin;
    QString preferredTask;
    QString preferredBackend;

    if (datasetFormat == QStringLiteral("yolo_detection") || datasetFormat == QStringLiteral("yolo_txt")) {
        preferredPlugin = QStringLiteral("com.aitrain.plugins.yolo_native");
        preferredTask = QStringLiteral("detection");
        preferredBackend = QStringLiteral("ultralytics_yolo_detect");
    } else if (datasetFormat == QStringLiteral("yolo_segmentation")) {
        preferredPlugin = QStringLiteral("com.aitrain.plugins.yolo_native");
        preferredTask = QStringLiteral("segmentation");
        preferredBackend = QStringLiteral("ultralytics_yolo_segment");
    } else if (datasetFormat == QStringLiteral("paddleocr_det")) {
        preferredPlugin = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        preferredTask = QStringLiteral("ocr_detection");
        preferredBackend = QStringLiteral("paddleocr_det_official");
    } else if (datasetFormat == QStringLiteral("paddleocr_rec")) {
        preferredPlugin = QStringLiteral("com.aitrain.plugins.ocr_rec_native");
        preferredTask = QStringLiteral("ocr_recognition");
        preferredBackend = QStringLiteral("paddleocr_rec");
    }

    if (!preferredPlugin.isEmpty() && pluginCombo_) {
        QSignalBlocker block(pluginCombo_);
        setComboCurrentData(pluginCombo_, preferredPlugin);
    }

    if (taskTypeCombo_) {
        const QString currentTask = currentTaskType();
        QSignalBlocker block(taskTypeCombo_);
        taskTypeCombo_->clear();
        auto* plugin = pluginCombo_ ? pluginManager_.pluginById(pluginCombo_->currentData().toString()) : nullptr;
        if (plugin) {
            addTaskTypeItems(taskTypeCombo_, plugin->manifest().taskTypes);
        }
        const QString targetTask = preferredTask.isEmpty() ? currentTask : preferredTask;
        const int taskIndex = taskTypeCombo_->findData(targetTask);
        if (taskIndex >= 0) {
            taskTypeCombo_->setCurrentIndex(taskIndex);
        } else if (taskTypeCombo_->count() > 0) {
            taskTypeCombo_->setCurrentIndex(0);
        }
    }

    if (preferredBackend.isEmpty()) {
        preferredBackend = defaultBackendForTask(currentTaskType());
    }
    {
        QSignalBlocker block(trainingBackendCombo_);
        setComboCurrentData(trainingBackendCombo_, preferredBackend);
    }
    modelPresetCombo_->setCurrentText(defaultModelForBackend(trainingBackendCombo_->currentData().toString()));
    updateTrainingSelectionSummary();
}
