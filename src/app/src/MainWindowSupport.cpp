#include "MainWindowSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"

#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QRegularExpression>
#include <QSizePolicy>
#include <QStandardPaths>
#include <QVBoxLayout>

namespace aitrain_app {

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

void allowLabelToShrink(QLabel* label)
{
    if (!label) {
        return;
    }
    label->setMinimumWidth(0);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
}

QString compactPathForStatus(const QString& path, int maxChars)
{
    const QString nativePath = QDir::toNativeSeparators(path);
    if (nativePath.size() <= maxChars) {
        return nativePath;
    }
    return QStringLiteral("...") + nativePath.right(qMax(0, maxChars - 3));
}

QString compactTextForStatus(const QString& text, int maxChars)
{
    if (text.size() <= maxChars) {
        return text;
    }
    const int head = qMax(12, maxChars / 2 - 3);
    const int tail = qMax(12, maxChars - head - 5);
    return text.left(head) + QStringLiteral(" ... ") + text.right(tail);
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

QString defaultProjectPathSettingsKey()
{
    return QStringLiteral("settings/defaultProjectPath");
}

QString taskTypeLabel(const QString& taskType)
{
    if (taskType == QStringLiteral("detection")) {
        return uiText("检测");
    }
    if (taskType == QStringLiteral("segmentation")) {
        return uiText("分割");
    }
    if (taskType == QStringLiteral("obb_detection") || taskType == QStringLiteral("obb")) {
        return uiText("OBB 旋转框检测");
    }
    if (taskType == QStringLiteral("semantic_segmentation")) {
        return uiText("语义分割");
    }
    if (taskType == QStringLiteral("anomaly_detection")) {
        return uiText("异常检测");
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
    if (backend == QStringLiteral("ultralytics_yolo_obb")) {
        return uiText("Ultralytics YOLO OBB（官方）");
    }
    if (backend == QStringLiteral("smp_semantic_segmentation")) {
        return uiText("SMP 语义分割（官方）");
    }
    if (backend == QStringLiteral("anomalib_patchcore")) {
        return uiText("Anomalib PatchCore（官方）");
    }
    if (backend == QStringLiteral("anomalib_efficientad")) {
        return uiText("Anomalib EfficientAD（官方）");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return uiText("PaddleOCR Det（官方/隔离环境）");
    }
    if (backend == QStringLiteral("paddleocr_rec_official")) {
        return uiText("PaddleOCR Rec（PP-OCRv4/v5/v6 官方）");
    }
    if (backend == QStringLiteral("paddleocr_system_official")) {
        return uiText("PaddleOCR System 推理（官方）");
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
        QStringLiteral("imageF1"),
        QStringLiteral("imageAUROC"),
        QStringLiteral("pixelAUROC"),
        QStringLiteral("threshold"),
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

QLabel* inferenceBadge(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("InferenceBadge"));
    label->setAlignment(Qt::AlignCenter);
    return label;
}

QFrame* createInferenceStep(const QString& index, const QString& title, const QString& caption)
{
    auto* frame = new QFrame;
    frame->setObjectName(QStringLiteral("InferenceStep"));
    frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    auto* layout = new QHBoxLayout(frame);
    layout->setContentsMargins(10, 8, 10, 8);
    layout->setSpacing(9);

    auto* indexLabel = new QLabel(index);
    indexLabel->setObjectName(QStringLiteral("InferenceStepIndex"));
    indexLabel->setAlignment(Qt::AlignCenter);
    indexLabel->setFixedSize(24, 24);

    auto* textBlock = new QWidget;
    auto* textLayout = new QVBoxLayout(textBlock);
    textLayout->setContentsMargins(0, 0, 0, 0);
    textLayout->setSpacing(1);
    auto* titleLabel = new QLabel(title);
    titleLabel->setObjectName(QStringLiteral("InferenceStepTitle"));
    auto* captionLabel = new QLabel(caption);
    captionLabel->setObjectName(QStringLiteral("InferenceStepCaption"));
    captionLabel->setWordWrap(true);
    textLayout->addWidget(titleLabel);
    textLayout->addWidget(captionLabel);

    layout->addWidget(indexLabel);
    layout->addWidget(textBlock, 1);
    return frame;
}

QFrame* createInferenceCapability(const QString& title, const QString& caption)
{
    auto* frame = new QFrame;
    frame->setObjectName(QStringLiteral("InferenceCapability"));
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(10, 8, 10, 8);
    layout->setSpacing(3);
    auto* titleLabel = new QLabel(title);
    titleLabel->setObjectName(QStringLiteral("InferenceCapabilityTitle"));
    auto* captionLabel = new QLabel(caption);
    captionLabel->setObjectName(QStringLiteral("InferenceCapabilityCaption"));
    captionLabel->setWordWrap(true);
    layout->addWidget(titleLabel);
    layout->addWidget(captionLabel);
    return frame;
}

QFrame* createWorkbenchHeader(
    const QString& kickerText,
    const QString& titleText,
    const QString& subtitleText,
    QPushButton* actionButton,
    const QStringList& badges)
{
    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    auto* headerRoot = new QHBoxLayout(headerPanel);
    headerRoot->setContentsMargins(10, 6, 10, 6);
    headerRoot->setSpacing(10);
    Q_UNUSED(kickerText)
    Q_UNUSED(subtitleText)
    auto* contextLabel = new QLabel(titleText);
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarContext"));
    headerRoot->addWidget(contextLabel);

    if (!badges.isEmpty()) {
        auto* badgeRow = new QWidget;
        auto* badgeLayout = new QHBoxLayout(badgeRow);
        badgeLayout->setContentsMargins(0, 0, 0, 0);
        badgeLayout->setSpacing(7);
        for (const QString& badge : badges) {
            badgeLayout->addWidget(inferenceBadge(badge));
        }
        badgeLayout->addStretch();
        headerRoot->addWidget(badgeRow, 1);
    } else {
        headerRoot->addStretch(1);
    }
    if (actionButton) {
        headerRoot->addWidget(actionButton);
    }
    return headerPanel;
}

void setInferenceOverlayText(QLabel* label, const QString& text)
{
    if (!label) {
        return;
    }
    label->clear();
    label->setText(text);
}

void loadInferenceOverlay(QLabel* label, const QString& path)
{
    if (!label) {
        return;
    }
    QPixmap overlay(path);
    if (overlay.isNull()) {
        setInferenceOverlayText(label, uiText("推理 overlay 加载失败"));
        return;
    }
    QSize targetSize = label->size().boundedTo(QSize(900, 560));
    if (targetSize.width() < 160 || targetSize.height() < 120) {
        targetSize = QSize(720, 420);
    }
    label->setPixmap(overlay.scaled(
        targetSize,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation));
}

QString taskStateLabel(aitrain::TaskState state)
{
    switch (state) {
    case aitrain::TaskState::Queued: return uiText("排队中");
    case aitrain::TaskState::Running: return uiText("运行中");
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
    if (taskType == QStringLiteral("obb_detection") || taskType == QStringLiteral("obb")) {
        return uiText("OBB 旋转框检测");
    }
    if (taskType == QStringLiteral("semantic_segmentation")) {
        return uiText("语义分割");
    }
    if (taskType == QStringLiteral("anomaly_detection")) {
        return uiText("异常检测");
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
    if (format == QStringLiteral("yolo_obb")) {
        return uiText("YOLO OBB 旋转框");
    }
    if (format == QStringLiteral("semantic_segmentation_mask")) {
        return uiText("语义分割 Mask PNG");
    }
    if (format == QStringLiteral("anomaly_folder")) {
        return uiText("异常检测 Folder");
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
    if (taskType == QStringLiteral("detection")) {
        return QStringLiteral("ultralytics_yolo_detect");
    }
    if (taskType == QStringLiteral("segmentation")) {
        return QStringLiteral("ultralytics_yolo_segment");
    }
    if (taskType == QStringLiteral("obb_detection") || taskType == QStringLiteral("obb")) {
        return QStringLiteral("ultralytics_yolo_obb");
    }
    if (taskType == QStringLiteral("semantic_segmentation")) {
        return QStringLiteral("smp_semantic_segmentation");
    }
    if (taskType == QStringLiteral("anomaly_detection")) {
        return QStringLiteral("anomalib_patchcore");
    }
    if (taskType == QStringLiteral("ocr_detection")) {
        return QStringLiteral("paddleocr_det_official");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return QStringLiteral("paddleocr_rec_official");
    }
    return {};
}

QStringList modelPresetItemsForBackend(const QString& backend)
{
    const aitrain::BackendDescriptor selected =
        aitrain::BuiltinCapabilityRegistry::instance().backend(backend);
    if (!selected.id.isEmpty()) {
        return selected.modelPresets;
    }

    QStringList presets;
    for (const aitrain::BackendDescriptor& descriptor :
        aitrain::BuiltinCapabilityRegistry::instance().backends()) {
        presets.append(descriptor.modelPresets);
    }
    presets.removeDuplicates();
    return presets;
}

bool yoloModelPresetMatchesBackend(const QString& modelPreset, const QString& backend)
{
    const QString model = modelPreset.trimmed().toLower();
    const QString normalizedBackend = backend.trimmed().toLower();
    if (model.isEmpty()
        || !(normalizedBackend.startsWith(QStringLiteral("ultralytics_yolo")))) {
        return true;
    }

    const bool modelLooksSegment = model.contains(QStringLiteral("-seg."));
    const bool modelLooksObb = model.contains(QStringLiteral("-obb."));
    if (normalizedBackend == QStringLiteral("ultralytics_yolo_obb")) {
        if (modelLooksObb || model.contains(QStringLiteral("-obb"))) {
            return true;
        }
        return !modelLooksSegment && !model.contains(QStringLiteral("yolo"));
    }
    if (normalizedBackend == QStringLiteral("ultralytics_yolo_segment")) {
        return modelLooksSegment && !modelLooksObb;
    }
    if (normalizedBackend == QStringLiteral("ultralytics_yolo_detect")
        || normalizedBackend == QStringLiteral("ultralytics_yolo")) {
        return !modelLooksSegment && !modelLooksObb;
    }
    return true;
}

struct Yolo26TargetedMatrixStatus {
    bool trainingAccepted = false;
    bool ncnnUnsupported = true;
    QString summaryPath;
    QString pythonExecutable;
};

bool deploymentStatusIsExplicit(const QString& status)
{
    return status == QStringLiteral("passed")
        || status == QStringLiteral("blocked")
        || status == QStringLiteral("failed")
        || status == QStringLiteral("hardware-blocked");
}

Yolo26TargetedMatrixStatus yolo26TargetedMatrixStatus()
{
    Yolo26TargetedMatrixStatus accepted;
    QStringList candidates;
    const QString envPath = QString::fromLocal8Bit(qgetenv("AITRAIN_YOLO26_MATRIX_SUMMARY")).trimmed();
    if (!envPath.isEmpty()) {
        candidates.append(envPath);
    }
    candidates.append(QDir(QDir::currentPath()).filePath(QStringLiteral(".deps/phase-yolo26-model-matrix/yolo26_model_matrix_summary.json")));
    candidates.append(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../.deps/phase-yolo26-model-matrix/yolo26_model_matrix_summary.json")));

    for (const QString& candidate : candidates) {
        const QJsonObject summary = readJsonObjectFile(QDir::cleanPath(candidate));
        if (summary.isEmpty()
            || !summary.value(QStringLiteral("ok")).toBool()
            || summary.value(QStringLiteral("status")).toString() != QStringLiteral("passed")
            || summary.value(QStringLiteral("mode")).toString() != QStringLiteral("full")) {
            continue;
        }
        const QJsonArray results = summary.value(QStringLiteral("results")).toArray();
        int requiredCount = 0;
        bool allRequiredPassed = true;
        bool allOnnxPassed = true;
        bool allTensorRtExplicit = true;
        for (const QJsonValue& value : results) {
            const QJsonObject result = value.toObject();
            if (!result.value(QStringLiteral("required")).toBool()) {
                continue;
            }
            ++requiredCount;
            if (result.value(QStringLiteral("status")).toString() != QStringLiteral("passed")) {
                allRequiredPassed = false;
                break;
            }
            const QJsonObject deployments = result.value(QStringLiteral("deployments")).toObject();
            const QString onnxStatus = deployments.value(QStringLiteral("onnx")).toObject().value(QStringLiteral("status")).toString();
            if (onnxStatus != QStringLiteral("passed")) {
                allOnnxPassed = false;
                break;
            }

            const QString tensorrtStatus = deployments.value(QStringLiteral("tensorrt")).toObject().value(QStringLiteral("status")).toString();
            if (!deploymentStatusIsExplicit(tensorrtStatus)) {
                allTensorRtExplicit = false;
                break;
            }
            if (!allRequiredPassed) {
                break;
            }
        }
        QString pythonExecutable = QDir::fromNativeSeparators(summary.value(QStringLiteral("yolo26Python")).toString().trimmed());
        if (!pythonExecutable.isEmpty() && QFileInfo(pythonExecutable).isRelative()) {
            pythonExecutable = QFileInfo(candidate).absoluteDir().filePath(pythonExecutable);
        }
        pythonExecutable = QDir::cleanPath(pythonExecutable);
        if (pythonExecutable.isEmpty() || !QFileInfo::exists(pythonExecutable)) {
            continue;
        }

        if (requiredCount >= 20
            && allRequiredPassed
            && allOnnxPassed
            && allTensorRtExplicit) {
            accepted.trainingAccepted = true;
            accepted.ncnnUnsupported = true;
            accepted.summaryPath = QDir::cleanPath(candidate);
            accepted.pythonExecutable = pythonExecutable;
            return accepted;
        }
    }
    return accepted;
}

QString defaultModelForBackend(const QString& backend)
{
    if (backend == QStringLiteral("smp_semantic_segmentation")) {
        return QStringLiteral("smp_unet_resnet34");
    }
    if (backend == QStringLiteral("anomalib_patchcore")) {
        return QStringLiteral("anomalib_patchcore_wide_resnet50_2");
    }
    if (backend == QStringLiteral("anomalib_efficientad")) {
        return QStringLiteral("anomalib_efficientad_s");
    }
    if (backend == QStringLiteral("ultralytics_yolo_segment")) {
        return QStringLiteral("yolov8n-seg.yaml");
    }
    if (backend == QStringLiteral("ultralytics_yolo_obb")) {
        return QStringLiteral("yolo11n-obb.pt");
    }
    if (backend == QStringLiteral("ultralytics_yolo_detect") || backend == QStringLiteral("ultralytics_yolo")) {
        return QStringLiteral("yolov8n.yaml");
    }
    if (backend == QStringLiteral("paddleocr_rec_official") || backend == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return QStringLiteral("PP-OCRv5_mobile_rec");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return QStringLiteral("PP-OCRv5_mobile_det");
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
    if (backend == QStringLiteral("ultralytics_yolo_obb")) {
        return uiText("当前模型能力：官方 Ultralytics YOLO OBB 旋转框检测。适合 9 列四点 YOLO OBB 数据，输出 best.pt、ONNX、官方 val 指标，并支持 ONNX Runtime 旋转框 JSON、overlay、benchmark 与部署验证；NCNN 不属于 OBB v1。");
    }
    if (backend == QStringLiteral("smp_semantic_segmentation")) {
        return uiText("当前模型能力：SMP 专用语义分割。适合 Mask PNG class-id 数据，输出 best.pt、best.onnx、训练/评估报告，并支持 ONNX Runtime 单图 mask overlay 和 benchmark；SMP 不需要 NCNN/TensorRT 导出。");
    }
    if (backend == QStringLiteral("anomalib_patchcore")) {
        return uiText("当前模型能力：Anomalib PatchCore 异常检测。适合 anomaly_folder/MVTec 兼容目录，输出 checkpoint、anomaly_sidecar、训练/评估报告和热力图；v1 部署边界是 Worker-managed Python/Anomalib，不是 C++ ONNX/TensorRT/NCNN runtime。");
    }
    if (backend == QStringLiteral("anomalib_efficientad")) {
        return uiText("当前模型能力：Anomalib EfficientAD 异常检测。需要 imagenetDir、AITRAIN_ANOMALIB_IMAGENET_DIR 或 .deps/anomalib/imagenette；Anomalib 2.5 下训练 batchSize 固定为 1，缺失外部数据时 blocked，不自动下载。");
    }
    if (backend == QStringLiteral("paddleocr_rec_official") || backend == QStringLiteral("paddleocr_ppocrv4_rec")) {
        return uiText("当前模型能力：官方 PaddleOCR PP-OCRv4/v5/v6 Rec 适配器。通过模型预设选择版本，适合隔离 OCR Python 环境，记录 train/export/predict 命令、checkpoint、inference model 和官方预测报告。");
    }
    if (backend == QStringLiteral("paddleocr_det_official")) {
        return uiText("当前模型能力：官方 PaddleOCR PP-OCRv4/v5/v6 Det 适配器。通过模型预设选择版本，适合 PaddleOCR 原生 det_gt.txt 数据，输出官方配置、checkpoint、inference model 和报告。");
    }
    if (backend == QStringLiteral("paddleocr_system_official")) {
        return uiText("当前模型能力：官方 PaddleOCR 端到端推理编排。使用已导出的 Det/Rec inference model 调用 predict_system.py；本阶段不做 C++ DB 后处理。");
    }
    return uiText("当前模型能力：通过 Worker 执行，产物、指标和失败原因会写入任务历史。");
}

namespace {
bool paddleOcrOfficialRepoConfigured()
{
    const QString repo = QString::fromLocal8Bit(qgetenv("AITRAIN_PADDLEOCR_REPO")).trimmed();
    const QString legacyRepo = QString::fromLocal8Bit(qgetenv("AITRAIN_PADDLEOCR_SOURCE_ROOT")).trimmed();
    if (!repo.isEmpty() || !legacyRepo.isEmpty()) {
        return true;
    }

    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir(appDir).absoluteFilePath(QStringLiteral("python_env/PaddleOCR")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../python_env/PaddleOCR")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/repos/PaddleOCR")),
        QDir::current().absoluteFilePath(QStringLiteral(".deps/PaddleOCR")),
        QDir::current().absoluteFilePath(QStringLiteral("python_env/PaddleOCR"))
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(QDir(candidate).filePath(QStringLiteral("tools/train.py")))) {
            return true;
        }
    }
    return false;
}

QString expectedTrainingTaskForDatasetFormat(const QString& format)
{
    const auto& registry = aitrain::BuiltinCapabilityRegistry::instance();
    for (const aitrain::CapabilityDescriptor& capability : registry.capabilities()) {
        for (const QString& taskType : capability.taskTypes) {
            if (registry.datasetFormatsForTask(taskType).contains(format)) {
                return taskType;
            }
        }
    }
    return {};
}

bool isTrainingBackendCompatible(const QString& format, const QString& backend)
{
    const QString taskType = expectedTrainingTaskForDatasetFormat(format);
    return !taskType.isEmpty()
        && aitrain::BuiltinCapabilityRegistry::instance()
            .backendsForTask(taskType, format)
            .contains(backend.trimmed());
}
} // namespace

QJsonObject trainingPreflightReport(
    const QString& datasetPath,
    const QString& datasetFormat,
    bool datasetReady,
    const QString& datasetSnapshotManifest,
    const QString& taskType,
    const QString& backend,
    const QString& modelPreset,
    int epochs,
    int batchSize,
    int imageSize)
{
    QJsonArray blockers;
    QJsonArray warnings;
    QJsonArray nextActions;

    if (datasetPath.trimmed().isEmpty()) {
        blockers.append(QStringLiteral("dataset_missing"));
    }
    if (datasetFormat.trimmed().isEmpty()) {
        blockers.append(QStringLiteral("dataset_format_missing"));
    }
    if (!datasetReady) {
        blockers.append(QStringLiteral("dataset_not_validated"));
        nextActions.append(QStringLiteral("validate_dataset"));
        nextActions.append(QStringLiteral("run_dataset_quality_report"));
    }

    const QString expectedTask = expectedTrainingTaskForDatasetFormat(datasetFormat);
    if (!expectedTask.isEmpty() && taskType != expectedTask) {
        blockers.append(QStringLiteral("task_type_dataset_mismatch"));
    }
    if (!backend.isEmpty() && !isTrainingBackendCompatible(datasetFormat, backend)) {
        blockers.append(QStringLiteral("backend_dataset_mismatch"));
    }
    if (!yoloModelPresetMatchesBackend(modelPreset, backend)) {
        blockers.append(QStringLiteral("yolo_model_backend_mismatch"));
        nextActions.append(QStringLiteral("select_matching_yolo_model_preset"));
    }
    const QString normalizedBackend = backend.trimmed().toLower();
    const QString normalizedModel = modelPreset.trimmed().toLower();
    const bool efficientAdBackend = normalizedBackend == QStringLiteral("anomalib_efficientad");
    const int effectiveBatchSize = efficientAdBackend ? 1 : batchSize;
    const Yolo26TargetedMatrixStatus yolo26Status = yolo26TargetedMatrixStatus();
    const bool yolo26Model = normalizedBackend.startsWith(QStringLiteral("ultralytics_yolo"))
        && normalizedModel.startsWith(QStringLiteral("yolo26"));
    if (yolo26Model && !yolo26Status.trainingAccepted) {
        blockers.append(QStringLiteral("yolo26_requires_targeted_compatibility_validation"));
        nextActions.append(QStringLiteral("run_yolo26_targeted_matrix_after_full_lifecycle"));
    }
    if (yolo26Model
        && yolo26Status.trainingAccepted
        && yolo26Status.ncnnUnsupported) {
        warnings.append(QStringLiteral("yolo26_ncnn_export_unsupported"));
        nextActions.append(QStringLiteral("use_yolo26_onnx_or_tensorrt"));
    }
    if (normalizedBackend == QStringLiteral("ultralytics_yolo_segment")
        && normalizedModel.startsWith(QStringLiteral("yolo12"))
        && normalizedModel.endsWith(QStringLiteral("-seg.pt"))) {
        blockers.append(QStringLiteral("yolo12_seg_pt_missing_official_weight"));
        nextActions.append(QStringLiteral("use_yolo12_seg_yaml_or_wait_for_official_weight"));
    }
    if ((backend == QStringLiteral("paddleocr_det_official")
            || backend == QStringLiteral("paddleocr_rec_official")
            || backend == QStringLiteral("paddleocr_ppocrv4_rec"))
        && !paddleOcrOfficialRepoConfigured()) {
        blockers.append(QStringLiteral("paddleocr_repo_missing"));
        nextActions.append(QStringLiteral("set_AITRAIN_PADDLEOCR_REPO"));
        nextActions.append(QStringLiteral("use_isolated_ocr_python"));
    }
    if (epochs <= 0) {
        blockers.append(QStringLiteral("epochs_invalid"));
    }
    if (efficientAdBackend && batchSize != 1) {
        warnings.append(QStringLiteral("efficientad_batch_size_forced_to_1"));
        nextActions.append(QStringLiteral("use_efficientad_batch_size_1"));
    }
    if (effectiveBatchSize <= 0) {
        blockers.append(QStringLiteral("batch_size_invalid"));
    }
    if (imageSize <= 0) {
        blockers.append(QStringLiteral("image_size_invalid"));
    }
    if (datasetSnapshotManifest.trimmed().isEmpty()) {
        warnings.append(QStringLiteral("snapshot_missing_auto_create"));
    }
    if (modelPreset.trimmed().isEmpty()) {
        warnings.append(QStringLiteral("model_preset_missing"));
    }
    if (nextActions.isEmpty()) {
        if (blockers.isEmpty()) {
            nextActions.append(QStringLiteral("create_snapshot_if_missing"));
            nextActions.append(QStringLiteral("start_worker_training"));
        } else {
            nextActions.append(QStringLiteral("fix_preflight_blockers"));
        }
    }

    QJsonObject preflight;
    preflight.insert(QStringLiteral("schemaVersion"), 1);
    preflight.insert(QStringLiteral("kind"), QStringLiteral("training_preflight"));
    preflight.insert(QStringLiteral("status"), blockers.isEmpty() ? QStringLiteral("ready") : QStringLiteral("blocked"));
    preflight.insert(QStringLiteral("canStart"), blockers.isEmpty());
    preflight.insert(QStringLiteral("datasetPath"), datasetPath);
    preflight.insert(QStringLiteral("datasetFormat"), datasetFormat);
    preflight.insert(QStringLiteral("datasetReady"), datasetReady);
    preflight.insert(QStringLiteral("datasetSnapshotManifest"), datasetSnapshotManifest);
    preflight.insert(QStringLiteral("taskType"), taskType);
    preflight.insert(QStringLiteral("expectedTaskType"), expectedTask);
    preflight.insert(QStringLiteral("trainingBackend"), backend);
    preflight.insert(QStringLiteral("modelPreset"), modelPreset);
    if (yolo26Model && yolo26Status.trainingAccepted) {
        preflight.insert(QStringLiteral("yolo26MatrixSummary"), yolo26Status.summaryPath);
        preflight.insert(QStringLiteral("yolo26PythonExecutable"), yolo26Status.pythonExecutable);
    }
    preflight.insert(QStringLiteral("epochs"), epochs);
    preflight.insert(QStringLiteral("batchSize"), effectiveBatchSize);
    if (effectiveBatchSize != batchSize) {
        preflight.insert(QStringLiteral("requestedBatchSize"), batchSize);
    }
    preflight.insert(QStringLiteral("imageSize"), imageSize);
    preflight.insert(QStringLiteral("blockers"), blockers);
    preflight.insert(QStringLiteral("warnings"), warnings);
    preflight.insert(QStringLiteral("nextActions"), nextActions);
    return preflight;
}

QString trainingPreflightSummaryText(const QJsonObject& preflight)
{
    const QString status = preflight.value(QStringLiteral("status")).toString(QStringLiteral("blocked"));
    const int blockerCount = preflight.value(QStringLiteral("blockers")).toArray().size();
    const int warningCount = preflight.value(QStringLiteral("warnings")).toArray().size();
    QStringList parts;
    parts.append(QStringLiteral("preflight %1").arg(status));
    parts.append(QStringLiteral("blockers=%1").arg(blockerCount));
    parts.append(QStringLiteral("warnings=%1").arg(warningCount));
    const QString expectedTask = preflight.value(QStringLiteral("expectedTaskType")).toString();
    if (!expectedTask.isEmpty() && expectedTask != preflight.value(QStringLiteral("taskType")).toString()) {
        parts.append(QStringLiteral("expectedTask=%1").arg(expectedTask));
    }
    return parts.join(QStringLiteral(" | "));
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
        return uiText("需要配置 onnx2ncnn，输出 param/bin；部署验证需要 NCNN SDK/runtime 和样本图。");
    }
    if (format == QStringLiteral("tensorrt")) {
        return uiText("需要 RTX / SM 75+ 真机外部验收。");
    }
    return uiText("仅支持官方模型导出格式。");
}

QString compactListSummary(const QStringList& values, int maxItems)
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
    if (combo->currentIndex() == index) {
        return true;
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
        QDir(QDir::currentPath()).filePath(QStringLiteral(".deps/tools/annotation-tools/X-AnyLabeling/X-AnyLabeling.exe")),
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
        return uiText("状态：未检测到 X-AnyLabeling。请检查 PATH、环境变量或 .deps/tools/annotation-tools。");
    }
    return uiText("状态：已安装 | %1").arg(compactPathForStatus(program, 72));
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
    if (QFileInfo::exists(root.filePath(QStringLiteral("classes.txt")))
        && QDir(root.filePath(QStringLiteral("images/train"))).exists()
        && QDir(root.filePath(QStringLiteral("images/val"))).exists()
        && QDir(root.filePath(QStringLiteral("masks/train"))).exists()
        && QDir(root.filePath(QStringLiteral("masks/val"))).exists()) {
        return QStringLiteral("semantic_segmentation_mask");
    }
    if (QDir(root.filePath(QStringLiteral("train/good"))).exists()) {
        return QStringLiteral("anomaly_folder");
    }
    if (QDir(root.filePath(QStringLiteral("test/good"))).exists()
        && (QDir(root.filePath(QStringLiteral("ground_truth"))).exists()
            || QDir(root.filePath(QStringLiteral("masks/test/anomaly"))).exists())) {
        return QStringLiteral("anomaly_folder");
    }

    if (!QFileInfo::exists(root.filePath(QStringLiteral("data.yaml")))) {
        return QString();
    }
    bool yamlTaskObb = false;
    QFile yamlFile(root.filePath(QStringLiteral("data.yaml")));
    if (yamlFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QString yamlText = QString::fromUtf8(yamlFile.readAll()).toLower();
        yamlTaskObb = yamlText.contains(QRegularExpression(QStringLiteral("(?m)^\\s*task\\s*:\\s*obb\\b")));
    }
    const QString normalizedPath = QDir::fromNativeSeparators(QFileInfo(path).absoluteFilePath()).toLower();
    const bool pathSuggestsObb = normalizedPath.contains(QStringLiteral("obb")) || normalizedPath.contains(QStringLiteral("dota"));

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
                if (parts.size() == 9) {
                    return (yamlTaskObb || pathSuggestsObb) ? QStringLiteral("yolo_obb") : QString();
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

QString inferenceSummaryFromPredictions(const QString& predictionsPath, const QJsonObject& fallback)
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
    int resultCount = predictions.size();
    QString detail;
    if (!predictions.isEmpty()) {
        const QJsonObject first = predictions.at(0).toObject();
        if (taskType == QStringLiteral("semantic_segmentation")) {
            const QJsonObject pixelCounts = first.value(QStringLiteral("pixelCounts")).toObject();
            const QJsonArray classNames = first.value(QStringLiteral("classNames")).toArray();
            double totalPixels = 0.0;
            double foregroundPixels = 0.0;
            int activeClasses = 0;
            QStringList classParts;
            for (auto it = pixelCounts.constBegin(); it != pixelCounts.constEnd(); ++it) {
                const double count = it.value().toDouble();
                totalPixels += count;
                bool classOk = false;
                const int classId = it.key().toInt(&classOk);
                if (count <= 0.0 || it.key() == QStringLiteral("0")) {
                    continue;
                }
                foregroundPixels += count;
                ++activeClasses;
                if (classParts.size() < 3) {
                    QString className = classOk && classId >= 0 && classId < classNames.size()
                        ? classNames.at(classId).toString()
                        : QStringLiteral("class %1").arg(it.key());
                    if (className.trimmed().isEmpty()) {
                        className = QStringLiteral("class %1").arg(it.key());
                    }
                    classParts.append(QStringLiteral("%1 %2 px").arg(className).arg(static_cast<qint64>(count)));
                }
            }
            resultCount = activeClasses;
            detail = uiText("前景 %1 px / 总像素 %2")
                .arg(static_cast<qint64>(foregroundPixels))
                .arg(static_cast<qint64>(totalPixels));
            if (!classParts.isEmpty()) {
                detail.append(QStringLiteral("，%1").arg(classParts.join(QStringLiteral(", "))));
            }
        } else if (taskType == QStringLiteral("anomaly_detection")) {
            const double score = first.value(QStringLiteral("anomalyScore")).toDouble();
            const double threshold = first.value(QStringLiteral("threshold")).toDouble();
            const QString decision = first.value(QStringLiteral("decision")).toString(QStringLiteral("unknown")).toUpper();
            detail = uiText("%1，score %2 / threshold %3")
                .arg(decision)
                .arg(score, 0, 'f', 4)
                .arg(threshold, 0, 'f', 4);
            resultCount = 1;
        } else if (taskType == QStringLiteral("ocr_recognition")) {
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
        .arg(resultCount)
        .arg(detail)
        .arg(elapsedMs)
        .arg(nativePath);
}

} // namespace aitrain_app
