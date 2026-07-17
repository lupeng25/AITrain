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

} // namespace aitrain_app
