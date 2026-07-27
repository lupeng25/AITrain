#include "TrainingPage.h"

#include "MetricsWidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QTime>
#include <QTextEdit>

namespace {

QString comboValue(const QWidget* page, const char* name)
{
    const auto* combo = page->findChild<QComboBox*>(
        QString::fromLatin1(name));
    if (!combo) {
        return {};
    }
    const QString data = combo->currentData().toString().trimmed();
    return data.isEmpty() ? combo->currentText().trimmed() : data;
}

int integerValue(const QWidget* page, const char* name)
{
    const auto* edit = page->findChild<QLineEdit*>(
        QString::fromLatin1(name));
    return edit ? edit->text().toInt() : 0;
}

} // namespace

TrainingWorkspacePage::TrainingWorkspacePage(QWidget* parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("TrainingWorkspacePage"));
}

TrainingFormData TrainingWorkspacePage::formData() const
{
    TrainingFormData result;
    result.capabilityId = comboValue(this, "TrainingCapability");
    result.taskType = comboValue(this, "TrainingTaskType");
    result.backendId = comboValue(this, "TrainingBackend");
    result.modelPreset = comboValue(this, "TrainingModelPreset");
    result.epochs = integerValue(this, "TrainingEpochs");
    result.batchSize = integerValue(this, "TrainingBatchSize");
    result.imageSize = integerValue(this, "TrainingImageSize");
    result.gridSize = integerValue(this, "TrainingGridSize");
    if (const auto* check =
            findChild<QCheckBox*>(QStringLiteral("TrainingHorizontalFlip"))) {
        result.horizontalFlip = check->isChecked();
    }
    if (const auto* check =
            findChild<QCheckBox*>(QStringLiteral("TrainingColorJitter"))) {
        result.colorJitter = check->isChecked();
    }
    return result;
}

void TrainingWorkspacePage::setDatasetSummary(const QString& text)
{
    if (auto* label =
            findChild<QLabel*>(QStringLiteral("TrainingDatasetNote"))) {
        label->setText(text);
    }
}

void TrainingWorkspacePage::setBackendSummary(const QString& text)
{
    if (auto* label =
            findChild<QLabel*>(QStringLiteral("TrainingBackendHint"))) {
        label->setText(text);
    }
}

void TrainingWorkspacePage::setRunSummary(const QString& text)
{
    if (auto* label = findChild<QLabel*>(QStringLiteral("TrainingRunNote"))) {
        label->setText(text);
    }
}

void TrainingWorkspacePage::setDatasetSummaryToolTip(const QString& text)
{
    if (auto* label =
            findChild<QLabel*>(QStringLiteral("TrainingDatasetNote"))) {
        label->setToolTip(text);
    }
}

void TrainingWorkspacePage::setRunSummaryToolTip(const QString& text)
{
    if (auto* label = findChild<QLabel*>(QStringLiteral("TrainingRunNote"))) {
        label->setToolTip(text);
    }
}

void TrainingWorkspacePage::setBackendPanels(const QString& backendId)
{
    const QString backend = backendId.trimmed().toLower();
    const auto setPanel = [this, &backend](
                              const QString& name, bool matches) {
        if (auto* panel = findChild<QWidget*>(name)) {
            panel->setVisible(
                panel->property("advancedExpanded").toBool() && matches);
        }
    };
    setPanel(QStringLiteral("YoloOfficialArgsGroup"),
        backend.startsWith(QStringLiteral("ultralytics_yolo_")));
    setPanel(QStringLiteral("SmpSemanticArgsGroup"),
        backend == QStringLiteral("smp_semantic_segmentation"));
    setPanel(QStringLiteral("AnomalyDetectionArgsGroup"),
        backend.startsWith(QStringLiteral("anomalib_")));
    if (auto* caption = findChild<QLabel*>(
            QStringLiteral("TrainingLiveCaption_TrainingMapValue"))) {
        caption->setText(backend.startsWith(QStringLiteral("anomalib_"))
                ? QStringLiteral("Score/F1") : QStringLiteral("mAP"));
    }
}

void TrainingWorkspacePage::setProgress(int progress)
{
    if (auto* bar =
            findChild<QProgressBar*>(QStringLiteral("TrainingProgress"))) {
        bar->setValue(progress);
    }
}

void TrainingWorkspacePage::setPhase(const QString& text)
{
    if (auto* label =
            findChild<QLabel*>(QStringLiteral("TrainingPhaseStatus"))) {
        label->setText(text);
    }
}

void TrainingWorkspacePage::setLiveValue(
    const QString& objectName, const QString& value)
{
    if (auto* label = findChild<QLabel*>(objectName)) {
        label->setText(value);
    }
}

void TrainingWorkspacePage::addMetric(const QString& name, double value)
{
    if (auto* metrics = findChild<MetricsWidget*>()) {
        metrics->addMetric(name, value);
    }
}

void TrainingWorkspacePage::updateArtifact(
    const QString& artifactId, const QString& kind,
    const QString& relativePath)
{
    const QString normalizedKind = kind.toLower();
    const QString suffix = QFileInfo(relativePath).suffix().toLower();
    const QString value = QStringLiteral("%1 · %2")
        .arg(artifactId.left(8), relativePath);
    const auto update = [this, &value](
                            const QString& objectName, const QString& prefix) {
        if (auto* label = findChild<QLabel*>(objectName)) {
            label->setText(prefix.arg(value));
        }
    };
    if (normalizedKind.contains(QStringLiteral("checkpoint"))) {
        update(QStringLiteral("TrainingLatestCheckpoint"),
            tr("最新 checkpoint：%1"));
    }
    if (suffix == QStringLiteral("onnx")
        || normalizedKind.contains(QStringLiteral("onnx"))) {
        update(QStringLiteral("TrainingLatestOnnx"), tr("最新 ONNX：%1"));
    }
    if (normalizedKind.contains(QStringLiteral("report"))) {
        update(QStringLiteral("TrainingLatestReport"), tr("训练报告：%1"));
    }
    if (normalizedKind.contains(QStringLiteral("preview"))
        || normalizedKind.contains(QStringLiteral("overlay"))) {
        update(QStringLiteral("TrainingLatestPreview"), tr("最新预览：%1"));
    }
}

void TrainingWorkspacePage::appendLog(const QString& text)
{
    if (auto* log = findChild<QTextEdit*>(QStringLiteral("LogView"))) {
        QString line = text;
        constexpr int maxLogLineChars = 8000;
        if (line.size() > maxLogLineChars) {
            line = line.left(maxLogLineChars)
                + QStringLiteral(" ... [log_truncated]");
        }
        log->append(QStringLiteral("[%1] %2").arg(
            QTime::currentTime().toString(QStringLiteral("HH:mm:ss")), line));
    }
}

void TrainingWorkspacePage::resetRuntimeProjection()
{
    if (auto* progress =
            findChild<QProgressBar*>(QStringLiteral("TrainingProgress"))) {
        progress->setValue(0);
    }
    if (auto* log = findChild<QTextEdit*>(QStringLiteral("LogView"))) {
        log->clear();
    }
    if (auto* metrics = findChild<MetricsWidget*>()) {
        metrics->clear();
    }
    const QStringList values = {
        QStringLiteral("TrainingEpochValue"),
        QStringLiteral("TrainingBatchValue"),
        QStringLiteral("TrainingEtaValue"),
        QStringLiteral("TrainingDeviceValue"),
        QStringLiteral("TrainingLossValue"),
        QStringLiteral("TrainingMapValue")};
    for (const QString& name : values) {
        if (auto* label = findChild<QLabel*>(name)) {
            label->setText(QStringLiteral("--"));
        }
    }
}
