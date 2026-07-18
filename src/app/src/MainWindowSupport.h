#pragma once

#include <QByteArray>
#include <QFileInfoList>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include <functional>

class InfoPanel;
class QComboBox;
class QDir;
class QFrame;
class QLabel;
class QPushButton;
class QObject;

namespace aitrain_app {

struct ReviewSamplePathView final {
    QString imageRelativePath;
    QString labelRelativePath;
};

QLabel* mutedLabel(const QString& text);
QLabel* emptyStateLabel(const QString& text);
QLabel* inlineStatusLabel(const QString& text);
void allowLabelToShrink(QLabel* label);
QString compactPathForStatus(const QString& path, int maxChars = 72);
QString compactTextForStatus(const QString& text, int maxChars = 96);
QPushButton* primaryButton(const QString& text);
QPushButton* dangerButton(const QString& text);
QString uiText(const char* source);
QString taskTypeLabel(const QString& taskType);
void addComboItem(QComboBox* combo, const QString& displayText, const QString& value);
QString backendLabel(const QString& backend);
QString metricValueText(const QJsonObject& metrics, const QStringList& keys);
QString modelSummaryText(const QJsonObject& summary);
InfoPanel* createCompactSummaryCard(const QString& label, const QString& value, const QString& caption);
QLabel* inferenceBadge(const QString& text);
QFrame* createInferenceStep(const QString& index, const QString& title, const QString& caption);
QFrame* createInferenceCapability(const QString& title, const QString& caption);
QFrame* createWorkbenchHeader(
    const QString& kickerText,
    const QString& titleText,
    const QString& subtitleText,
    QPushButton* actionButton,
    const QStringList& badges);
void setInferenceOverlayText(QLabel* label, const QString& text);
QString environmentStatusLabel(const QString& status);
QString issueSeverityLabel(const QString& severity);
QString inferenceTaskTypeLabel(const QString& taskType);
QString datasetFormatLabel(const QString& format);
QString defaultBackendForTask(const QString& taskType);
QString defaultModelForBackend(const QString& backend);
QString trainingBackendDescription(const QString& backend);
QStringList modelPresetItemsForBackend(const QString& backend);
QString compactListSummary(const QStringList& values, int maxItems = 3);
int uniqueStringCount(const QStringList& values);
bool setComboCurrentData(QComboBox* combo, const QString& data);
QString confidencePercent(double confidence);
QString resolvedXAnyLabelingProgram();
QString xAnyLabelingStatusText();
QString detectDatasetFormatFromPath(const QString& path);
// 在后台线程执行有界格式探测，并将结果排队回调到 context 所在线程。
// 回调只返回诊断结果，真正的 Dataset Driver 校验仍必须由 Worker/Core 完成。
using DatasetFormatProbeCallback = std::function<void(const QString& detectedFormat)>;
void detectDatasetFormatAsync(QObject* context, const QString& path,
    DatasetFormatProbeCallback callback);
QString formatJsonTextForPreview(const QByteArray& data);
void addTaskTypeItems(QComboBox* combo, const QStringList& taskTypes);
QString comboCurrentDataOrText(const QComboBox* combo);
ReviewSamplePathView reviewSamplePathView(const QJsonObject& sample);

} // namespace aitrain_app
