#pragma once

#include "aitrain/core/TaskModels.h"

#include <QByteArray>
#include <QFileInfoList>
#include <QJsonObject>
#include <QString>
#include <QStringList>

class InfoPanel;
class QComboBox;
class QDir;
class QFrame;
class QLabel;
class QPushButton;

namespace aitrain_app {

QLabel* mutedLabel(const QString& text);
QLabel* emptyStateLabel(const QString& text);
QLabel* inlineStatusLabel(const QString& text);
void allowLabelToShrink(QLabel* label);
QString compactPathForStatus(const QString& path, int maxChars = 72);
QString compactTextForStatus(const QString& text, int maxChars = 96);
QPushButton* primaryButton(const QString& text);
QPushButton* dangerButton(const QString& text);
QString uiText(const char* source);
QString defaultProjectPathSettingsKey();
QString taskTypeLabel(const QString& taskType);
void addComboItem(QComboBox* combo, const QString& displayText, const QString& value);
QString backendLabel(const QString& backend);
QJsonObject readJsonObjectFile(const QString& path);
QJsonObject compactEvaluationSummary(const QString& reportPath);
QJsonObject compactBenchmarkSummary(const QString& reportPath);
QString metricValueText(const QJsonObject& metrics, const QStringList& keys);
QString modelSummaryText(const QJsonObject& summary);
QString exportComboLabel(const QString& format);
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
void loadInferenceOverlay(QLabel* label, const QString& path);
QString taskStateLabel(aitrain::TaskState state);
QString taskKindLabel(aitrain::TaskKind kind);
QString environmentStatusLabel(const QString& status);
QString issueSeverityLabel(const QString& severity);
QString inferenceTaskTypeLabel(const QString& taskType);
QString datasetFormatLabel(const QString& format);
QString defaultBackendForTask(const QString& taskType);
QString defaultModelForBackend(const QString& backend);
QString trainingBackendDescription(const QString& backend);
QString exportFormatLabel(const QString& format);
QString defaultExportFileName(const QString& format);
QString exportFileFilter(const QString& format);
QString exportFormatNote(const QString& format);
QString compactListSummary(const QStringList& values, int maxItems = 3);
int uniqueStringCount(const QStringList& values);
bool setComboCurrentData(QComboBox* combo, const QString& data);
QString confidencePercent(double confidence);
QString resolvedXAnyLabelingProgram();
QString xAnyLabelingStatusText();
QString detectDatasetFormatFromPath(const QString& path);
QString formatJsonTextForPreview(const QByteArray& data);
void addTaskTypeItems(QComboBox* combo, const QStringList& taskTypes);
QString comboCurrentDataOrText(const QComboBox* combo);
QString inferenceSummaryFromPredictions(const QString& predictionsPath, const QJsonObject& fallback = {});

} // namespace aitrain_app
