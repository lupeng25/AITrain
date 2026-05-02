#pragma once

#include <QJsonObject>
#include <QString>

namespace aitrain {

struct WorkflowResult {
    bool ok = false;
    QString error;
    QString reportPath;
    QJsonObject payload;
};

WorkflowResult createDatasetSnapshotReport(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options = {});

WorkflowResult curateDatasetQualityReport(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options = {});

WorkflowResult evaluateModelReport(
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options = {});

WorkflowResult benchmarkModelReport(
    const QString& modelPath,
    const QString& outputPath,
    const QJsonObject& options = {});

WorkflowResult generateTrainingDeliveryReport(
    const QString& outputPath,
    const QJsonObject& context = {});

WorkflowResult runLocalPipelinePlan(
    const QString& outputPath,
    const QString& templateId,
    const QJsonObject& options = {});

} // namespace aitrain
