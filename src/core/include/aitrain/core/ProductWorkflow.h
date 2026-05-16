#pragma once

#include "aitrain/core/Cancellation.h"

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
WorkflowResult createDatasetSnapshotReport(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel);

WorkflowResult curateDatasetQualityReport(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options = {});
WorkflowResult curateDatasetQualityReport(
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel);

WorkflowResult evaluateModelReport(
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options = {});
WorkflowResult evaluateModelReport(
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel);

WorkflowResult benchmarkModelReport(
    const QString& modelPath,
    const QString& outputPath,
    const QJsonObject& options = {});
WorkflowResult benchmarkModelReport(
    const QString& modelPath,
    const QString& outputPath,
    const QJsonObject& options,
    const CancellationCallback& shouldCancel);

WorkflowResult generateTrainingDeliveryReport(
    const QString& outputPath,
    const QJsonObject& context = {});

WorkflowResult runCustomerOcrAcceptanceReport(
    const QString& outputPath,
    const QJsonObject& options = {});

WorkflowResult collectDiagnosticsReport(
    const QString& outputPath,
    const QJsonObject& context = {});

WorkflowResult validateDeploymentArtifactReport(
    const QString& modelPath,
    const QString& outputPath,
    const QString& format = QString(),
    const QJsonObject& options = {});

WorkflowResult runLocalPipelinePlan(
    const QString& outputPath,
    const QString& templateId,
    const QJsonObject& options = {});

} // namespace aitrain
