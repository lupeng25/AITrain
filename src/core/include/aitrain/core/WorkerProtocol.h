#pragma once

#include "aitrain/core/TaskModels.h"

#include <QJsonObject>
#include <QString>

namespace aitrain {
namespace worker_protocol {

namespace command {
QString startTrain();
QString pause();
QString resume();
QString heartbeat();
QString environmentCheck();
QString validateDataset();
QString splitDataset();
QString convertDataset();
QString curateDataset();
QString createDatasetSnapshot();
QString evaluateModel();
QString benchmarkModel();
QString runLocalPipeline();
QString generateDeliveryReport();
QString runCustomerOcrAcceptance();
QString collectDiagnostics();
QString validateDeploymentArtifact();
QString exportModel();
QString infer();
QString cancel();
} // namespace command

namespace event {
QString ready();
QString log();
QString heartbeat();
QString progress();
QString metric();
QString artifact();
QString completed();
QString paused();
QString resumed();
QString canceled();
QString failed();
QString environmentCheck();
QString datasetValidation();
QString datasetSplit();
QString datasetConversion();
QString datasetQuality();
QString datasetSnapshot();
QString evaluationReport();
QString benchmarkReport();
QString pipelinePlan();
QString deliveryReport();
QString modelExport();
QString deploymentValidation();
QString inferenceResult();
QString customerOcrAcceptance();
QString diagnosticBundle();
} // namespace event

namespace field {
QString taskId();
QString command();
QString status();
QString errorCode();
QString message();
QString datasetPath();
QString sourcePath();
QString outputPath();
QString modelPath();
QString checkpointPath();
QString imagePath();
QString format();
QString sourceFormat();
QString targetFormat();
QString taskType();
QString templateId();
QString sampleImagePath();
QString options();
QString context();
QString reportPath();
QString bundlePath();
} // namespace field

bool isControlCommand(const QString& type);
bool isTerminalEvent(const QString& type);
bool isTaskStateEvent(const QString& type);
TaskState taskStateForEvent(const QString& type, TaskState fallback = TaskState::Failed);

QJsonObject datasetValidationRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& format,
    const QJsonObject& options,
    const QString& outputPath = QString());
QJsonObject datasetSplitRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options);
QJsonObject datasetConversionRequest(
    const QString& taskId,
    const QString& sourcePath,
    const QString& outputPath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QJsonObject& options);
QJsonObject datasetCurationRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options);
QJsonObject datasetSnapshotRequest(
    const QString& taskId,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& format,
    const QJsonObject& options);
QJsonObject modelEvaluationRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& datasetPath,
    const QString& outputPath,
    const QString& taskType,
    const QJsonObject& options);
QJsonObject modelBenchmarkRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& outputPath,
    const QJsonObject& options);
QJsonObject localPipelineRequest(
    const QString& taskId,
    const QString& outputPath,
    const QString& templateId,
    const QJsonObject& options);
QJsonObject deliveryReportRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& context);
QJsonObject customerOcrAcceptanceRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& options);
QJsonObject diagnosticsBundleRequest(
    const QString& taskId,
    const QString& outputPath,
    const QJsonObject& context);
QJsonObject deploymentValidationRequest(
    const QString& taskId,
    const QString& modelPath,
    const QString& outputPath,
    const QString& format,
    const QString& sampleImagePath,
    const QJsonObject& options);
QJsonObject modelExportRequest(
    const QString& taskId,
    const QString& checkpointPath,
    const QString& outputPath,
    const QString& format);
QJsonObject inferenceRequest(
    const QString& taskId,
    const QString& checkpointPath,
    const QString& imagePath,
    const QString& outputPath);

} // namespace worker_protocol
} // namespace aitrain
