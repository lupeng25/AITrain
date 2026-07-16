#pragma once

#include "aitrain/protocol/Protocol.h"

#include <QJsonObject>
#include <QString>

namespace aitrain {
namespace worker_protocol {

namespace command {
QString runEnvironmentCheckWorkflow();
QString runDatasetSplitWorkflow();
QString runDatasetConversionWorkflow();
QString runDataQualityWorkflow();
QString runDiagnosticsWorkflow();
QString createAnnotationSession();
QString syncAnnotationSession();
QString runDatasetSnapshotImportWorkflow();
QString importOcrOfficialReports();
QString runOcrAcceptanceWorkflow();
QString runRuntimeDeliveryWorkflow();
QString importModel();
QString runTrainingWorkflow();
} // namespace command

namespace event {
QString ready();
QString log();
QString progress();
QString metric();
QString artifact();
QString completed();
QString canceled();
QString failed();
QString environmentCheckWorkflow();
QString datasetSplitWorkflow();
QString datasetConversionWorkflow();
QString dataQualityWorkflow();
QString diagnosticsWorkflow();
QString annotationSession();
QString annotationSync();
QString datasetSnapshotImportWorkflow();
QString runtimeDeliveryWorkflow();
QString modelImport();
QString ocrOfficialReportsImported();
QString ocrAcceptanceWorkflow();
} // namespace event

namespace field {
QString taskId();
QString command();
QString status();
QString errorCode();
QString message();
QString datasetPath();
QString sourcePath();
QString format();
QString sourceFormat();
QString targetFormat();
QString taskType();
QString sampleImagePath();
QString options();
} // namespace field

bool isTerminalEvent(const QString& type);
bool isTaskStateEvent(const QString& type);

QJsonObject datasetSplitWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceDatasetId,
    const QString& sourceDatasetVersionId,
    const QString& sourceSnapshotId,
    const QString& sourceSnapshotArtifactId,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject datasetConversionWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject dataQualityWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& datasetId,
    const QString& datasetVersionId,
    const QString& snapshotId,
    const QString& snapshotArtifactId,
    const QJsonObject& options);
QJsonObject annotationSessionCreateRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& repairManifestArtifactId,
    const QString& workingDirectory,
    const QJsonObject& toolSummary,
    const QJsonObject& options);
QJsonObject annotationSessionSyncRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sessionArtifactId,
    const QString& workingDirectory,
    const QJsonObject& options);
QJsonObject datasetSnapshotImportWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject ocrOfficialReportImportRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& det,
    const QJsonObject& rec,
    const QJsonObject& system,
    const QString& acceptanceCohortId,
    const QString& customerDomainId,
    const QString& evidenceClass);
QJsonObject ocrAcceptanceWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& detReportArtifactId,
    const QString& recReportArtifactId,
    const QString& systemReportArtifactId,
    const QJsonObject& thresholds);
QJsonObject diagnosticsWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& options);
QJsonObject runtimeDeliveryWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& modelPackageId,
    const QString& runtimeRoute,
    const QString& sampleImagePath,
    const QJsonObject& options);
QJsonObject modelImportRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceFilePath,
    const QJsonObject& manifestDraft);

namespace control {
aitrain::ProtocolEnvelope startTaskEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const QString& businessCommand,
    const QJsonObject& businessPayload);
aitrain::ProtocolEnvelope cancelTaskEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence);
aitrain::ProtocolEnvelope eventEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const QString& businessEvent,
    const QJsonObject& businessPayload);
bool unpackStartTask(const aitrain::ProtocolEnvelope& envelope,
    QString* businessCommand,
    QJsonObject* businessPayload,
    QString* error = nullptr);
bool unpackBusinessEvent(const aitrain::ProtocolEnvelope& envelope,
    QString* businessEvent,
    QJsonObject* businessPayload,
    QString* error = nullptr);
} // namespace control

} // namespace worker_protocol
} // namespace aitrain
