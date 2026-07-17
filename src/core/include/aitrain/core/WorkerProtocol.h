#pragma once

#include "aitrain/protocol/Protocol.h"
#include "aitrain/core/TaskTypes.h"

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
QString importExternalAcceptanceEvidence();
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
QString externalAcceptanceEvidenceImported();
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
QString sampleDatasetId();
QString sampleDatasetVersionId();
QString sampleSnapshotId();
QString sampleSnapshotArtifactId();
QString sampleRelativePath();
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
    const QString& sampleDatasetId,
    const QString& sampleDatasetVersionId,
    const QString& sampleSnapshotId,
    const QString& sampleSnapshotArtifactId,
    const QString& sampleRelativePath,
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
    const TaskCommand& command,
    const QString& controlToken = {});
aitrain::ProtocolEnvelope cancelTaskEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const QString& controlToken = {});
aitrain::ProtocolEnvelope eventEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const TaskEvent& event,
    const QString& controlToken = {});
bool unpackStartTask(const aitrain::ProtocolEnvelope& envelope,
    TaskCommand* command,
    QString* error = nullptr);
bool unpackTaskEvent(const aitrain::ProtocolEnvelope& envelope,
    TaskEvent* event,
    QString* error = nullptr);
} // namespace control

QString taskCommandType(const TaskCommand& command);
bool taskCommandFromPayload(const QString& type,
    const QJsonObject& payload,
    TaskCommand* command,
    QString* error = nullptr);
QJsonObject taskCommandPayload(const TaskCommand& command);

TaskEventKind taskEventKindFromType(const QString& type, bool* ok = nullptr);
QString taskEventType(const TaskEvent& event);
TaskEvent taskEventFromType(const QString& type, const QJsonObject& details);

} // namespace worker_protocol
} // namespace aitrain
