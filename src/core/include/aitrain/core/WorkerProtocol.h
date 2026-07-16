#pragma once

#include "aitrain/v2/ProtocolV2.h"

#include <QJsonObject>
#include <QString>

namespace aitrain {
namespace worker_protocol {

namespace command {
QString runEnvironmentCheckWorkflowV2();
QString runDatasetSplitWorkflowV2();
QString runDatasetConversionWorkflowV2();
QString runDataQualityWorkflowV2();
QString runDiagnosticsWorkflowV2();
QString createAnnotationSessionV2();
QString syncAnnotationSessionV2();
QString runDatasetSnapshotImportWorkflowV2();
QString importOcrOfficialReportsV2();
QString runOcrAcceptanceWorkflowV2();
QString runRuntimeDeliveryWorkflowV2();
QString importModelV2();
QString runTrainingWorkflowV2();
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
QString environmentCheckWorkflowV2();
QString datasetSplitWorkflowV2();
QString datasetConversionWorkflowV2();
QString dataQualityWorkflowV2();
QString diagnosticsWorkflowV2();
QString annotationSessionV2();
QString annotationSyncV2();
QString datasetSnapshotImportWorkflowV2();
QString runtimeDeliveryWorkflowV2();
QString modelImportV2();
QString ocrOfficialReportsImportedV2();
QString ocrAcceptanceWorkflowV2();
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

QJsonObject datasetSplitWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceDatasetId,
    const QString& sourceDatasetVersionId,
    const QString& sourceSnapshotId,
    const QString& sourceSnapshotArtifactId,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject datasetConversionWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject dataQualityWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& datasetId,
    const QString& datasetVersionId,
    const QString& snapshotId,
    const QString& snapshotArtifactId,
    const QJsonObject& options);
QJsonObject annotationSessionCreateV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& repairManifestArtifactId,
    const QString& workingDirectory,
    const QJsonObject& toolSummary,
    const QJsonObject& options);
QJsonObject annotationSessionSyncV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sessionArtifactId,
    const QString& workingDirectory,
    const QJsonObject& options);
QJsonObject datasetSnapshotImportWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options);
QJsonObject ocrOfficialReportImportV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& det,
    const QJsonObject& rec,
    const QJsonObject& system,
    const QString& acceptanceCohortId,
    const QString& customerDomainId,
    const QString& evidenceClass);
QJsonObject ocrAcceptanceWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& detReportArtifactId,
    const QString& recReportArtifactId,
    const QString& systemReportArtifactId,
    const QJsonObject& thresholds);
QJsonObject diagnosticsWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& options);
QJsonObject runtimeDeliveryWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& modelPackageId,
    const QString& runtimeRoute,
    const QString& sampleImagePath,
    const QJsonObject& options);
QJsonObject modelImportV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceFilePath,
    const QJsonObject& manifestDraft);

namespace control_v2 {
aitrain::v2::ProtocolEnvelope startTaskEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence,
    const QString& businessCommand,
    const QJsonObject& businessPayload);
aitrain::v2::ProtocolEnvelope cancelTaskEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence);
aitrain::v2::ProtocolEnvelope eventEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence,
    const QString& businessEvent,
    const QJsonObject& businessPayload);
bool unpackStartTask(const aitrain::v2::ProtocolEnvelope& envelope,
    QString* businessCommand,
    QJsonObject* businessPayload,
    QString* error = nullptr);
bool unpackBusinessEvent(const aitrain::v2::ProtocolEnvelope& envelope,
    QString* businessEvent,
    QJsonObject* businessPayload,
    QString* error = nullptr);
} // namespace control_v2

} // namespace worker_protocol
} // namespace aitrain
