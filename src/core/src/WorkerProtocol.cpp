#include "aitrain/core/WorkerProtocol.h"

namespace aitrain {
namespace worker_protocol {

namespace {
QString controlEventKind(const QString& businessEvent)
{
    if (businessEvent == event::ready()) return QStringLiteral("event.ready");
    if (businessEvent == event::log()) return QStringLiteral("event.log");
    if (businessEvent == event::metric()) return QStringLiteral("event.metric");
    if (businessEvent == event::artifact()) return QStringLiteral("event.artifact");
    if (businessEvent == event::completed()) return QStringLiteral("event.succeeded");
    if (businessEvent == event::failed()) return QStringLiteral("event.failed");
    if (businessEvent == event::canceled()) return QStringLiteral("event.canceled");
    if (businessEvent == event::progress()) {
        return QStringLiteral("event.progress");
    }
    return QStringLiteral("event.result");
}

aitrain::v2::ProtocolEnvelope makeControlEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence,
    const QString& kind,
    const QJsonObject& payload)
{
    aitrain::v2::ProtocolEnvelope envelope;
    envelope.messageId = aitrain::v2::MessageId::create();
    envelope.requestId = requestId;
    envelope.taskId = taskId;
    envelope.sequence = sequence;
    envelope.kind = kind;
    envelope.timestamp = QDateTime::currentDateTimeUtc();
    envelope.payload = payload;
    return envelope;
}
} // namespace

namespace command {
QString runEnvironmentCheckWorkflowV2() { return QStringLiteral("runEnvironmentCheckWorkflowV2"); }
QString runDatasetSplitWorkflowV2() { return QStringLiteral("runDatasetSplitWorkflowV2"); }
QString runDatasetConversionWorkflowV2() { return QStringLiteral("runDatasetConversionWorkflowV2"); }
QString runDataQualityWorkflowV2() { return QStringLiteral("runDataQualityWorkflowV2"); }
QString runDiagnosticsWorkflowV2() { return QStringLiteral("runDiagnosticsWorkflowV2"); }
QString createAnnotationSessionV2() { return QStringLiteral("createAnnotationSessionV2"); }
QString syncAnnotationSessionV2() { return QStringLiteral("syncAnnotationSessionV2"); }
QString runDatasetSnapshotImportWorkflowV2() { return QStringLiteral("runDatasetSnapshotImportWorkflowV2"); }
QString importOcrOfficialReportsV2() { return QStringLiteral("importOcrOfficialReportsV2"); }
QString runOcrAcceptanceWorkflowV2() { return QStringLiteral("runOcrAcceptanceWorkflowV2"); }
QString runRuntimeDeliveryWorkflowV2() { return QStringLiteral("runRuntimeDeliveryWorkflowV2"); }
QString importModelV2() { return QStringLiteral("importModelV2"); }
QString runTrainingWorkflowV2() { return QStringLiteral("runTrainingWorkflowV2"); }
} // namespace command

namespace event {
QString ready() { return QStringLiteral("ready"); }
QString log() { return QStringLiteral("log"); }
QString progress() { return QStringLiteral("progress"); }
QString metric() { return QStringLiteral("metric"); }
QString artifact() { return QStringLiteral("artifact"); }
QString completed() { return QStringLiteral("completed"); }
QString canceled() { return QStringLiteral("canceled"); }
QString failed() { return QStringLiteral("failed"); }
QString environmentCheckWorkflowV2() { return QStringLiteral("environmentCheckWorkflowV2"); }
QString datasetSplitWorkflowV2() { return QStringLiteral("datasetSplitWorkflowV2"); }
QString datasetConversionWorkflowV2() { return QStringLiteral("datasetConversionWorkflowV2"); }
QString dataQualityWorkflowV2() { return QStringLiteral("dataQualityWorkflowV2"); }
QString diagnosticsWorkflowV2() { return QStringLiteral("diagnosticsWorkflowV2"); }
QString annotationSessionV2() { return QStringLiteral("annotationSessionV2"); }
QString annotationSyncV2() { return QStringLiteral("annotationSyncV2"); }
QString datasetSnapshotImportWorkflowV2() { return QStringLiteral("datasetSnapshotImportWorkflowV2"); }
QString runtimeDeliveryWorkflowV2() { return QStringLiteral("runtimeDeliveryWorkflowV2"); }
QString modelImportV2() { return QStringLiteral("modelImportV2"); }
QString ocrOfficialReportsImportedV2() { return QStringLiteral("ocrOfficialReportsImportedV2"); }
QString ocrAcceptanceWorkflowV2() { return QStringLiteral("ocrAcceptanceWorkflowV2"); }
} // namespace event

namespace field {
QString taskId() { return QStringLiteral("taskId"); }
QString command() { return QStringLiteral("command"); }
QString status() { return QStringLiteral("status"); }
QString errorCode() { return QStringLiteral("errorCode"); }
QString message() { return QStringLiteral("message"); }
QString datasetPath() { return QStringLiteral("datasetPath"); }
QString sourcePath() { return QStringLiteral("sourcePath"); }
QString format() { return QStringLiteral("format"); }
QString sourceFormat() { return QStringLiteral("sourceFormat"); }
QString targetFormat() { return QStringLiteral("targetFormat"); }
QString taskType() { return QStringLiteral("taskType"); }
QString sampleImagePath() { return QStringLiteral("sampleImagePath"); }
QString options() { return QStringLiteral("options"); }
} // namespace field

bool isTerminalEvent(const QString& type)
{
    return type == event::completed()
        || type == event::failed()
        || type == event::canceled();
}

bool isTaskStateEvent(const QString& type)
{
    return isTerminalEvent(type);
}

QJsonObject datasetSplitWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceDatasetId,
    const QString& sourceDatasetVersionId,
    const QString& sourceSnapshotId,
    const QString& sourceSnapshotArtifactId,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("sourceDatasetId"), sourceDatasetId},
        {QStringLiteral("sourceDatasetVersionId"), sourceDatasetVersionId},
        {QStringLiteral("sourceSnapshotId"), sourceSnapshotId},
        {QStringLiteral("sourceSnapshotArtifactId"), sourceSnapshotArtifactId},
        {QStringLiteral("targetDatasetId"), targetDatasetId},
        {QStringLiteral("targetDatasetName"), targetDatasetName},
        {field::options(), options}};
}

QJsonObject datasetConversionWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {field::sourcePath(), sourcePath},
        {field::sourceFormat(), sourceFormat},
        {field::targetFormat(), targetFormat},
        {QStringLiteral("targetDatasetId"), targetDatasetId},
        {QStringLiteral("targetDatasetName"), targetDatasetName},
        {field::options(), options}};
}

QJsonObject dataQualityWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& datasetId,
    const QString& datasetVersionId,
    const QString& snapshotId,
    const QString& snapshotArtifactId,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("datasetId"), datasetId},
        {QStringLiteral("datasetVersionId"), datasetVersionId},
        {QStringLiteral("snapshotId"), snapshotId},
        {QStringLiteral("snapshotArtifactId"), snapshotArtifactId},
        {field::options(), options}};
}

QJsonObject annotationSessionCreateV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& repairManifestArtifactId,
    const QString& workingDirectory,
    const QJsonObject& toolSummary,
    const QJsonObject& options)
{
    return QJsonObject{
        {field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("repairManifestArtifactId"), repairManifestArtifactId},
        {QStringLiteral("workingDirectory"), workingDirectory},
        {QStringLiteral("toolSummary"), toolSummary},
        {field::options(), options}};
}

QJsonObject annotationSessionSyncV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sessionArtifactId,
    const QString& workingDirectory,
    const QJsonObject& options)
{
    return QJsonObject{
        {field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("sessionArtifactId"), sessionArtifactId},
        {QStringLiteral("workingDirectory"), workingDirectory},
        {field::options(), options}};
}

QJsonObject datasetSnapshotImportWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourcePath,
    const QString& sourceFormat,
    const QString& targetDatasetId,
    const QString& targetDatasetName,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {field::sourcePath(), sourcePath},
        {field::sourceFormat(), sourceFormat},
        {QStringLiteral("targetDatasetId"), targetDatasetId},
        {QStringLiteral("targetDatasetName"), targetDatasetName},
        {field::options(), options}};
}

QJsonObject ocrOfficialReportImportV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& det,
    const QJsonObject& rec,
    const QJsonObject& system,
    const QString& acceptanceCohortId,
    const QString& customerDomainId,
    const QString& evidenceClass)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("det"), det},
        {QStringLiteral("rec"), rec},
        {QStringLiteral("system"), system},
        {QStringLiteral("acceptanceCohortId"), acceptanceCohortId},
        {QStringLiteral("customerDomainId"), customerDomainId},
        {QStringLiteral("evidenceClass"), evidenceClass}};
}

QJsonObject ocrAcceptanceWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& detReportArtifactId,
    const QString& recReportArtifactId,
    const QString& systemReportArtifactId,
    const QJsonObject& thresholds)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("detReportArtifactId"), detReportArtifactId},
        {QStringLiteral("recReportArtifactId"), recReportArtifactId},
        {QStringLiteral("systemReportArtifactId"), systemReportArtifactId},
        {QStringLiteral("thresholds"), thresholds}};
}

QJsonObject diagnosticsWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {field::options(), options}};
}

QJsonObject runtimeDeliveryWorkflowV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& modelPackageId,
    const QString& runtimeRoute,
    const QString& sampleImagePath,
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(QStringLiteral("projectRoot"), projectRoot);
    payload.insert(QStringLiteral("modelPackageId"), modelPackageId);
    payload.insert(QStringLiteral("runtimeRoute"), runtimeRoute);
    payload.insert(field::sampleImagePath(), sampleImagePath);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject modelImportV2Request(
    const QString& taskId,
    const QString& projectRoot,
    const QString& sourceFilePath,
    const QJsonObject& manifestDraft)
{
    return QJsonObject{
        {field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {QStringLiteral("sourceFilePath"), sourceFilePath},
        {QStringLiteral("manifestDraft"), manifestDraft}};
}

namespace control_v2 {
aitrain::v2::ProtocolEnvelope startTaskEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence,
    const QString& businessCommand,
    const QJsonObject& businessPayload)
{
    return makeControlEnvelope(requestId, taskId, sequence, QStringLiteral("command.start_task"),
        QJsonObject{{QStringLiteral("businessCommand"), businessCommand},
            {QStringLiteral("businessPayload"), businessPayload}});
}

aitrain::v2::ProtocolEnvelope cancelTaskEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence)
{
    return makeControlEnvelope(requestId, taskId, sequence, QStringLiteral("command.cancel_task"), QJsonObject{});
}

aitrain::v2::ProtocolEnvelope eventEnvelope(
    const aitrain::v2::RequestId& requestId,
    const aitrain::v2::TaskId& taskId,
    quint64 sequence,
    const QString& businessEvent,
    const QJsonObject& businessPayload)
{
    return makeControlEnvelope(requestId, taskId, sequence, controlEventKind(businessEvent),
        QJsonObject{{QStringLiteral("businessEvent"), businessEvent},
            {QStringLiteral("businessPayload"), businessPayload}});
}

bool unpackStartTask(const aitrain::v2::ProtocolEnvelope& envelope,
    QString* businessCommand,
    QJsonObject* businessPayload,
    QString* error)
{
    const QString command = envelope.payload.value(QStringLiteral("businessCommand")).toString().trimmed();
    const QJsonValue payloadValue = envelope.payload.value(QStringLiteral("businessPayload"));
    if (envelope.kind != QStringLiteral("command.start_task") || command.isEmpty() || !payloadValue.isObject()) {
        if (error) *error = QStringLiteral("command.start_task payload 必须包含 businessCommand 和对象 businessPayload。");
        return false;
    }
    if (businessCommand) *businessCommand = command;
    if (businessPayload) *businessPayload = payloadValue.toObject();
    return true;
}

bool unpackBusinessEvent(const aitrain::v2::ProtocolEnvelope& envelope,
    QString* businessEvent,
    QJsonObject* businessPayload,
    QString* error)
{
    const QString eventName = envelope.payload.value(QStringLiteral("businessEvent")).toString().trimmed();
    const QJsonValue payloadValue = envelope.payload.value(QStringLiteral("businessPayload"));
    if (!envelope.kind.startsWith(QStringLiteral("event.")) || eventName.isEmpty() || !payloadValue.isObject()
        || controlEventKind(eventName) != envelope.kind) {
        if (error) *error = QStringLiteral("Protocol V2 事件 payload 必须包含 businessEvent 和对象 businessPayload。");
        return false;
    }
    if (businessEvent) *businessEvent = eventName;
    if (businessPayload) *businessPayload = payloadValue.toObject();
    return true;
}
} // namespace control_v2

} // namespace worker_protocol
} // namespace aitrain
