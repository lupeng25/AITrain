#include "aitrain/core/WorkerProtocol.h"

#include <QtMath>

#include <limits>
#include <type_traits>

namespace aitrain {
namespace worker_protocol {

namespace {
QString controlEventKind(const QString& eventType)
{
    if (eventType == event::ready()) return QStringLiteral("event.ready");
    if (eventType == event::log()) return QStringLiteral("event.log");
    if (eventType == event::metric()) return QStringLiteral("event.metric");
    if (eventType == event::artifact()) return QStringLiteral("event.artifact");
    if (eventType == event::completed()) return QStringLiteral("event.succeeded");
    if (eventType == event::failed()) return QStringLiteral("event.failed");
    if (eventType == event::canceled()) return QStringLiteral("event.canceled");
    if (eventType == event::progress()) {
        return QStringLiteral("event.progress");
    }
    return QStringLiteral("event.result");
}

QString taskCommandSchema()
{
    return QStringLiteral("aitrain.task-command.typed");
}

QString taskEventSchema()
{
    return QStringLiteral("aitrain.task-event.typed");
}

bool requiredString(const QJsonObject& object, const QString& key, QString* value, QString* error)
{
    const QJsonValue jsonValue = object.value(key);
    if (!jsonValue.isString() || jsonValue.toString().trimmed().isEmpty()) {
        if (error) {
            *error = QStringLiteral("字段 %1 必须是非空字符串。").arg(key);
        }
        return false;
    }
    if (value) {
        *value = jsonValue.toString().trimmed();
    }
    return true;
}

bool optionalObject(const QJsonObject& object, const QString& key, QJsonObject* value, QString* error)
{
    if (!object.contains(key)) {
        if (value) *value = QJsonObject();
        return true;
    }
    if (!object.value(key).isObject()) {
        if (error) *error = QStringLiteral("字段 %1 必须是对象。").arg(key);
        return false;
    }
    if (value) *value = object.value(key).toObject();
    return true;
}

bool requiredObject(const QJsonObject& object, const QString& key, QJsonObject* value, QString* error)
{
    if (!object.contains(key) || !object.value(key).isObject()) {
        if (error) *error = QStringLiteral("字段 %1 必须是对象。").arg(key);
        return false;
    }
    if (value) *value = object.value(key).toObject();
    return true;
}

bool parseContext(const QJsonObject& object, TaskCommandContext* context, QString* error)
{
    QString taskIdText;
    QString projectRoot;
    if (!requiredString(object, QStringLiteral("taskId"), &taskIdText, error)
        || !requiredString(object, QStringLiteral("projectRoot"), &projectRoot, error)) {
        return false;
    }
    TaskId taskId;
    if (!TaskId::parse(taskIdText, &taskId, error)) {
        return false;
    }
    if (context) {
        context->taskId = taskId;
        context->projectRoot = projectRoot;
    }
    return true;
}

void putContext(QJsonObject* object, const TaskCommandContext& context)
{
    object->insert(QStringLiteral("taskId"), context.taskId.toString());
    object->insert(QStringLiteral("projectRoot"), context.projectRoot);
}

bool hasOnlyKeys(const QJsonObject& object,
    const QStringList& required,
    const QStringList& optional,
    QString* error)
{
    for (const QString& key : required) {
        if (!object.contains(key)) {
            if (error) *error = QStringLiteral("缺少字段 %1。").arg(key);
            return false;
        }
    }
    QSet<QString> optionalSet;
    for (const QString& key : optional) optionalSet.insert(key);
    QSet<QString> requiredSet;
    for (const QString& key : required) requiredSet.insert(key);
    for (const QString& key : object.keys()) {
        if (!requiredSet.contains(key) && !optionalSet.contains(key)) {
            if (error) *error = QStringLiteral("未知字段 %1。").arg(key);
            return false;
        }
    }
    return true;
}

bool requiredEventString(const QJsonObject& object,
    const QString& key,
    QString* error)
{
    const QJsonValue value = object.value(key);
    if (!value.isString() || value.toString().trimmed().isEmpty()) {
        if (error) {
            *error = QStringLiteral("事件字段 %1 必须是非空字符串。").arg(key);
        }
        return false;
    }
    return true;
}

bool optionalEventString(const QJsonObject& object,
    const QString& key,
    QString* error)
{
    if (!object.contains(key)) {
        return true;
    }
    if (!object.value(key).isString()) {
        if (error) {
            *error = QStringLiteral("事件字段 %1 必须是字符串。").arg(key);
        }
        return false;
    }
    return true;
}

bool optionalEventNumber(const QJsonObject& object,
    const QString& key,
    double minimum,
    double maximum,
    QString* error)
{
    if (!object.contains(key)) {
        return true;
    }
    const QJsonValue value = object.value(key);
    const double number = value.toDouble(qQNaN());
    if (!value.isDouble() || !qIsFinite(number) || number < minimum || number > maximum) {
        if (error) {
            *error = QStringLiteral("事件字段 %1 必须是范围 [%2, %3] 内的有限数字。")
                .arg(key).arg(minimum).arg(maximum);
        }
        return false;
    }
    return true;
}

bool requiredEventNumber(const QJsonObject& object,
    const QString& key,
    double minimum,
    double maximum,
    QString* error)
{
    if (!object.contains(key)) {
        if (error) {
            *error = QStringLiteral("事件缺少字段 %1。").arg(key);
        }
        return false;
    }
    return optionalEventNumber(object, key, minimum, maximum, error);
}

bool validateTaskEventDetails(TaskEventKind kind,
    const QString& resultType,
    const QJsonObject& details,
    QString* error)
{
    // TaskId 是所有事件的身份字段；具体值和 envelope 的一致性在调用方校验。
    QString taskIdText;
    if (!requiredString(details, QStringLiteral("taskId"), &taskIdText, error)) {
        return false;
    }
    TaskId taskId;
    if (!TaskId::parse(taskIdText, &taskId, error)) {
        return false;
    }

    switch (kind) {
    case TaskEventKind::Ready:
        return requiredEventString(details, QStringLiteral("message"), error);
    case TaskEventKind::Log:
        return requiredEventString(details, QStringLiteral("message"), error)
            && optionalEventString(details, QStringLiteral("level"), error);
    case TaskEventKind::Progress: {
        const bool hasPercent = details.contains(QStringLiteral("percent"));
        const bool hasValue = details.contains(QStringLiteral("value"));
        if (!hasPercent && !hasValue) {
            if (error) *error = QStringLiteral("progress 事件必须包含 percent 或 value。");
            return false;
        }
        if (!optionalEventNumber(details, QStringLiteral("percent"), 0.0, 100.0, error)
            || !optionalEventNumber(details, QStringLiteral("value"), 0.0, 1.0, error)) {
            return false;
        }
        return optionalEventString(details, QStringLiteral("message"), error);
    }
    case TaskEventKind::Metric:
        return requiredEventString(details, QStringLiteral("name"), error)
            && requiredEventNumber(details, QStringLiteral("value"), -std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(), error);
    case TaskEventKind::Artifact:
        return requiredEventString(details, QStringLiteral("artifactId"), error)
            && requiredEventString(details, QStringLiteral("kind"), error)
            && requiredEventString(details, QStringLiteral("relativePath"), error)
            && optionalEventString(details, QStringLiteral("message"), error);
    case TaskEventKind::Succeeded:
    case TaskEventKind::Failed:
    case TaskEventKind::Canceled:
        if (!requiredEventString(details, QStringLiteral("message"), error)) {
            return false;
        }
        if (!optionalEventString(details, QStringLiteral("status"), error)
            || !optionalEventString(details, QStringLiteral("errorCode"), error)) {
            return false;
        }
        if (kind == TaskEventKind::Failed
            && !details.contains(QStringLiteral("errorCode"))
            && !details.contains(QStringLiteral("failureCode"))) {
            if (error) *error = QStringLiteral("failed 事件必须包含 errorCode 或 failureCode。");
            return false;
        }
        return optionalEventString(details, QStringLiteral("failureCode"), error);
    case TaskEventKind::Result:
        if (resultType.trimmed().isEmpty()) {
            if (error) *error = QStringLiteral("result 事件必须包含 resultType。");
            return false;
        }
        return true;
    }
    if (error) *error = QStringLiteral("未知 TaskEvent kind。");
    return false;
}

QString commandTypeForKind(TaskEventKind kind)
{
    switch (kind) {
    case TaskEventKind::Ready: return event::ready();
    case TaskEventKind::Log: return event::log();
    case TaskEventKind::Progress: return event::progress();
    case TaskEventKind::Metric: return event::metric();
    case TaskEventKind::Artifact: return event::artifact();
    case TaskEventKind::Succeeded: return event::completed();
    case TaskEventKind::Failed: return event::failed();
    case TaskEventKind::Canceled: return event::canceled();
    case TaskEventKind::Result: return QStringLiteral("result");
    }
    return QString();
}

aitrain::ProtocolEnvelope makeControlEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const QString& kind,
    const QJsonObject& payload,
    const QString& controlToken)
{
    aitrain::ProtocolEnvelope envelope;
    envelope.messageId = aitrain::MessageId::create();
    envelope.requestId = requestId;
    envelope.taskId = taskId;
    envelope.controlToken = controlToken;
    envelope.sequence = sequence;
    envelope.kind = kind;
    envelope.timestamp = QDateTime::currentDateTimeUtc();
    envelope.payload = payload;
    return envelope;
}
} // namespace

namespace command {
QString runEnvironmentCheckWorkflow() { return QStringLiteral("runEnvironmentCheckWorkflow"); }
QString runDatasetSplitWorkflow() { return QStringLiteral("runDatasetSplitWorkflow"); }
QString runDatasetConversionWorkflow() { return QStringLiteral("runDatasetConversionWorkflow"); }
QString runDataQualityWorkflow() { return QStringLiteral("runDataQualityWorkflow"); }
QString runDiagnosticsWorkflow() { return QStringLiteral("runDiagnosticsWorkflow"); }
QString importExternalAcceptanceEvidence() { return QStringLiteral("importExternalAcceptanceEvidence"); }
QString createAnnotationSession() { return QStringLiteral("createAnnotationSession"); }
QString syncAnnotationSession() { return QStringLiteral("syncAnnotationSession"); }
QString runDatasetSnapshotImportWorkflow() { return QStringLiteral("runDatasetSnapshotImportWorkflow"); }
QString importOcrOfficialReports() { return QStringLiteral("importOcrOfficialReports"); }
QString runOcrAcceptanceWorkflow() { return QStringLiteral("runOcrAcceptanceWorkflow"); }
QString runRuntimeDeliveryWorkflow() { return QStringLiteral("runRuntimeDeliveryWorkflow"); }
QString importModel() { return QStringLiteral("importModel"); }
QString runTrainingWorkflow() { return QStringLiteral("runTrainingWorkflow"); }
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
QString environmentCheckWorkflow() { return QStringLiteral("environmentCheckWorkflow"); }
QString datasetSplitWorkflow() { return QStringLiteral("datasetSplitWorkflow"); }
QString datasetConversionWorkflow() { return QStringLiteral("datasetConversionWorkflow"); }
QString dataQualityWorkflow() { return QStringLiteral("dataQualityWorkflow"); }
QString diagnosticsWorkflow() { return QStringLiteral("diagnosticsWorkflow"); }
QString annotationSession() { return QStringLiteral("annotationSession"); }
QString annotationSync() { return QStringLiteral("annotationSync"); }
QString datasetSnapshotImportWorkflow() { return QStringLiteral("datasetSnapshotImportWorkflow"); }
QString runtimeDeliveryWorkflow() { return QStringLiteral("runtimeDeliveryWorkflow"); }
QString modelImport() { return QStringLiteral("modelImport"); }
QString ocrOfficialReportsImported() { return QStringLiteral("ocrOfficialReportsImported"); }
QString ocrAcceptanceWorkflow() { return QStringLiteral("ocrAcceptanceWorkflow"); }
QString externalAcceptanceEvidenceImported() { return QStringLiteral("externalAcceptanceEvidenceImported"); }
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
QString sampleDatasetId() { return QStringLiteral("sampleDatasetId"); }
QString sampleDatasetVersionId() { return QStringLiteral("sampleDatasetVersionId"); }
QString sampleSnapshotId() { return QStringLiteral("sampleSnapshotId"); }
QString sampleSnapshotArtifactId() { return QStringLiteral("sampleSnapshotArtifactId"); }
QString sampleRelativePath() { return QStringLiteral("sampleRelativePath"); }
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

QJsonObject datasetSplitWorkflowRequest(
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

QJsonObject datasetConversionWorkflowRequest(
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

QJsonObject dataQualityWorkflowRequest(
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

QJsonObject annotationSessionCreateRequest(
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

QJsonObject annotationSessionSyncRequest(
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

QJsonObject datasetSnapshotImportWorkflowRequest(
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

QJsonObject ocrOfficialReportImportRequest(
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

QJsonObject ocrAcceptanceWorkflowRequest(
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

QJsonObject diagnosticsWorkflowRequest(
    const QString& taskId,
    const QString& projectRoot,
    const QJsonObject& options)
{
    return QJsonObject{{field::taskId(), taskId},
        {QStringLiteral("projectRoot"), projectRoot},
        {field::options(), options}};
}

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
    const QJsonObject& options)
{
    QJsonObject payload;
    payload.insert(field::taskId(), taskId);
    payload.insert(QStringLiteral("projectRoot"), projectRoot);
    payload.insert(QStringLiteral("modelPackageId"), modelPackageId);
    payload.insert(QStringLiteral("runtimeRoute"), runtimeRoute);
    payload.insert(field::sampleDatasetId(), sampleDatasetId);
    payload.insert(field::sampleDatasetVersionId(), sampleDatasetVersionId);
    payload.insert(field::sampleSnapshotId(), sampleSnapshotId);
    payload.insert(field::sampleSnapshotArtifactId(), sampleSnapshotArtifactId);
    payload.insert(field::sampleRelativePath(), sampleRelativePath);
    payload.insert(field::options(), options);
    return payload;
}

QJsonObject modelImportRequest(
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

QString taskCommandType(const TaskCommand& command)
{
    return std::visit([](const auto& value) -> QString {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, EnvironmentCheckCommand>) {
            return command::runEnvironmentCheckWorkflow();
        } else if constexpr (std::is_same_v<T, DatasetSplitCommand>) {
            return command::runDatasetSplitWorkflow();
        } else if constexpr (std::is_same_v<T, DatasetConversionCommand>) {
            return command::runDatasetConversionWorkflow();
        } else if constexpr (std::is_same_v<T, DataQualityCommand>) {
            return command::runDataQualityWorkflow();
        } else if constexpr (std::is_same_v<T, AnnotationSessionCreateCommand>) {
            return command::createAnnotationSession();
        } else if constexpr (std::is_same_v<T, AnnotationSessionSyncCommand>) {
            return command::syncAnnotationSession();
        } else if constexpr (std::is_same_v<T, DatasetSnapshotImportCommand>) {
            return command::runDatasetSnapshotImportWorkflow();
        } else if constexpr (std::is_same_v<T, OcrOfficialReportImportCommand>) {
            return command::importOcrOfficialReports();
        } else if constexpr (std::is_same_v<T, OcrAcceptanceCommand>) {
            return command::runOcrAcceptanceWorkflow();
        } else if constexpr (std::is_same_v<T, DiagnosticsCommand>) {
            return command::runDiagnosticsWorkflow();
        } else if constexpr (std::is_same_v<T, ExternalAcceptanceEvidenceImportCommand>) {
            return command::importExternalAcceptanceEvidence();
        } else if constexpr (std::is_same_v<T, RuntimeDeliveryCommand>) {
            return command::runRuntimeDeliveryWorkflow();
        } else if constexpr (std::is_same_v<T, ModelImportCommand>) {
            return command::importModel();
        } else if constexpr (std::is_same_v<T, TrainingCommand>) {
            return command::runTrainingWorkflow();
        }
        return QString();
    }, command.payload);
}

QJsonObject taskCommandPayload(const TaskCommand& command)
{
    return std::visit([](const auto& value) {
        QJsonObject object;
        putContext(&object, value.context);
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, DatasetSplitCommand>) {
            object.insert(QStringLiteral("sourceDatasetId"), value.sourceDatasetId);
            object.insert(QStringLiteral("sourceDatasetVersionId"), value.sourceDatasetVersionId);
            object.insert(QStringLiteral("sourceSnapshotId"), value.sourceSnapshotId);
            object.insert(QStringLiteral("sourceSnapshotArtifactId"), value.sourceSnapshotArtifactId);
            object.insert(QStringLiteral("targetDatasetId"), value.targetDatasetId);
            object.insert(QStringLiteral("targetDatasetName"), value.targetDatasetName);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, DatasetConversionCommand>) {
            object.insert(field::sourcePath(), value.sourcePath);
            object.insert(field::sourceFormat(), value.sourceFormat);
            object.insert(field::targetFormat(), value.targetFormat);
            object.insert(QStringLiteral("targetDatasetId"), value.targetDatasetId);
            object.insert(QStringLiteral("targetDatasetName"), value.targetDatasetName);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, DataQualityCommand>) {
            object.insert(QStringLiteral("datasetId"), value.datasetId);
            object.insert(QStringLiteral("datasetVersionId"), value.datasetVersionId);
            object.insert(QStringLiteral("snapshotId"), value.snapshotId);
            object.insert(QStringLiteral("snapshotArtifactId"), value.snapshotArtifactId);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, AnnotationSessionCreateCommand>) {
            object.insert(QStringLiteral("repairManifestArtifactId"), value.repairManifestArtifactId);
            object.insert(QStringLiteral("workingDirectory"), value.workingDirectory);
            object.insert(QStringLiteral("toolSummary"), value.toolSummary);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, AnnotationSessionSyncCommand>) {
            object.insert(QStringLiteral("sessionArtifactId"), value.sessionArtifactId);
            object.insert(QStringLiteral("workingDirectory"), value.workingDirectory);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, DatasetSnapshotImportCommand>) {
            object.insert(field::sourcePath(), value.sourcePath);
            object.insert(field::sourceFormat(), value.sourceFormat);
            object.insert(QStringLiteral("targetDatasetId"), value.targetDatasetId);
            object.insert(QStringLiteral("targetDatasetName"), value.targetDatasetName);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, OcrOfficialReportImportCommand>) {
            object.insert(QStringLiteral("det"), value.det);
            object.insert(QStringLiteral("rec"), value.rec);
            object.insert(QStringLiteral("system"), value.system);
            object.insert(QStringLiteral("acceptanceCohortId"), value.acceptanceCohortId);
            object.insert(QStringLiteral("customerDomainId"), value.customerDomainId);
            object.insert(QStringLiteral("evidenceClass"), value.evidenceClass);
        } else if constexpr (std::is_same_v<T, OcrAcceptanceCommand>) {
            object.insert(QStringLiteral("detReportArtifactId"), value.detReportArtifactId);
            object.insert(QStringLiteral("recReportArtifactId"), value.recReportArtifactId);
            object.insert(QStringLiteral("systemReportArtifactId"), value.systemReportArtifactId);
            object.insert(QStringLiteral("thresholds"), value.thresholds);
        } else if constexpr (std::is_same_v<T, DiagnosticsCommand>) {
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, ExternalAcceptanceEvidenceImportCommand>) {
            object.insert(field::sourcePath(), value.sourcePath);
        } else if constexpr (std::is_same_v<T, RuntimeDeliveryCommand>) {
            object.insert(QStringLiteral("modelPackageId"), value.modelPackageId);
            object.insert(QStringLiteral("runtimeRoute"), value.runtimeRoute);
            object.insert(field::sampleDatasetId(), value.sampleDatasetId);
            object.insert(field::sampleDatasetVersionId(), value.sampleDatasetVersionId);
            object.insert(field::sampleSnapshotId(), value.sampleSnapshotId);
            object.insert(field::sampleSnapshotArtifactId(), value.sampleSnapshotArtifactId);
            object.insert(field::sampleRelativePath(), value.sampleRelativePath);
            object.insert(field::options(), value.options);
        } else if constexpr (std::is_same_v<T, ModelImportCommand>) {
            object.insert(QStringLiteral("sourceFilePath"), value.sourceFilePath);
            object.insert(QStringLiteral("manifestDraft"), value.manifestDraft);
        } else if constexpr (std::is_same_v<T, TrainingCommand>) {
            object.insert(QStringLiteral("datasetId"), value.datasetId);
            object.insert(QStringLiteral("datasetVersionId"), value.datasetVersionId);
            object.insert(QStringLiteral("snapshotId"), value.snapshotId);
            object.insert(QStringLiteral("snapshotArtifactId"), value.snapshotArtifactId);
            object.insert(QStringLiteral("capabilityId"), value.capabilityId);
            object.insert(field::taskType(), value.taskType);
            object.insert(QStringLiteral("trainingBackend"), value.trainingBackend);
            if (!value.deploymentSampleRelativePath.isEmpty()) {
                object.insert(QStringLiteral("deploymentSampleRelativePath"), value.deploymentSampleRelativePath);
            }
            object.insert(QStringLiteral("parameters"), value.parameters);
        }
        return object;
    }, command.payload);
}

bool taskCommandFromPayload(const QString& type,
    const QJsonObject& payload,
    TaskCommand* command,
    QString* error)
{
    TaskCommandContext context;
    if (!parseContext(payload, &context, error)) {
        return false;
    }

    auto string = [&](const QString& key, QString* value) {
        return requiredString(payload, key, value, error);
    };
    auto object = [&](const QString& key, QJsonObject* value) {
        return requiredObject(payload, key, value, error);
    };
    auto optional = [&](const QString& key, QJsonObject* value) {
        return optionalObject(payload, key, value, error);
    };

    if (type == command::runEnvironmentCheckWorkflow()) {
        if (!hasOnlyKeys(payload, {QStringLiteral("taskId"), QStringLiteral("projectRoot")}, {}, error)) return false;
        if (command) *command = TaskCommand{EnvironmentCheckCommand{context}};
        return true;
    }
    if (type == command::runDatasetSplitWorkflow()) {
        DatasetSplitCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), QStringLiteral("sourceDatasetId"),
                    QStringLiteral("sourceDatasetVersionId"), QStringLiteral("sourceSnapshotId"),
                    QStringLiteral("sourceSnapshotArtifactId"), QStringLiteral("targetDatasetId"),
                    QStringLiteral("targetDatasetName")}, {field::options()}, error)
            || !string(QStringLiteral("sourceDatasetId"), &value.sourceDatasetId)
            || !string(QStringLiteral("sourceDatasetVersionId"), &value.sourceDatasetVersionId)
            || !string(QStringLiteral("sourceSnapshotId"), &value.sourceSnapshotId)
            || !string(QStringLiteral("sourceSnapshotArtifactId"), &value.sourceSnapshotArtifactId)
            || !string(QStringLiteral("targetDatasetId"), &value.targetDatasetId)
            || !string(QStringLiteral("targetDatasetName"), &value.targetDatasetName)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runDatasetConversionWorkflow()) {
        DatasetConversionCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), field::sourcePath(),
                    field::sourceFormat(), field::targetFormat(), QStringLiteral("targetDatasetId"),
                    QStringLiteral("targetDatasetName")}, {field::options()}, error)
            || !string(field::sourcePath(), &value.sourcePath)
            || !string(field::sourceFormat(), &value.sourceFormat)
            || !string(field::targetFormat(), &value.targetFormat)
            || !string(QStringLiteral("targetDatasetId"), &value.targetDatasetId)
            || !string(QStringLiteral("targetDatasetName"), &value.targetDatasetName)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runDataQualityWorkflow()) {
        DataQualityCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), QStringLiteral("datasetId"),
                    QStringLiteral("datasetVersionId"), QStringLiteral("snapshotId"),
                    QStringLiteral("snapshotArtifactId")}, {field::options()}, error)
            || !string(QStringLiteral("datasetId"), &value.datasetId)
            || !string(QStringLiteral("datasetVersionId"), &value.datasetVersionId)
            || !string(QStringLiteral("snapshotId"), &value.snapshotId)
            || !string(QStringLiteral("snapshotArtifactId"), &value.snapshotArtifactId)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::createAnnotationSession()) {
        AnnotationSessionCreateCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"),
                    QStringLiteral("repairManifestArtifactId"), QStringLiteral("workingDirectory"),
                    QStringLiteral("toolSummary")}, {field::options()}, error)
            || !string(QStringLiteral("repairManifestArtifactId"), &value.repairManifestArtifactId)
            || !string(QStringLiteral("workingDirectory"), &value.workingDirectory)
            || !object(QStringLiteral("toolSummary"), &value.toolSummary)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::syncAnnotationSession()) {
        AnnotationSessionSyncCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"),
                    QStringLiteral("sessionArtifactId"), QStringLiteral("workingDirectory")},
                {field::options()}, error)
            || !string(QStringLiteral("sessionArtifactId"), &value.sessionArtifactId)
            || !string(QStringLiteral("workingDirectory"), &value.workingDirectory)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runDatasetSnapshotImportWorkflow()) {
        DatasetSnapshotImportCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), field::sourcePath(),
                    field::sourceFormat(), QStringLiteral("targetDatasetId"),
                    QStringLiteral("targetDatasetName")}, {field::options()}, error)
            || !string(field::sourcePath(), &value.sourcePath)
            || !string(field::sourceFormat(), &value.sourceFormat)
            || !string(QStringLiteral("targetDatasetId"), &value.targetDatasetId)
            || !string(QStringLiteral("targetDatasetName"), &value.targetDatasetName)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::importOcrOfficialReports()) {
        OcrOfficialReportImportCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), QStringLiteral("det"),
                    QStringLiteral("rec"), QStringLiteral("system"), QStringLiteral("acceptanceCohortId"),
                    QStringLiteral("customerDomainId"), QStringLiteral("evidenceClass")}, {}, error)
            || !object(QStringLiteral("det"), &value.det)
            || !object(QStringLiteral("rec"), &value.rec)
            || !object(QStringLiteral("system"), &value.system)
            || !string(QStringLiteral("acceptanceCohortId"), &value.acceptanceCohortId)
            || !string(QStringLiteral("customerDomainId"), &value.customerDomainId)
            || !string(QStringLiteral("evidenceClass"), &value.evidenceClass)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runOcrAcceptanceWorkflow()) {
        OcrAcceptanceCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"),
                    QStringLiteral("detReportArtifactId"), QStringLiteral("recReportArtifactId"),
                    QStringLiteral("systemReportArtifactId"), QStringLiteral("thresholds")}, {}, error)
            || !string(QStringLiteral("detReportArtifactId"), &value.detReportArtifactId)
            || !string(QStringLiteral("recReportArtifactId"), &value.recReportArtifactId)
            || !string(QStringLiteral("systemReportArtifactId"), &value.systemReportArtifactId)
            || !object(QStringLiteral("thresholds"), &value.thresholds)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runDiagnosticsWorkflow()) {
        DiagnosticsCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot")}, {field::options()}, error)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::importExternalAcceptanceEvidence()) {
        ExternalAcceptanceEvidenceImportCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), field::sourcePath()}, {}, error)
            || !string(field::sourcePath(), &value.sourcePath)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runRuntimeDeliveryWorkflow()) {
        RuntimeDeliveryCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), QStringLiteral("modelPackageId"),
                    QStringLiteral("runtimeRoute"), field::sampleDatasetId(), field::sampleDatasetVersionId(),
                    field::sampleSnapshotId(), field::sampleSnapshotArtifactId(), field::sampleRelativePath()},
                {field::options()}, error)
            || !string(QStringLiteral("modelPackageId"), &value.modelPackageId)
            || !string(QStringLiteral("runtimeRoute"), &value.runtimeRoute)
            || !string(field::sampleDatasetId(), &value.sampleDatasetId)
            || !string(field::sampleDatasetVersionId(), &value.sampleDatasetVersionId)
            || !string(field::sampleSnapshotId(), &value.sampleSnapshotId)
            || !string(field::sampleSnapshotArtifactId(), &value.sampleSnapshotArtifactId)
            || !string(field::sampleRelativePath(), &value.sampleRelativePath)
            || !optional(field::options(), &value.options)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::importModel()) {
        ModelImportCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"),
                    QStringLiteral("sourceFilePath"), QStringLiteral("manifestDraft")}, {}, error)
            || !string(QStringLiteral("sourceFilePath"), &value.sourceFilePath)
            || !object(QStringLiteral("manifestDraft"), &value.manifestDraft)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }
    if (type == command::runTrainingWorkflow()) {
        TrainingCommand value{context};
        if (!hasOnlyKeys(payload,
                {QStringLiteral("taskId"), QStringLiteral("projectRoot"), QStringLiteral("datasetId"),
                    QStringLiteral("datasetVersionId"), QStringLiteral("snapshotId"),
                    QStringLiteral("snapshotArtifactId"), QStringLiteral("capabilityId"),
                    field::taskType(), QStringLiteral("trainingBackend")},
                {QStringLiteral("deploymentSampleRelativePath"), QStringLiteral("parameters")}, error)
            || !string(QStringLiteral("datasetId"), &value.datasetId)
            || !string(QStringLiteral("datasetVersionId"), &value.datasetVersionId)
            || !string(QStringLiteral("snapshotId"), &value.snapshotId)
            || !string(QStringLiteral("snapshotArtifactId"), &value.snapshotArtifactId)
            || !string(QStringLiteral("capabilityId"), &value.capabilityId)
            || !string(field::taskType(), &value.taskType)
            || !string(QStringLiteral("trainingBackend"), &value.trainingBackend)) return false;
        if (payload.contains(QStringLiteral("deploymentSampleRelativePath"))
            && !string(QStringLiteral("deploymentSampleRelativePath"), &value.deploymentSampleRelativePath)) return false;
        if (!optional(QStringLiteral("parameters"), &value.parameters)) return false;
        if (command) *command = TaskCommand{value};
        return true;
    }

    if (error) *error = QStringLiteral("Unsupported task command type: %1").arg(type);
    return false;
}

TaskEventKind taskEventKindFromType(const QString& type, bool* ok)
{
    const QString normalized = type.trimmed();
    bool recognized = true;
    TaskEventKind kind = TaskEventKind::Result;
    if (normalized == event::ready()) kind = TaskEventKind::Ready;
    else if (normalized == event::log()) kind = TaskEventKind::Log;
    else if (normalized == event::progress()) kind = TaskEventKind::Progress;
    else if (normalized == event::metric()) kind = TaskEventKind::Metric;
    else if (normalized == event::artifact()) kind = TaskEventKind::Artifact;
    else if (normalized == event::completed()) kind = TaskEventKind::Succeeded;
    else if (normalized == event::failed()) kind = TaskEventKind::Failed;
    else if (normalized == event::canceled()) kind = TaskEventKind::Canceled;
    else if (normalized == event::environmentCheckWorkflow()
        || normalized == event::datasetSplitWorkflow()
        || normalized == event::datasetConversionWorkflow()
        || normalized == event::dataQualityWorkflow()
        || normalized == event::diagnosticsWorkflow()
        || normalized == event::annotationSession()
        || normalized == event::annotationSync()
        || normalized == event::datasetSnapshotImportWorkflow()
        || normalized == event::runtimeDeliveryWorkflow()
        || normalized == event::modelImport()
        || normalized == event::ocrOfficialReportsImported()
        || normalized == event::ocrAcceptanceWorkflow()
        || normalized == event::externalAcceptanceEvidenceImported()) {
        kind = TaskEventKind::Result;
    } else {
        recognized = false;
    }
    if (ok) *ok = recognized;
    return kind;
}

QString taskEventType(const TaskEvent& eventValue)
{
    if (eventValue.kind == TaskEventKind::Result && !eventValue.resultType.isEmpty()) {
        return eventValue.resultType;
    }
    return commandTypeForKind(eventValue.kind);
}

TaskEvent taskEventFromType(const QString& type, const QJsonObject& details)
{
    bool ok = false;
    const TaskEventKind kind = taskEventKindFromType(type, &ok);
    TaskEvent result;
    result.kind = ok ? kind : TaskEventKind::Failed;
    result.details = details;
    QString taskIdText;
    if (requiredString(details, QStringLiteral("taskId"), &taskIdText, nullptr)) {
        TaskId::parse(taskIdText, &result.taskId, nullptr);
    }
    if (kind == TaskEventKind::Result) {
        result.resultType = type;
    }
    return result;
}

namespace control {
aitrain::ProtocolEnvelope startTaskEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const TaskCommand& command,
    const QString& controlToken)
{
    QJsonObject payload = taskCommandPayload(command);
    payload.insert(QStringLiteral("schema"), taskCommandSchema());
    payload.insert(QStringLiteral("type"), taskCommandType(command));
    if (!taskId.isValid()) {
        return aitrain::ProtocolEnvelope();
    }
    if (command.payload.index() == std::variant_npos) {
        return aitrain::ProtocolEnvelope();
    }
    return makeControlEnvelope(requestId, taskId, sequence, QStringLiteral("command.start_task"),
        payload, controlToken);
}

aitrain::ProtocolEnvelope cancelTaskEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const QString& controlToken)
{
    return makeControlEnvelope(requestId, taskId, sequence, QStringLiteral("command.cancel_task"),
        QJsonObject{}, controlToken);
}

aitrain::ProtocolEnvelope eventEnvelope(
    const aitrain::RequestId& requestId,
    const aitrain::TaskId& taskId,
    quint64 sequence,
    const TaskEvent& eventValue,
    const QString& controlToken)
{
    if (!requestId.isValid() || !taskId.isValid()) {
        return aitrain::ProtocolEnvelope();
    }
    if (eventValue.taskId.isValid() && eventValue.taskId != taskId) {
        return aitrain::ProtocolEnvelope();
    }
    QJsonObject payload = eventValue.details;
    const QJsonValue existingTaskId = payload.value(QStringLiteral("taskId"));
    // Protocol-level rejection may be emitted before Worker has assigned its
    // active task, so that diagnostic payload legitimately carries an empty
    // taskId. An explicit non-empty value must still match the envelope.
    if (!existingTaskId.isUndefined()
        && (!existingTaskId.isString()
            || (!existingTaskId.toString().trimmed().isEmpty()
                && existingTaskId.toString().trimmed() != taskId.toString()))) {
        return aitrain::ProtocolEnvelope();
    }
    // Ready 事件在 Worker 尚未接收 start_task 时没有 activeTaskId；身份仍由
    // control envelope 提供，不能让 payload 缺少 TaskId。
    payload.insert(QStringLiteral("taskId"), taskId.toString());
    payload.insert(QStringLiteral("schema"), taskEventSchema());
    const QString eventType = eventValue.kind == TaskEventKind::Result
        ? QStringLiteral("result")
        : taskEventType(eventValue);
    payload.insert(QStringLiteral("type"), eventType);
    if (eventValue.kind == TaskEventKind::Result && !eventValue.resultType.isEmpty()) {
        payload.insert(QStringLiteral("resultType"), eventValue.resultType);
    }
    QString validationError;
    if (!validateTaskEventDetails(eventValue.kind, eventValue.resultType, payload, &validationError)) {
        return aitrain::ProtocolEnvelope();
    }
    return makeControlEnvelope(requestId, taskId, sequence,
        eventType == QStringLiteral("result") ? QStringLiteral("event.result") : controlEventKind(eventType),
        payload, controlToken);
}

bool unpackStartTask(const aitrain::ProtocolEnvelope& envelope,
    TaskCommand* command,
    QString* error)
{
    if (!envelope.taskId.isValid()) {
        if (error) *error = QStringLiteral("command.start_task 缺少有效 envelope taskId。");
        return false;
    }
    if (envelope.kind != QStringLiteral("command.start_task")) {
        if (error) *error = QStringLiteral("不是 command.start_task envelope。");
        return false;
    }
    const QString schema = envelope.payload.value(QStringLiteral("schema")).toString();
    const QString type = envelope.payload.value(QStringLiteral("type")).toString().trimmed();
    if (schema != taskCommandSchema() || type.isEmpty()) {
        if (error) *error = QStringLiteral("command.start_task 必须包含有效 schema/type。");
        return false;
    }
    QJsonObject payload = envelope.payload;
    payload.remove(QStringLiteral("schema"));
    payload.remove(QStringLiteral("type"));
    TaskCommand decoded;
    if (!taskCommandFromPayload(type, payload, &decoded, error)) {
        return false;
    }
    const QString commandTaskId = std::visit([](const auto& value) {
        return value.context.taskId.toString();
    }, decoded.payload);
    if (commandTaskId != envelope.taskId.toString()) {
        if (error) *error = QStringLiteral("命令 taskId 与 envelope 不一致。");
        return false;
    }
    if (command) *command = decoded;
    return true;
}

bool unpackTaskEvent(const aitrain::ProtocolEnvelope& envelope,
    TaskEvent* eventValue,
    QString* error)
{
    if (!envelope.taskId.isValid()) {
        if (error) *error = QStringLiteral("事件缺少有效 envelope taskId。");
        return false;
    }
    if (!envelope.kind.startsWith(QStringLiteral("event."))) {
        if (error) *error = QStringLiteral("不是 event envelope。");
        return false;
    }
    if (envelope.payload.value(QStringLiteral("schema")).toString() != taskEventSchema()) {
        if (error) *error = QStringLiteral("事件 schema 无效。");
        return false;
    }
    const QString type = envelope.payload.value(QStringLiteral("type")).toString().trimmed();
    bool recognized = false;
    TaskEventKind kind = taskEventKindFromType(type, &recognized);
    QString resultType;
    if (type == QStringLiteral("result")) {
        resultType = envelope.payload.value(QStringLiteral("resultType")).toString().trimmed();
        kind = taskEventKindFromType(resultType, &recognized);
        if (kind != TaskEventKind::Result) {
            // resultType 必须是当前已知的 Workflow result；未知结果不会进入 GUI。
            recognized = false;
        }
    }
    if (!recognized || (type == QStringLiteral("result") && resultType.isEmpty())) {
        if (error) *error = QStringLiteral("未知或无效的 TaskEvent type。");
        return false;
    }
    const QString expectedKind = type == QStringLiteral("result")
        ? QStringLiteral("event.result")
        : controlEventKind(type);
    if (expectedKind != envelope.kind) {
        if (error) *error = QStringLiteral("事件 kind 与 type 不一致。");
        return false;
    }
    QJsonObject details = envelope.payload;
    details.remove(QStringLiteral("schema"));
    details.remove(QStringLiteral("type"));
    details.remove(QStringLiteral("resultType"));
    QString eventTaskIdText;
    if (!requiredString(details, QStringLiteral("taskId"), &eventTaskIdText, error)) {
        return false;
    }
    TaskId eventTaskId;
    if (!TaskId::parse(eventTaskIdText, &eventTaskId, error)
        || eventTaskId != envelope.taskId) {
        if (error) *error = QStringLiteral("事件 taskId 与 envelope 不一致。");
        return false;
    }
    if (!validateTaskEventDetails(
            type == QStringLiteral("result") ? TaskEventKind::Result : kind,
            resultType, details, error)) {
        return false;
    }
    TaskEvent decoded;
    decoded.kind = type == QStringLiteral("result") ? TaskEventKind::Result : kind;
    decoded.taskId = envelope.taskId;
    decoded.resultType = resultType;
    decoded.details = details;
    if (eventValue) *eventValue = decoded;
    return true;
}
} // namespace control

} // namespace worker_protocol
} // namespace aitrain
