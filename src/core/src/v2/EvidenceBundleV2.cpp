#include "aitrain/v2/EvidenceBundleV2.h"

#include <QJsonArray>
#include <QSet>

#include <algorithm>

namespace aitrain::v2 {
namespace {

QString timestamp(const QDateTime& value)
{
    return value.toUTC().toString(Qt::ISODateWithMs);
}

bool parseTimestamp(const QJsonValue& value, QDateTime* result, QString* error, const QString& field)
{
    const QDateTime parsed = QDateTime::fromString(value.toString(), Qt::ISODate);
    if (!parsed.isValid()) {
        if (error) *error = QStringLiteral("Evidence Bundle 字段 %1 必须是有效 ISO-8601 UTC 时间。").arg(field);
        return false;
    }
    if (result) *result = parsed.toUTC();
    return true;
}

bool isSha256(const QString& value)
{
    if (value.size() != 64) return false;
    return std::all_of(value.cbegin(), value.cend(), [](QChar character) {
        const ushort code = character.unicode();
        return (code >= '0' && code <= '9') || (code >= 'a' && code <= 'f');
    });
}

QJsonObject encodeFailure(const Failure& failure)
{
    return {
        {QStringLiteral("code"), failureCodeToString(failure.code)},
        {QStringLiteral("message"), failure.message},
        {QStringLiteral("suggestedAction"), failure.suggestedAction},
        {QStringLiteral("occurredAt"), failure.occurredAt.isValid() ? timestamp(failure.occurredAt) : QString()}
    };
}

bool decodeFailure(const QJsonObject& object, Failure* result, QString* error)
{
    Failure failure;
    if (!failureCodeFromString(object.value(QStringLiteral("code")).toString(), &failure.code)) {
        if (error) *error = QStringLiteral("Evidence Bundle failure.code 无效。");
        return false;
    }
    failure.message = object.value(QStringLiteral("message")).toString();
    failure.suggestedAction = object.value(QStringLiteral("suggestedAction")).toString();
    const QString occurredAt = object.value(QStringLiteral("occurredAt")).toString();
    if (!occurredAt.isEmpty()
        && !parseTimestamp(object.value(QStringLiteral("occurredAt")), &failure.occurredAt, error, QStringLiteral("failure.occurredAt"))) {
        return false;
    }
    if (result) *result = failure;
    return true;
}

QJsonObject encodeTask(const TaskSnapshot& task)
{
    return {
        {QStringLiteral("id"), task.id.toString()},
        {QStringLiteral("requestId"), task.requestId.toString()},
        {QStringLiteral("state"), taskStateToString(task.state)},
        {QStringLiteral("capabilityId"), task.capabilityId},
        {QStringLiteral("taskType"), task.taskType},
        {QStringLiteral("createdAt"), timestamp(task.createdAt)},
        {QStringLiteral("updatedAt"), timestamp(task.updatedAt)},
        {QStringLiteral("failure"), encodeFailure(task.failure)}
    };
}

bool decodeTask(const QJsonObject& object, TaskSnapshot* result, QString* error)
{
    TaskSnapshot task;
    if (!TaskId::parse(object.value(QStringLiteral("id")).toString(), &task.id, error)
        || !RequestId::parse(object.value(QStringLiteral("requestId")).toString(), &task.requestId, error)
        || !taskStateFromString(object.value(QStringLiteral("state")).toString(), &task.state)
        || !parseTimestamp(object.value(QStringLiteral("createdAt")), &task.createdAt, error, QStringLiteral("task.createdAt"))
        || !parseTimestamp(object.value(QStringLiteral("updatedAt")), &task.updatedAt, error, QStringLiteral("task.updatedAt"))
        || !decodeFailure(object.value(QStringLiteral("failure")).toObject(), &task.failure, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("Evidence Bundle task 无效。");
        return false;
    }
    task.capabilityId = object.value(QStringLiteral("capabilityId")).toString();
    task.taskType = object.value(QStringLiteral("taskType")).toString();
    if (result) *result = task;
    return true;
}

} // namespace

bool validateEvidenceBundleV2(const EvidenceBundleV2& bundle, QString* error)
{
    if (bundle.schemaVersion != kEvidenceBundleV2SchemaVersion || bundle.projectIdentity.trimmed().isEmpty()
        || !bundle.task.id.isValid() || !bundle.task.requestId.isValid() || !isTerminalTaskState(bundle.task.state)
        || bundle.task.capabilityId.trimmed().isEmpty() || bundle.task.taskType.trimmed().isEmpty()
        || !bundle.createdAt.isValid() || bundle.backendEnvironment.isEmpty()) {
        if (error) *error = QStringLiteral("Evidence Bundle 缺少必需的项目、终态任务、后端环境或创建时间事实。");
        return false;
    }
    if (bundle.task.state == TaskState::Succeeded && bundle.task.failure.isFailure()) {
        if (error) *error = QStringLiteral("成功 Evidence Bundle 不能携带 Failure。");
        return false;
    }
    if (bundle.task.state != TaskState::Succeeded && !bundle.task.failure.isFailure()) {
        if (error) *error = QStringLiteral("非成功 Evidence Bundle 必须携带 Failure。");
        return false;
    }
    if (bundle.task.state != TaskState::Succeeded
        && (bundle.task.failure.message.trimmed().isEmpty() || !bundle.task.failure.occurredAt.isValid())) {
        if (error) *error = QStringLiteral("非成功 Evidence Bundle 的 Failure 必须包含非空消息和有效发生时间。");
        return false;
    }
    QSet<QString> artifactIds;
    for (const EvidenceArtifactV2& artifact : bundle.artifacts) {
        if (!artifact.artifactId.isValid() || artifact.kind.trimmed().isEmpty()
            || artifactIds.contains(artifact.artifactId.toString())) {
            if (error) *error = QStringLiteral("Evidence Bundle 必须只索引不重复的已提交 Artifact ID。");
            return false;
        }
        artifactIds.insert(artifact.artifactId.toString());
    }
    QSet<QString> externalRoles;
    for (const EvidenceExternalInputV2& input : bundle.externalInputs) {
        if (input.role.trimmed().isEmpty() || externalRoles.contains(input.role)
            || !input.producerTaskId.isValid() || !input.artifactId.isValid()
            || !input.datasetId.isValid() || !input.datasetVersionId.isValid()
            || !input.datasetSnapshotId.isValid() || !isSha256(input.manifestSha256)
            || !isSha256(input.rootHash)) {
            if (error) *error = QStringLiteral("Evidence Bundle 外部输入必须包含唯一角色、完整生产者身份和小写 SHA-256 摘要。");
            return false;
        }
        externalRoles.insert(input.role);
    }
    return true;
}

QJsonObject encodeEvidenceBundleV2(const EvidenceBundleV2& bundle, QString* error)
{
    if (!validateEvidenceBundleV2(bundle, error)) return {};
    QJsonArray artifacts;
    for (const EvidenceArtifactV2& artifact : bundle.artifacts) {
        artifacts.append(QJsonObject{{QStringLiteral("artifactId"), artifact.artifactId.toString()},
            {QStringLiteral("kind"), artifact.kind}, {QStringLiteral("facts"), artifact.facts}});
    }
    QJsonArray limitations;
    for (const QString& limitation : bundle.limitations) limitations.append(limitation);
    QJsonArray externalInputs;
    for (const EvidenceExternalInputV2& input : bundle.externalInputs) {
        externalInputs.append(QJsonObject{{QStringLiteral("role"), input.role},
            {QStringLiteral("producerTaskId"), input.producerTaskId.toString()},
            {QStringLiteral("artifactId"), input.artifactId.toString()},
            {QStringLiteral("datasetId"), input.datasetId.toString()},
            {QStringLiteral("datasetVersionId"), input.datasetVersionId.toString()},
            {QStringLiteral("datasetSnapshotId"), input.datasetSnapshotId.toString()},
            {QStringLiteral("manifestSha256"), input.manifestSha256},
            {QStringLiteral("rootHash"), input.rootHash}});
    }
    return {
        {QStringLiteral("schemaVersion"), bundle.schemaVersion},
        {QStringLiteral("kind"), QStringLiteral("aitrain_evidence_bundle_v2")},
        {QStringLiteral("projectIdentity"), bundle.projectIdentity},
        {QStringLiteral("task"), encodeTask(bundle.task)},
        {QStringLiteral("workflowRunId"), bundle.workflowRunId.toString()},
        {QStringLiteral("datasetSnapshotId"), bundle.datasetSnapshotId.toString()},
        {QStringLiteral("backendEnvironment"), bundle.backendEnvironment},
        {QStringLiteral("parameters"), bundle.parameters},
        {QStringLiteral("metrics"), bundle.metrics},
        {QStringLiteral("runtimeStatus"), bundle.runtimeStatus},
        {QStringLiteral("evaluation"), bundle.evaluation},
        {QStringLiteral("benchmark"), bundle.benchmark},
        {QStringLiteral("externalInputs"), externalInputs},
        {QStringLiteral("artifacts"), artifacts},
        {QStringLiteral("limitations"), limitations},
        {QStringLiteral("createdAt"), timestamp(bundle.createdAt)}
    };
}

bool decodeEvidenceBundleV2(const QJsonObject& object, EvidenceBundleV2* bundle, QString* error)
{
    if (!bundle || object.value(QStringLiteral("schemaVersion")).toInt(-1) != kEvidenceBundleV2SchemaVersion
        || object.value(QStringLiteral("kind")).toString() != QStringLiteral("aitrain_evidence_bundle_v2")) {
        if (error) *error = QStringLiteral("不支持的 Evidence Bundle V2。");
        return false;
    }
    EvidenceBundleV2 parsed;
    parsed.projectIdentity = object.value(QStringLiteral("projectIdentity")).toString();
    if (!decodeTask(object.value(QStringLiteral("task")).toObject(), &parsed.task, error)
        || !parseTimestamp(object.value(QStringLiteral("createdAt")), &parsed.createdAt, error, QStringLiteral("createdAt"))) {
        return false;
    }
    const QString workflowRunId = object.value(QStringLiteral("workflowRunId")).toString();
    if (!workflowRunId.isEmpty() && !WorkflowRunId::parse(workflowRunId, &parsed.workflowRunId, error)) return false;
    const QString datasetSnapshotId = object.value(QStringLiteral("datasetSnapshotId")).toString();
    if (!datasetSnapshotId.isEmpty() && !SnapshotId::parse(datasetSnapshotId, &parsed.datasetSnapshotId, error)) return false;
    parsed.backendEnvironment = object.value(QStringLiteral("backendEnvironment")).toObject();
    parsed.parameters = object.value(QStringLiteral("parameters")).toObject();
    parsed.metrics = object.value(QStringLiteral("metrics")).toObject();
    parsed.runtimeStatus = object.value(QStringLiteral("runtimeStatus")).toObject();
    parsed.evaluation = object.value(QStringLiteral("evaluation")).toObject();
    parsed.benchmark = object.value(QStringLiteral("benchmark")).toObject();
    for (const QJsonValue& value : object.value(QStringLiteral("externalInputs")).toArray()) {
        const QJsonObject inputObject = value.toObject();
        EvidenceExternalInputV2 input;
        input.role = inputObject.value(QStringLiteral("role")).toString();
        if (!TaskId::parse(inputObject.value(QStringLiteral("producerTaskId")).toString(), &input.producerTaskId, error)
            || !ArtifactId::parse(inputObject.value(QStringLiteral("artifactId")).toString(), &input.artifactId, error)
            || !DatasetId::parse(inputObject.value(QStringLiteral("datasetId")).toString(), &input.datasetId, error)
            || !DatasetVersionId::parse(inputObject.value(QStringLiteral("datasetVersionId")).toString(), &input.datasetVersionId, error)
            || !SnapshotId::parse(inputObject.value(QStringLiteral("datasetSnapshotId")).toString(), &input.datasetSnapshotId, error)) {
            return false;
        }
        input.manifestSha256 = inputObject.value(QStringLiteral("manifestSha256")).toString();
        input.rootHash = inputObject.value(QStringLiteral("rootHash")).toString();
        parsed.externalInputs.append(input);
    }
    for (const QJsonValue& value : object.value(QStringLiteral("artifacts")).toArray()) {
        const QJsonObject artifactObject = value.toObject();
        EvidenceArtifactV2 artifact;
        if (!ArtifactId::parse(artifactObject.value(QStringLiteral("artifactId")).toString(), &artifact.artifactId, error)) return false;
        artifact.kind = artifactObject.value(QStringLiteral("kind")).toString();
        artifact.facts = artifactObject.value(QStringLiteral("facts")).toObject();
        parsed.artifacts.append(artifact);
    }
    for (const QJsonValue& value : object.value(QStringLiteral("limitations")).toArray()) {
        parsed.limitations.append(value.toString());
    }
    if (!validateEvidenceBundleV2(parsed, error)) return false;
    *bundle = parsed;
    return true;
}

} // namespace aitrain::v2
