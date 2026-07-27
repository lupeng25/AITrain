#include "aitrain/storage/WorkflowRepository.h"

#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "StoragePagination.h"

#include <QJsonDocument>
#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

namespace aitrain {
namespace {

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool parsePolicy(const QString& value, WorkflowTerminalPolicy* result)
{
    if (value == QStringLiteral("immediate")) {
        *result = WorkflowTerminalPolicy::Immediate;
        return true;
    }
    if (value == QStringLiteral("evidence_required")) {
        *result = WorkflowTerminalPolicy::EvidenceRequired;
        return true;
    }
    return false;
}

bool parseStep(QSqlQuery& query, WorkflowStepSnapshot* result,
    QString* error)
{
    WorkflowStepSnapshot parsed;
    if (!WorkflowStepId::parse(query.value(0).toString(), &parsed.id, error)
        || !WorkflowRunId::parse(query.value(1).toString(),
            &parsed.workflowRunId, error)
        || !workflowStepStateFromString(
            query.value(4).toString(), &parsed.state)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral(
                "工作流步骤记录包含无效状态或标识。");
        }
        return false;
    }
    parsed.ordinal = query.value(2).toInt();
    parsed.kind = query.value(3).toString();
    if ((!query.value(5).toString().isEmpty()
            && !ArtifactId::parse(query.value(5).toString(),
                &parsed.inputArtifactId, error))
        || (!query.value(6).toString().isEmpty()
            && !ArtifactId::parse(query.value(6).toString(),
                &parsed.outputArtifactId, error))) {
        return false;
    }
    parsed.backend = query.value(7).toString();
    QJsonParseError jsonError;
    const QJsonDocument parameters = QJsonDocument::fromJson(
        query.value(8).toString().toUtf8(), &jsonError);
    if (jsonError.error != QJsonParseError::NoError
        || !parameters.isObject()) {
        if (error) {
            *error = QStringLiteral(
                "工作流步骤参数摘要不是对象 JSON。");
        }
        return false;
    }
    parsed.parameterSummary = parameters.object();
    parsed.startedAt = parseUtc(query.value(9).toString());
    parsed.finishedAt = parseUtc(query.value(10).toString());
    if (!failureCodeFromString(
            query.value(11).toString(), &parsed.failure.code)) {
        if (error) {
            *error = QStringLiteral(
                "工作流步骤记录包含未知 FailureCode。");
        }
        return false;
    }
    parsed.failure.message = query.value(12).toString();
    parsed.failure.suggestedAction = query.value(13).toString();
    parsed.failure.occurredAt = parseUtc(query.value(14).toString());
    parsed.retryCount = query.value(15).toInt();
    if (parsed.ordinal < 0 || parsed.kind.isEmpty()
        || parsed.backend.isEmpty() || parsed.retryCount < 0) {
        if (error) *error = QStringLiteral("工作流步骤记录字段无效。");
        return false;
    }
    *result = parsed;
    return true;
}

} // namespace

WorkflowRepository::WorkflowRepository(const ProjectDatabase& database)
    : database_(database)
{
}

bool WorkflowRepository::readRun(const WorkflowRunId& workflowRunId,
    WorkflowRunSnapshot* result, QString* error) const
{
    if (!database_.isOpen() || !workflowRunId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询工作流运行需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select id, task_id, template_id, terminal_policy, created_at "
        "from workflow_runs where id = :id"));
    query.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text() : QStringLiteral("工作流运行不存在。");
        }
        return false;
    }
    WorkflowRunSnapshot parsed;
    if (!WorkflowRunId::parse(query.value(0).toString(), &parsed.id, error)
        || !TaskId::parse(query.value(1).toString(), &parsed.taskId, error)) {
        return false;
    }
    parsed.templateId = query.value(2).toString();
    if (!parsePolicy(query.value(3).toString(), &parsed.terminalPolicy)) {
        if (error) *error = QStringLiteral("工作流运行包含无效终态策略。");
        return false;
    }
    parsed.createdAt = parseUtc(query.value(4).toString());
    if (parsed.templateId.trimmed().isEmpty() || !parsed.createdAt.isValid()) {
        if (error) *error = QStringLiteral("工作流运行记录字段无效。");
        return false;
    }
    *result = parsed;
    return true;
}

bool WorkflowRepository::readInput(const WorkflowRunId& workflowRunId,
    const QString& role, WorkflowInputBinding* result, QString* error) const
{
    const QString normalizedRole = role.trimmed();
    if (!database_.isOpen() || !workflowRunId.isValid()
        || normalizedRole.isEmpty() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询 Workflow 输入需要已打开的数据库、有效运行、角色和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select workflow_run_id, role, source_artifact_id, source_task_id, "
        "source_artifact_kind, coalesce(dataset_id, ''), "
        "coalesce(dataset_snapshot_id, ''), "
        "coalesce(dataset_version_id, ''), coalesce(model_package_id, ''), "
        "manifest_sha256, root_hash, bound_at from workflow_input_bindings "
        "where workflow_run_id = :workflow_run_id and role = :role"));
    query.bindValue(QStringLiteral(":workflow_run_id"),
        workflowRunId.toString());
    query.bindValue(QStringLiteral(":role"), normalizedRole);
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text()
                : QStringLiteral("Workflow 输入绑定不存在。");
        }
        return false;
    }
    WorkflowInputBinding parsed;
    if (!WorkflowRunId::parse(query.value(0).toString(),
            &parsed.workflowRunId, error)
        || !ArtifactId::parse(query.value(2).toString(),
            &parsed.sourceArtifactId, error)
        || !TaskId::parse(query.value(3).toString(),
            &parsed.sourceTaskId, error)) {
        return false;
    }
    parsed.role = query.value(1).toString();
    parsed.sourceArtifactKind = query.value(4).toString();
    const QString datasetId = query.value(5).toString();
    const QString snapshotId = query.value(6).toString();
    const QString datasetVersionId = query.value(7).toString();
    const QString modelPackageId = query.value(8).toString();
    if ((!datasetId.isEmpty()
            && !DatasetId::parse(datasetId, &parsed.datasetId, error))
        || (!snapshotId.isEmpty()
            && !SnapshotId::parse(
                snapshotId, &parsed.datasetSnapshotId, error))
        || (!datasetVersionId.isEmpty()
            && !DatasetVersionId::parse(
                datasetVersionId, &parsed.datasetVersionId, error))
        || (!modelPackageId.isEmpty()
            && !ModelPackageId::parse(
                modelPackageId, &parsed.modelPackageId, error))) {
        return false;
    }
    parsed.manifestSha256 = query.value(9).toString();
    parsed.rootHash = query.value(10).toString();
    parsed.boundAt = parseUtc(query.value(11).toString());
    if (parsed.workflowRunId != workflowRunId
        || parsed.role != normalizedRole
        || parsed.sourceArtifactKind.trimmed().isEmpty()
        || !parsed.boundAt.isValid()) {
        if (error) *error = QStringLiteral("Workflow 输入绑定记录损坏。");
        return false;
    }
    *result = parsed;
    return true;
}

QVector<WorkflowStepSnapshot> WorkflowRepository::steps(
    const WorkflowRunId& workflowRunId, QString* error) const
{
    QVector<WorkflowStepSnapshot> results;
    if (!database_.isOpen() || !workflowRunId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "查询工作流步骤需要已打开的数据库和有效运行 ID。");
        }
        return results;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select id, workflow_run_id, ordinal, kind, state, "
        "coalesce(input_artifact_id, ''), "
        "coalesce(output_artifact_id, ''), backend, "
        "parameter_summary_json, coalesce(started_at, ''), "
        "coalesce(finished_at, ''), failure_code, failure_details, "
        "failure_suggested_action, coalesce(failure_occurred_at, ''), "
        "retry_count from workflow_steps "
        "where workflow_run_id = :workflow_run_id order by ordinal asc"));
    query.bindValue(QStringLiteral(":workflow_run_id"),
        workflowRunId.toString());
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    while (query.next()) {
        WorkflowStepSnapshot step;
        if (!parseStep(query, &step, error)) return {};
        results.append(step);
    }
    return results;
}

Page<WorkflowRunSnapshot> WorkflowRepository::runsForTask(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    Page<WorkflowRunSnapshot> result;
    PageCursor cursor;
    if (!database_.isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "查询任务工作流需要已打开的数据库和有效任务 ID。");
        }
        return result;
    }
    if (!storage_internal::validatePageRequest(request,
            QStringLiteral("workflow_runs"), &cursor, error)) {
        return result;
    }
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select id, task_id, template_id, terminal_policy, created_at "
        "from workflow_runs where task_id = :task_id ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (created_at > :after_time "
            "or (created_at = :after_time and id > :after_id)) ");
    }
    sql += QStringLiteral(
        "order by created_at asc, id asc limit :limit");
    query.prepare(sql);
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":after_time"), cursor.timestamp);
        query.bindValue(QStringLiteral(":after_id"), cursor.id);
    }
    query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    while (query.next()) {
        WorkflowRunSnapshot snapshot;
        if (!WorkflowRunId::parse(
                query.value(0).toString(), &snapshot.id, error)
            || !TaskId::parse(
                query.value(1).toString(), &snapshot.taskId, error)) {
            return {};
        }
        snapshot.templateId = query.value(2).toString();
        if (!parsePolicy(
                query.value(3).toString(), &snapshot.terminalPolicy)) {
            if (error) {
                *error = QStringLiteral(
                    "工作流运行包含无效终态策略。");
            }
            return {};
        }
        snapshot.createdAt = parseUtc(query.value(4).toString());
        if (snapshot.templateId.trimmed().isEmpty()
            || !snapshot.createdAt.isValid()) {
            if (error) {
                *error = QStringLiteral("工作流运行记录字段无效。");
            }
            return {};
        }
        result.items.append(snapshot);
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        const WorkflowRunSnapshot& last = result.items.constLast();
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("workflow_runs"),
            {last.createdAt.toUTC().toString(Qt::ISODateWithMs),
                last.id.toString()});
    }
    return result;
}

} // namespace aitrain
