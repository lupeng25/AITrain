#include "aitrain/storage/WorkflowTerminalizationStore.h"

#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "aitrain/storage/WorkflowRepository.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

namespace aitrain {
namespace {

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool parseState(const QString& value, WorkflowTerminalizationState* result)
{
    if (value == QStringLiteral("sealed")) {
        *result = WorkflowTerminalizationState::Sealed;
        return true;
    }
    if (value == QStringLiteral("evidence_attached")) {
        *result = WorkflowTerminalizationState::EvidenceAttached;
        return true;
    }
    if (value == QStringLiteral("closed")) {
        *result = WorkflowTerminalizationState::Closed;
        return true;
    }
    return false;
}

bool parseSnapshot(QSqlQuery& query, WorkflowTerminalizationSnapshot* result,
    QString* error)
{
    WorkflowTerminalizationSnapshot parsed;
    if (!WorkflowRunId::parse(query.value(0).toString(),
            &parsed.workflowRunId, error)
        || !TaskId::parse(query.value(1).toString(), &parsed.taskId, error)
        || !parseState(query.value(2).toString(), &parsed.state)
        || !taskStateFromString(query.value(3).toString(),
            &parsed.terminalState)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("工作流终态封存记录包含无效状态或标识。");
        }
        return false;
    }
    if (!failureCodeFromString(query.value(4).toString(),
            &parsed.failure.code)) {
        if (error) *error = QStringLiteral("工作流终态记录包含未知 FailureCode。");
        return false;
    }
    parsed.failure.message = query.value(5).toString();
    parsed.failure.suggestedAction = query.value(6).toString();
    parsed.failure.occurredAt = parseUtc(query.value(7).toString());
    parsed.terminalAt = parseUtc(query.value(8).toString());
    if (!query.value(9).toString().isEmpty()
        && !ArtifactId::parse(query.value(9).toString(),
            &parsed.evidenceArtifactId, error)) {
        return false;
    }
    parsed.evidenceAttemptCount = query.value(10).toInt();
    if (!failureCodeFromString(query.value(11).toString(),
            &parsed.lastEvidenceFailure.code)) {
        if (error) *error = QStringLiteral("Evidence 失败记录包含未知 FailureCode。");
        return false;
    }
    parsed.lastEvidenceFailure.message = query.value(12).toString();
    parsed.lastEvidenceFailure.suggestedAction = query.value(13).toString();
    parsed.lastEvidenceFailure.occurredAt = parseUtc(query.value(14).toString());
    parsed.sealedAt = parseUtc(query.value(15).toString());
    parsed.evidenceAttachedAt = parseUtc(query.value(16).toString());
    parsed.closedAt = parseUtc(query.value(17).toString());
    if (!isTerminalTaskState(parsed.terminalState)
        || !parsed.terminalAt.isValid() || !parsed.sealedAt.isValid()
        || parsed.evidenceAttemptCount < 0) {
        if (error) *error = QStringLiteral("工作流终态封存记录字段无效。");
        return false;
    }
    *result = parsed;
    return true;
}

} // namespace

WorkflowTerminalizationStore::WorkflowTerminalizationStore(
    const ProjectDatabase& database)
    : database_(database)
{
}

bool WorkflowTerminalizationStore::exists(
    const WorkflowRunId& workflowRunId, bool* result, QString* error) const
{
    if (!database_.isOpen() || !workflowRunId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "检查工作流终态封存需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select 1 from workflow_terminalizations where workflow_run_id = :id"));
    query.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return false;
    }
    *result = query.next();
    return true;
}

bool WorkflowTerminalizationStore::read(
    const WorkflowRunId& workflowRunId,
    WorkflowTerminalizationSnapshot* result, QString* error) const
{
    if (!database_.isOpen() || !workflowRunId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询工作流终态封存需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select workflow_run_id, task_id, state, terminal_state, failure_code, "
        "failure_details, failure_suggested_action, "
        "coalesce(failure_occurred_at,''), terminal_at, "
        "coalesce(evidence_artifact_id,''), evidence_attempt_count, "
        "last_evidence_failure_code, last_evidence_failure_details, "
        "last_evidence_failure_suggested_action, "
        "coalesce(last_evidence_failure_occurred_at,''), sealed_at, "
        "coalesce(evidence_attached_at,''), coalesce(closed_at,'') "
        "from workflow_terminalizations where workflow_run_id = :id"));
    query.bindValue(QStringLiteral(":id"), workflowRunId.toString());
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text()
                : QStringLiteral("工作流终态封存不存在。");
        }
        return false;
    }
    return parseSnapshot(query, result, error);
}

QVector<WorkflowTerminalizationSnapshot>
WorkflowTerminalizationStore::pending(int limit, QString* error) const
{
    QVector<WorkflowTerminalizationSnapshot> results;
    if (!database_.isOpen() || limit <= 0) {
        if (error) {
            *error = QStringLiteral(
                "查询待恢复工作流终态需要已打开的数据库和正数 limit。");
        }
        return results;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select workflow_run_id from workflow_terminalizations "
        "where state in ('sealed','evidence_attached') "
        "order by sealed_at asc, workflow_run_id asc limit :limit"));
    query.bindValue(QStringLiteral(":limit"), limit);
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    QVector<WorkflowRunId> ids;
    while (query.next()) {
        WorkflowRunId id;
        if (!WorkflowRunId::parse(query.value(0).toString(), &id, error)) {
            return {};
        }
        ids.append(id);
    }
    query.finish();
    for (const WorkflowRunId& id : ids) {
        WorkflowTerminalizationSnapshot item;
        if (!read(id, &item, error)) return {};
        results.append(item);
    }
    return results;
}

QVector<WorkflowRunSnapshot>
WorkflowTerminalizationStore::pendingEvidenceRequired(
    int limit, QString* error) const
{
    QVector<WorkflowRunSnapshot> results;
    if (!database_.isOpen() || limit <= 0) {
        if (error) {
            *error = QStringLiteral(
                "查询待恢复 Evidence 门控工作流需要已打开的数据库和正数 limit。");
        }
        return results;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select w.id from workflow_runs w "
        "join tasks t on t.id = w.task_id "
        "left join workflow_terminalizations z on z.workflow_run_id = w.id "
        "where w.terminal_policy = 'evidence_required' "
        "and t.state in ('queued','starting','running','cancel_requested') "
        "and (z.workflow_run_id is null or z.state <> 'closed') "
        "order by w.created_at asc, w.id asc limit :limit"));
    query.bindValue(QStringLiteral(":limit"), limit);
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    QVector<WorkflowRunId> ids;
    while (query.next()) {
        WorkflowRunId id;
        if (!WorkflowRunId::parse(query.value(0).toString(), &id, error)) {
            return {};
        }
        ids.append(id);
    }
    query.finish();
    const WorkflowRepository workflows(database_);
    for (const WorkflowRunId& id : ids) {
        WorkflowRunSnapshot workflow;
        if (!workflows.readRun(id, &workflow, error)) return {};
        results.append(workflow);
    }
    return results;
}

} // namespace aitrain
