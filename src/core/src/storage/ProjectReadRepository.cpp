#include "aitrain/storage/ProjectReadRepository.h"

#include "aitrain/domain/ArtifactMemberPath.h"
#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "StoragePagination.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

namespace aitrain {
namespace {

bool scalarCount(const ProjectDatabase& database, const QString& statement,
    qint64* result, QString* error)
{
    QSqlQuery query(database.connection());
    if (!query.exec(statement) || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text()
                : QStringLiteral("项目汇总计数查询没有返回结果。");
        }
        return false;
    }
    bool ok = false;
    const qint64 count = query.value(0).toLongLong(&ok);
    if (!ok || count < 0) {
        if (error) *error = QStringLiteral("项目汇总计数包含无效值。");
        return false;
    }
    *result = count;
    return true;
}

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool parseTask(QSqlQuery& query, TaskSnapshot* result, QString* error)
{
    TaskSnapshot parsed;
    if (!TaskId::parse(query.value(0).toString(), &parsed.id, error)
        || !RequestId::parse(
            query.value(1).toString(), &parsed.requestId, error)
        || !taskStateFromString(
            query.value(2).toString(), &parsed.state)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("任务记录包含无效状态。");
        }
        return false;
    }
    parsed.capabilityId = query.value(3).toString();
    parsed.taskType = query.value(4).toString();
    parsed.createdAt = parseUtc(query.value(5).toString());
    parsed.updatedAt = parseUtc(query.value(6).toString());
    if (!failureCodeFromString(
            query.value(7).toString(), &parsed.failure.code)) {
        if (error) {
            *error = QStringLiteral(
                "任务记录包含未知 FailureCode。");
        }
        return false;
    }
    parsed.failure.message = query.value(8).toString();
    parsed.failure.suggestedAction = query.value(9).toString();
    parsed.failure.occurredAt = parseUtc(query.value(10).toString());
    *result = parsed;
    return true;
}

} // namespace

ProjectReadRepository::ProjectReadRepository(const ProjectDatabase& database)
    : database_(database)
{
}

bool ProjectReadRepository::summary(ProjectSummarySnapshot* result,
    QString* error) const
{
    if (error) error->clear();
    if (!database_.isOpen() || !result) {
        if (error) {
            *error = QStringLiteral("查询项目汇总需要已打开的数据库和输出对象。");
        }
        return false;
    }

    ProjectSummarySnapshot summary;
    QSqlQuery taskCounts(database_.connection());
    if (!taskCounts.exec(
            QStringLiteral("select state, count(*) from tasks group by state"))) {
        if (error) *error = taskCounts.lastError().text();
        return false;
    }
    while (taskCounts.next()) {
        TaskState state;
        if (!taskStateFromString(taskCounts.value(0).toString(), &state)) {
            if (error) *error = QStringLiteral("项目汇总遇到无效任务状态。");
            return false;
        }
        bool ok = false;
        const qint64 count = taskCounts.value(1).toLongLong(&ok);
        if (!ok || count < 0) {
            if (error) *error = QStringLiteral("项目汇总任务状态计数无效。");
            return false;
        }
        switch (state) {
        case TaskState::Created: summary.tasks.created = count; break;
        case TaskState::Queued: summary.tasks.queued = count; break;
        case TaskState::Starting: summary.tasks.starting = count; break;
        case TaskState::Running: summary.tasks.running = count; break;
        case TaskState::CancelRequested:
            summary.tasks.cancelRequested = count;
            break;
        case TaskState::Succeeded: summary.tasks.succeeded = count; break;
        case TaskState::Failed: summary.tasks.failed = count; break;
        case TaskState::Canceled: summary.tasks.canceled = count; break;
        }
    }

    if (!scalarCount(database_, QStringLiteral("select count(*) from artifacts"),
            &summary.committedArtifactCount, error)
        || !scalarCount(database_, QStringLiteral("select count(*) from datasets"),
            &summary.datasetCount, error)
        || !scalarCount(database_,
            QStringLiteral("select count(*) from dataset_versions"),
            &summary.datasetVersionCount, error)
        || !scalarCount(database_,
            QStringLiteral("select count(*) from dataset_snapshots"),
            &summary.datasetSnapshotCount, error)
        || !scalarCount(database_,
            QStringLiteral("select count(*) from model_packages"),
            &summary.modelPackageCount, error)
        || !scalarCount(database_, QStringLiteral(
                "select count(*) from model_packages where verified = 1"),
            &summary.verifiedModelPackageCount, error)
        || !scalarCount(database_,
            QStringLiteral("select count(*) from workflow_runs"),
            &summary.workflowRunCount, error)
        || !scalarCount(database_, QStringLiteral(
                "select count(*) from workflow_runs "
                "where terminal_policy = 'evidence_required'"),
            &summary.evidenceRequiredWorkflowCount, error)
        || !scalarCount(database_, QStringLiteral(
                "select count(*) from workflow_terminalizations z "
                "join artifacts a on a.id = z.evidence_artifact_id "
                "where a.kind = 'evidence_bundle' "
                "and z.state in ('evidence_attached','closed')"),
            &summary.evidenceAvailableWorkflowCount, error)
        || !scalarCount(database_, QStringLiteral(
                "select count(*) from workflow_terminalizations "
                "where state = 'sealed' and evidence_artifact_id is null"),
            &summary.evidencePendingWorkflowCount, error)) {
        return false;
    }

    *result = summary;
    return true;
}

Page<DeliveryEvidenceCandidate> ProjectReadRepository::deliveryEvidence(
    const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    Page<DeliveryEvidenceCandidate> result;
    PageCursor cursor;
    if (!database_.isOpen()
        || !storage_internal::validatePageRequest(
            request, QStringLiteral("evidence"), &cursor, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral(
                "查询交付证据需要已打开的数据库。");
        }
        return result;
    }

    // 先限制 Artifact 数量再展开文件清单，避免多文件 Artifact 占用多页。
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select t.id, t.request_id, t.state, t.capability_id, t.task_type, "
        "t.created_at, t.updated_at, t.failure_code, t.failure_details, "
        "t.failure_suggested_action, coalesce(t.failure_occurred_at, ''), "
        "a.id, a.task_id, a.kind, a.created_at, "
        "f.relative_path, f.sha256, f.byte_count "
        "from (select id, created_at from artifacts "
        "where kind in ('evidence_bundle', "
            "'external_acceptance_evidence') ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (created_at < :after_time "
            "or (created_at = :after_time and id < :after_id)) ");
    }
    sql += QStringLiteral(
        "order by created_at desc, id desc limit :limit) selected "
        "join artifacts a on a.id = selected.id "
        "join tasks t on t.id = a.task_id "
        "left join artifact_files f on f.artifact_id = a.id "
        "order by selected.created_at desc, a.id desc, "
            "f.relative_path asc");
    query.prepare(sql);
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":after_time"), cursor.timestamp);
        query.bindValue(QStringLiteral(":after_id"), cursor.id);
    }
    query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }

    ArtifactId currentArtifactId;
    while (query.next()) {
        ArtifactId artifactId;
        if (!ArtifactId::parse(
                query.value(11).toString(), &artifactId, error)) {
            return {};
        }
        if (!currentArtifactId.isValid()
            || artifactId != currentArtifactId) {
            DeliveryEvidenceCandidate candidate;
            if (!parseTask(query, &candidate.task, error)
                || !TaskId::parse(query.value(12).toString(),
                    &candidate.artifact.taskId, error)
                || candidate.task.id != candidate.artifact.taskId) {
                if (error && error->isEmpty()) {
                    *error = QStringLiteral(
                        "交付证据候选的任务身份不一致。");
                }
                return {};
            }
            candidate.artifact.id = artifactId;
            candidate.artifact.kind = query.value(13).toString();
            candidate.artifact.createdAt =
                parseUtc(query.value(14).toString());
            if (candidate.artifact.kind.isEmpty()
                || !candidate.artifact.createdAt.isValid()) {
                if (error) {
                    *error = QStringLiteral(
                        "交付证据候选 Artifact 记录无效。");
                }
                return {};
            }
            result.items.append(candidate);
            currentArtifactId = artifactId;
        }
        if (!query.value(15).isNull()) {
            ArtifactFileSnapshot file;
            file.relativePath = query.value(15).toString();
            file.sha256 = query.value(16).toString();
            file.byteCount = query.value(17).toLongLong();
            if (file.relativePath.isEmpty()
                || !isSha256Hex(file.sha256) || file.byteCount < 0) {
                if (error) {
                    *error = QStringLiteral(
                        "交付证据 Artifact 文件记录无效。");
                }
                return {};
            }
            result.items.last().artifact.files.append(file);
        }
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        const ArtifactSnapshot& last =
            result.items.constLast().artifact;
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("evidence"),
            {last.createdAt.toUTC().toString(Qt::ISODateWithMs),
                last.id.toString()});
    }
    return result;
}

} // namespace aitrain
