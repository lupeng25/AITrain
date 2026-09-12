#include "aitrain/storage/TaskEventRepository.h"

#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "StoragePagination.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

#include <cmath>

namespace aitrain {
namespace {

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool parseTask(QSqlQuery& query, TaskSnapshot* result, QString* error)
{
    TaskSnapshot parsed;
    if (!TaskId::parse(query.value(0).toString(), &parsed.id, error)
        || !RequestId::parse(query.value(1).toString(), &parsed.requestId, error)
        || !taskStateFromString(query.value(2).toString(), &parsed.state)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("任务记录包含无效状态。");
        }
        return false;
    }
    parsed.capabilityId = query.value(3).toString();
    parsed.taskType = query.value(4).toString();
    parsed.createdAt = parseUtc(query.value(5).toString());
    parsed.updatedAt = parseUtc(query.value(6).toString());
    if (!failureCodeFromString(query.value(7).toString(), &parsed.failure.code)) {
        if (error) *error = QStringLiteral("任务记录包含未知 FailureCode。");
        return false;
    }
    parsed.failure.message = query.value(8).toString();
    parsed.failure.suggestedAction = query.value(9).toString();
    parsed.failure.occurredAt = parseUtc(query.value(10).toString());
    *result = parsed;
    return true;
}

} // namespace

TaskEventRepository::TaskEventRepository(const ProjectDatabase& database)
    : database_(database)
{
}

bool TaskEventRepository::exists(const TaskId& taskId, bool* result,
    QString* error) const
{
    if (!database_.isOpen() || !taskId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral("查询任务存在性需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral("select 1 from tasks where id = :id"));
    query.bindValue(QStringLiteral(":id"), taskId.toString());
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return false;
    }
    *result = query.next();
    return true;
}

bool TaskEventRepository::read(const TaskId& taskId, TaskSnapshot* result,
    QString* error) const
{
    if (!database_.isOpen() || !taskId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral("查询任务需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select id, request_id, state, capability_id, task_type, created_at, "
        "updated_at, failure_code, failure_details, failure_suggested_action, "
        "coalesce(failure_occurred_at, '') from tasks where id = :id"));
    query.bindValue(QStringLiteral(":id"), taskId.toString());
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text() : QStringLiteral("任务不存在。");
        }
        return false;
    }
    return parseTask(query, result, error);
}

int TaskEventRepository::eventCount(
    const TaskId& taskId, QString* error) const
{
    return countForTask(QStringLiteral("task_events"), taskId, error);
}

int TaskEventRepository::metricCount(
    const TaskId& taskId, QString* error) const
{
    return countForTask(QStringLiteral("task_metrics"), taskId, error);
}

int TaskEventRepository::countForTask(const QString& table,
    const TaskId& taskId, QString* error) const
{
    if (!database_.isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "统计任务事实需要已打开的数据库和有效任务 ID。");
        }
        return -1;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral("select count(*) from %1 where task_id = :task_id")
        .arg(table));
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    if (!query.exec() || !query.next()) {
        if (error) *error = query.lastError().text();
        return -1;
    }
    return query.value(0).toInt();
}

Page<TaskSnapshot> TaskEventRepository::page(
    const PageRequest& request, QString* error, const CatalogFilter& filter) const
{
    using storage_internal::PageCursor;
    Page<TaskSnapshot> result;
    PageCursor cursor;
    const QString queryType = storage_internal::catalogQueryType(QStringLiteral("tasks"), filter);
    if (!database_.isOpen()
        || !storage_internal::validatePageRequest(
            request, queryType, &cursor, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("查询任务目录需要已打开的数据库。");
        }
        return result;
    }
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select id, request_id, state, capability_id, task_type, created_at, "
        "updated_at, failure_code, failure_details, "
        "failure_suggested_action, coalesce(failure_occurred_at, '') "
        "from tasks where 1=1 ");
    sql += storage_internal::catalogKindClause(filter, QStringLiteral("task_type"));
    if (!filter.state.isEmpty()) sql += QStringLiteral("and state = :state ");
    if (!filter.text.trimmed().isEmpty()) sql += QStringLiteral(
        "and (instr(lower(task_type || ' ' || capability_id || ' ' || id || ' ' || updated_at || ' ' || failure_details), :search) > 0 "
        "or exists(select 1 from workflow_runs search_run join workflow_steps search_step on search_step.workflow_run_id = search_run.id "
        "where search_run.task_id = tasks.id and (instr(lower(search_step.backend), :search) > 0 or instr(lower(search_step.parameter_summary_json), :search) > 0))) ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (updated_at < :after_time "
            "or (updated_at = :after_time and id < :after_id)) ");
    }
    sql += QStringLiteral(
        "order by updated_at desc, id desc limit :limit");
    query.prepare(sql);
    storage_internal::bindCatalogFilter(query, filter);
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
        TaskSnapshot snapshot;
        if (!parseTask(query, &snapshot, error)) return {};
        result.items.append(snapshot);
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        const TaskSnapshot& last = result.items.constLast();
        result.nextCursor = storage_internal::encodePageCursor(
            queryType,
            {last.updatedAt.toUTC().toString(Qt::ISODateWithMs),
                last.id.toString()});
    }
    return result;
}

Page<MetricSnapshot> TaskEventRepository::metrics(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    Page<MetricSnapshot> result;
    PageCursor cursor;
    if (!database_.isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "查询任务指标需要已打开的数据库和有效任务 ID。");
        }
        return result;
    }
    if (!storage_internal::validatePageRequest(request,
            QStringLiteral("task_metrics"), &cursor, error)) {
        return result;
    }
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select id, name, value, occurred_at from task_metrics "
        "where task_id = :task_id ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (occurred_at > :after_time "
            "or (occurred_at = :after_time and id > :after_id)) ");
    }
    sql += QStringLiteral(
        "order by occurred_at asc, id asc limit :limit");
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
    QVector<PageCursor> rowCursors;
    while (query.next()) {
        MetricSnapshot metric;
        metric.name = query.value(1).toString();
        metric.value = query.value(2).toDouble();
        metric.occurredAt = parseUtc(query.value(3).toString());
        if (metric.name.trimmed().isEmpty()
            || !metric.occurredAt.isValid()
            || !std::isfinite(metric.value)) {
            if (error) {
                *error = QStringLiteral("任务指标记录字段无效。");
            }
            return {};
        }
        result.items.append(metric);
        rowCursors.append(
            {query.value(3).toString(), query.value(0).toString()});
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
        rowCursors.removeLast();
    }
    if (result.hasMore && !rowCursors.isEmpty()) {
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("task_metrics"), rowCursors.constLast());
    }
    return result;
}

} // namespace aitrain
