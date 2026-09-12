#include "aitrain/storage/ArtifactCatalogRepository.h"

#include "aitrain/domain/ArtifactMemberPath.h"
#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "StoragePagination.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>
#include <QJsonArray>
#include <QJsonDocument>

namespace aitrain {
namespace {

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

} // namespace

ArtifactCatalogRepository::ArtifactCatalogRepository(
    const ProjectDatabase& database)
    : database_(database)
{
}

bool ArtifactCatalogRepository::exists(const ArtifactId& artifactId,
    bool* result, QString* error) const
{
    if (!database_.isOpen() || !artifactId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询 Artifact 存在性需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral("select 1 from artifacts where id = :id"));
    query.bindValue(QStringLiteral(":id"), artifactId.toString());
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return false;
    }
    *result = query.next();
    return true;
}

bool ArtifactCatalogRepository::read(const ArtifactId& artifactId,
    ArtifactSnapshot* result, QString* error) const
{
    if (!database_.isOpen() || !artifactId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询 Artifact 需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery artifactQuery(database_.connection());
    artifactQuery.prepare(QStringLiteral(
        "select id, task_id, kind, created_at from artifacts where id = :id"));
    artifactQuery.bindValue(QStringLiteral(":id"), artifactId.toString());
    if (!artifactQuery.exec() || !artifactQuery.next()) {
        if (error) {
            *error = artifactQuery.lastError().isValid()
                ? artifactQuery.lastError().text()
                : QStringLiteral("Artifact 不存在。");
        }
        return false;
    }
    ArtifactSnapshot parsed;
    if (!ArtifactId::parse(artifactQuery.value(0).toString(), &parsed.id, error)
        || !TaskId::parse(artifactQuery.value(1).toString(), &parsed.taskId, error)) {
        return false;
    }
    parsed.kind = artifactQuery.value(2).toString();
    parsed.createdAt = parseUtc(artifactQuery.value(3).toString());
    if (parsed.kind.trimmed().isEmpty() || !parsed.createdAt.isValid()) {
        if (error) *error = QStringLiteral("Artifact 记录字段无效。");
        return false;
    }

    QSqlQuery filesQuery(database_.connection());
    filesQuery.prepare(QStringLiteral(
        "select relative_path, sha256, byte_count from artifact_files "
        "where artifact_id = :artifact_id order by relative_path asc"));
    filesQuery.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    if (!filesQuery.exec()) {
        if (error) *error = filesQuery.lastError().text();
        return false;
    }
    while (filesQuery.next()) {
        ArtifactFileSnapshot file;
        file.relativePath = filesQuery.value(0).toString();
        file.sha256 = filesQuery.value(1).toString();
        file.byteCount = filesQuery.value(2).toLongLong();
        if (file.relativePath.isEmpty() || !isSha256Hex(file.sha256)
            || file.byteCount < 0) {
            if (error) *error = QStringLiteral("Artifact 文件记录字段无效。");
            return false;
        }
        parsed.files.append(file);
    }
    *result = parsed;
    return true;
}

bool ArtifactCatalogRepository::isDiscardable(const ArtifactId& artifactId,
    bool* result, QString* error) const
{
    if (!database_.isOpen() || !artifactId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "检查 Artifact 引用需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select 1 from model_packages where source_artifact_id = :artifact_id "
        "union all select 1 from evaluation_reports where artifact_id = :artifact_id "
        "union all select 1 from dataset_snapshots where artifact_id = :artifact_id "
        "union all select 1 from workflow_input_bindings where source_artifact_id = :artifact_id "
        "union all select 1 from workflow_steps where input_artifact_id = :artifact_id "
            "or output_artifact_id = :artifact_id "
        "union all select 1 from workflow_terminalizations "
            "where evidence_artifact_id = :artifact_id "
        "union all select 1 from workflow_terminal_outbox "
            "where output_artifact_id = :artifact_id limit 1"));
    query.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return false;
    }
    *result = !query.next();
    return true;
}

int ArtifactCatalogRepository::countForTask(
    const TaskId& taskId, QString* error) const
{
    if (!database_.isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "统计任务 Artifact 需要已打开的数据库和有效任务 ID。");
        }
        return -1;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select count(*) from artifacts where task_id = :task_id"));
    query.bindValue(QStringLiteral(":task_id"), taskId.toString());
    if (!query.exec() || !query.next()) {
        if (error) *error = query.lastError().text();
        return -1;
    }
    return query.value(0).toInt();
}

int ArtifactCatalogRepository::fileCount(
    const ArtifactId& artifactId, QString* error) const
{
    if (!database_.isOpen() || !artifactId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "统计 Artifact 文件需要已打开的数据库和有效 Artifact ID。");
        }
        return -1;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select count(*) from artifact_files where artifact_id = :artifact_id"));
    query.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    if (!query.exec() || !query.next()) {
        if (error) *error = query.lastError().text();
        return -1;
    }
    return query.value(0).toInt();
}

Page<ArtifactFileSnapshot> ArtifactCatalogRepository::files(
    const ArtifactId& artifactId, const PageRequest& request,
    QString* error) const
{
    using storage_internal::PageCursor;
    Page<ArtifactFileSnapshot> result;
    PageCursor cursor;
    if (!database_.isOpen() || !artifactId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "查询 Artifact 文件需要已打开的数据库和有效 Artifact ID。");
        }
        return result;
    }
    if (!storage_internal::validatePageRequest(request,
            QStringLiteral("artifact_files"), &cursor, error)) {
        return result;
    }
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select relative_path, sha256, byte_count from artifact_files "
        "where artifact_id = :artifact_id ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral("and relative_path > :after_path ");
    }
    sql += QStringLiteral("order by relative_path asc limit :limit");
    query.prepare(sql);
    query.bindValue(QStringLiteral(":artifact_id"), artifactId.toString());
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":after_path"), cursor.timestamp);
    }
    query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    while (query.next()) {
        ArtifactFileSnapshot file;
        file.relativePath = query.value(0).toString();
        file.sha256 = query.value(1).toString();
        file.byteCount = query.value(2).toLongLong();
        if (file.relativePath.isEmpty() || !isSha256Hex(file.sha256)
            || file.byteCount < 0) {
            if (error) {
                *error = QStringLiteral("Artifact 文件记录字段无效。");
            }
            return {};
        }
        result.items.append(file);
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        const QString path = result.items.constLast().relativePath;
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("artifact_files"), {path, path});
    }
    return result;
}

Page<ArtifactSnapshot> ArtifactCatalogRepository::forTask(
    const TaskId& taskId, const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    Page<ArtifactSnapshot> result;
    PageCursor cursor;
    if (!database_.isOpen() || !taskId.isValid()) {
        if (error) {
            *error = QStringLiteral(
                "查询任务 Artifact 需要已打开的数据库和有效任务 ID。");
        }
        return result;
    }
    if (!storage_internal::validatePageRequest(request,
            QStringLiteral("task_artifacts"), &cursor, error)) {
        return result;
    }
    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral(
        "select id, created_at from artifacts where task_id = :task_id ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (created_at > :after_time "
            "or (created_at = :after_time and id > :after_id)) ");
    }
    sql += QStringLiteral("order by created_at asc, id asc limit :limit");
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
    QVector<QPair<ArtifactId, QString>> rows;
    while (query.next()) {
        ArtifactId id;
        if (!ArtifactId::parse(query.value(0).toString(), &id, error)) {
            return {};
        }
        rows.append({id, query.value(1).toString()});
    }
    query.finish();
    if (rows.size() > request.pageSize) {
        result.hasMore = true;
        rows.removeLast();
    }
    for (const auto& row : rows) {
        ArtifactSnapshot snapshot;
        if (!read(row.first, &snapshot, error)) return {};
        result.items.append(snapshot);
    }
    if (result.hasMore && !rows.isEmpty()) {
        const auto& last = rows.constLast();
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("task_artifacts"),
            {last.second, last.first.toString()});
    }
    return result;
}

Page<ArtifactSnapshot> ArtifactCatalogRepository::catalog(const QStringList& kinds,
    const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    QStringList normalized = kinds;
    normalized.removeDuplicates();
    normalized.sort();
    const QString type = QStringLiteral("artifact_catalog/") + QString::fromUtf8(
        QJsonDocument(QJsonArray::fromStringList(normalized)).toJson(QJsonDocument::Compact));
    PageCursor cursor;
    if (error) error->clear();
    if (!database_.isOpen()
        || !storage_internal::validatePageRequest(request, type, &cursor, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("查询产物目录需要已打开的项目。");
        return {};
    }
    QString sql = QStringLiteral("select id, task_id, kind, created_at from artifacts where 1 = 1 ");
    QStringList placeholders;
    for (int index = 0; index < normalized.size(); ++index)
        placeholders.append(QStringLiteral(":kind%1").arg(index));
    if (!placeholders.isEmpty()) sql += QStringLiteral("and kind in (%1) ").arg(placeholders.join(QLatin1Char(',')));
    if (!request.after.isEmpty()) sql += QStringLiteral("and (created_at < :time or (created_at = :time and id < :id)) ");
    sql += QStringLiteral("order by created_at desc, id desc limit :limit");
    QSqlQuery query(database_.connection());
    query.prepare(sql);
    for (int index = 0; index < normalized.size(); ++index) query.bindValue(placeholders.at(index), normalized.at(index));
    query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":time"), cursor.timestamp);
        query.bindValue(QStringLiteral(":id"), cursor.id);
    }
    if (!query.exec()) { if (error) *error = query.lastError().text(); return {}; }
    Page<ArtifactSnapshot> result;
    QVector<PageCursor> cursors;
    while (query.next()) {
        ArtifactSnapshot item;
        if (!ArtifactId::parse(query.value(0).toString(), &item.id, error)
            || !TaskId::parse(query.value(1).toString(), &item.taskId, error)) return {};
        item.kind = query.value(2).toString();
        item.createdAt = parseUtc(query.value(3).toString());
        result.items.append(item);
        cursors.append({query.value(3).toString(), item.id.toString()});
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
        cursors.removeLast();
        result.nextCursor = storage_internal::encodePageCursor(type, cursors.constLast());
    }
    return result;
}

} // namespace aitrain
