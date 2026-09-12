#include "aitrain/storage/DatasetCatalogRepository.h"

#include "aitrain/domain/ArtifactMemberPath.h"
#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"
#include "StoragePagination.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVariant>

namespace aitrain {
namespace {

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

bool parseSnapshot(QSqlQuery& query, DatasetSnapshotRecord* result,
    QString* error)
{
    DatasetSnapshotRecord parsed;
    if (!DatasetId::parse(query.value(0).toString(), &parsed.datasetId, error)
        || !DatasetVersionId::parse(query.value(1).toString(),
            &parsed.datasetVersionId, error)
        || !SnapshotId::parse(query.value(2).toString(), &parsed.id, error)
        || !TaskId::parse(query.value(3).toString(), &parsed.taskId, error)
        || !ArtifactId::parse(query.value(4).toString(), &parsed.artifactId, error)) {
        return false;
    }
    parsed.rootPath = query.value(5).toString();
    parsed.datasetFormat = query.value(6).toString();
    parsed.driverId = query.value(7).toString();
    parsed.driverVersion = query.value(8).toString();
    parsed.rootHash = query.value(9).toString();
    parsed.manifestSha256 = query.value(10).toString();
    parsed.fileCount = query.value(11).toLongLong();
    parsed.totalBytes = query.value(12).toLongLong();
    parsed.createdAt = parseUtc(query.value(13).toString());
    if (parsed.rootPath.isEmpty() || parsed.datasetFormat.isEmpty()
        || parsed.driverId.isEmpty() || parsed.driverVersion.isEmpty()
        || !isSha256Hex(parsed.rootHash)
        || !isSha256Hex(parsed.manifestSha256)
        || parsed.fileCount < 0 || parsed.totalBytes < 0
        || !parsed.createdAt.isValid()) {
        if (error) *error = QStringLiteral("数据集快照记录字段无效。");
        return false;
    }
    *result = parsed;
    return true;
}

} // namespace

DatasetCatalogRepository::DatasetCatalogRepository(
    const ProjectDatabase& database)
    : database_(database)
{
}

bool DatasetCatalogRepository::snapshot(const SnapshotId& snapshotId,
    DatasetSnapshotRecord* result, QString* error) const
{
    if (!snapshotId.isValid()) {
        if (error) *error = QStringLiteral("查询数据集快照需要有效 ID。");
        return false;
    }
    return executeSnapshotQuery(QStringLiteral("dataset_snapshots.id"),
        snapshotId.toString(), QStringLiteral("数据集快照不存在。"), result, error);
}

bool DatasetCatalogRepository::snapshotForArtifact(
    const ArtifactId& artifactId, DatasetSnapshotRecord* result,
    QString* error) const
{
    if (!artifactId.isValid()) {
        if (error) *error = QStringLiteral("按 Artifact 查询数据集快照需要有效 ID。");
        return false;
    }
    return executeSnapshotQuery(QStringLiteral("dataset_snapshots.artifact_id"),
        artifactId.toString(), QStringLiteral("Artifact 未关联数据集快照。"),
        result, error);
}

Page<DatasetCatalogItem> DatasetCatalogRepository::page(
    const PageRequest& request, QString* error, const CatalogFilter& filter) const
{
    using storage_internal::PageCursor;
    Page<DatasetCatalogItem> result;
    PageCursor cursor;
    const QString queryType = storage_internal::catalogQueryType(QStringLiteral("datasets"), filter);
    if (!database_.isOpen()
        || !storage_internal::validatePageRequest(
            request, queryType, &cursor, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("查询数据集目录需要已打开的数据库。");
        }
        return result;
    }

    QSqlQuery query(database_.connection());
    QVector<PageCursor> rowCursors;
    QString sql = QStringLiteral(
        "with catalog as (select d.id, d.dataset_format, "
        "(select count(*) from dataset_versions v "
            "where v.dataset_id = d.id), "
        "(select count(*) from dataset_snapshots s "
            "join dataset_versions version_count "
                "on version_count.id = s.dataset_version_id "
            "where version_count.dataset_id = d.id), "
        "s.dataset_version_id, s.id snapshot_id, s.artifact_id, "
        "v.root_hash, s.file_count, s.created_at, "
        "coalesce(s.created_at, d.created_at) sort_time, s.task_id source_task_id, "
        "(select ws.parameter_summary_json from dataset_snapshots ds "
            "join dataset_versions dv on dv.id = ds.dataset_version_id "
            "join workflow_runs wr on wr.task_id = ds.task_id "
            "join workflow_steps ws on ws.workflow_run_id = wr.id and ws.ordinal = 0 "
            "where dv.dataset_id = d.id and wr.template_id in "
                "('dataset_snapshot_import', 'dataset_conversion', 'dataset_split') "
            "order by ds.created_at asc, ds.id asc limit 1) name_parameters, "
        "(select wr.task_id from workflow_input_bindings ib "
            "join workflow_runs wr on wr.id = ib.workflow_run_id "
            "where ib.dataset_snapshot_id = s.id and wr.template_id = 'dataset_quality' "
            "order by wr.created_at desc, wr.id desc limit 1) quality_task_id "
        "from datasets d "
        "left join dataset_snapshots s on s.id = ("
            "select s2.id from dataset_snapshots s2 "
            "join dataset_versions latest_version "
                "on latest_version.id = s2.dataset_version_id "
            "where latest_version.dataset_id = d.id "
            "order by s2.created_at desc, s2.id desc limit 1) "
        "left join dataset_versions v on v.id = s.dataset_version_id) "
        "select * from catalog where 1=1 ");
    sql += storage_internal::catalogKindClause(filter, QStringLiteral("dataset_format"));
    if (!filter.text.trimmed().isEmpty()) sql += QStringLiteral("and (instr(lower(dataset_format), :search) > 0 or instr(lower(coalesce(name_parameters, '')), :json_search) > 0) ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "and (sort_time < :after_time "
            "or (sort_time = :after_time and id < :after_id)) ");
    }
    sql += QStringLiteral("order by sort_time desc, id desc ");
    // SQLite 旧版本不要求 JSON1；候选按序流式解码，精确按名称过滤到一页。
    if (filter.text.trimmed().isEmpty()) sql += QStringLiteral("limit :limit");
    query.setForwardOnly(true);
    query.prepare(sql);
    storage_internal::bindCatalogFilter(query, filter);
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":after_time"), cursor.timestamp);
        query.bindValue(QStringLiteral(":after_id"), cursor.id);
    }
    if (filter.text.trimmed().isEmpty()) query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    else {
        const QByteArray quoted = QJsonDocument(QJsonArray{filter.text.trimmed().toLower()}).toJson(QJsonDocument::Compact);
        query.bindValue(QStringLiteral(":json_search"), QString::fromUtf8(quoted.mid(2, quoted.size() - 4)));
    }
    if (!query.exec()) {
        if (error) *error = query.lastError().text();
        return {};
    }
    while (query.next()) {
        DatasetCatalogItem item;
        if (!DatasetId::parse(
                query.value(0).toString(), &item.datasetId, error)) {
            return {};
        }
        item.datasetFormat = query.value(1).toString();
        item.displayName = QJsonDocument::fromJson(query.value(12).toByteArray())
            .object().value(QStringLiteral("targetDatasetName")).toString().trimmed();
        if (!filter.text.trimmed().isEmpty() && !item.displayName.contains(filter.text.trimmed(), Qt::CaseInsensitive)
            && !item.datasetFormat.contains(filter.text.trimmed(), Qt::CaseInsensitive)) continue;
        item.versionCount = query.value(2).toLongLong();
        item.snapshotCount = query.value(3).toLongLong();
        if (!query.value(4).isNull()) {
            if (!DatasetVersionId::parse(query.value(4).toString(),
                    &item.latestVersionId, error)
                || !SnapshotId::parse(query.value(5).toString(),
                    &item.latestSnapshotId, error)
                || !ArtifactId::parse(query.value(6).toString(),
                    &item.latestArtifactId, error)) {
                return {};
            }
            item.latestRootHash = query.value(7).toString();
            item.latestFileCount = query.value(8).toLongLong();
            item.latestCreatedAt = parseUtc(query.value(9).toString());
            if (!query.value(11).isNull()
                && !TaskId::parse(query.value(11).toString(), &item.latestSourceTaskId, error)) return {};
            if (!query.value(13).isNull()
                && !TaskId::parse(query.value(13).toString(), &item.latestQualityTaskId, error)) return {};
        }
        result.items.append(item);
        rowCursors.append(
            {query.value(10).toString(), item.datasetId.toString()});
        if (result.items.size() > request.pageSize) break;
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
        rowCursors.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        result.nextCursor = storage_internal::encodePageCursor(
            queryType, rowCursors.constLast());
    }
    return result;
}

Page<DatasetSnapshotRecord> DatasetCatalogRepository::versions(
    const DatasetId& datasetId, const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    const QString type = QStringLiteral("dataset_versions/") + datasetId.toString();
    PageCursor cursor;
    if (error) error->clear();
    if (!database_.isOpen() || !datasetId.isValid()
        || !storage_internal::validatePageRequest(request, type, &cursor, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("查询版本需要已打开的项目和有效数据集。");
        return {};
    }
    QString sql = QStringLiteral(
        "select d.id, v.id, s.id, s.task_id, s.artifact_id, s.root_path, "
        "d.dataset_format, s.driver_id, s.driver_version, v.root_hash, "
        "s.manifest_sha256, s.file_count, s.total_bytes, s.created_at, "
        "(select wr.task_id from workflow_input_bindings ib "
            "join workflow_runs wr on wr.id = ib.workflow_run_id "
            "where ib.dataset_snapshot_id = s.id and wr.template_id = 'dataset_quality' "
            "order by wr.created_at desc, wr.id desc limit 1) "
        "from dataset_snapshots s join dataset_versions v on v.id = s.dataset_version_id "
        "join datasets d on d.id = v.dataset_id where d.id = :dataset ");
    if (!request.after.isEmpty()) sql += QStringLiteral(
        "and (s.created_at < :time or (s.created_at = :time and s.id < :id)) ");
    sql += QStringLiteral("order by s.created_at desc, s.id desc limit :limit");
    QSqlQuery query(database_.connection());
    query.prepare(sql);
    query.bindValue(QStringLiteral(":dataset"), datasetId.toString());
    query.bindValue(QStringLiteral(":limit"), request.pageSize + 1);
    if (!request.after.isEmpty()) {
        query.bindValue(QStringLiteral(":time"), cursor.timestamp);
        query.bindValue(QStringLiteral(":id"), cursor.id);
    }
    if (!query.exec()) { if (error) *error = query.lastError().text(); return {}; }
    Page<DatasetSnapshotRecord> result;
    QVector<PageCursor> cursors;
    while (query.next()) {
        DatasetSnapshotRecord item;
        if (!parseSnapshot(query, &item, error)) return {};
        if (!query.value(14).isNull()
            && !TaskId::parse(query.value(14).toString(), &item.latestQualityTaskId, error)) return {};
        result.items.append(item);
        cursors.append({query.value(13).toString(), item.id.toString()});
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
        cursors.removeLast();
        result.nextCursor = storage_internal::encodePageCursor(type, cursors.constLast());
    }
    return result;
}

bool DatasetCatalogRepository::executeSnapshotQuery(const QString& whereColumn,
    const QString& id, const QString& notFoundMessage,
    DatasetSnapshotRecord* result, QString* error) const
{
    if (!database_.isOpen() || !result) {
        if (error) {
            *error = QStringLiteral("查询数据集快照需要已打开的数据库和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select datasets.id, dataset_versions.id, dataset_snapshots.id, "
        "dataset_snapshots.task_id, dataset_snapshots.artifact_id, "
        "dataset_snapshots.root_path, datasets.dataset_format, "
        "dataset_snapshots.driver_id, dataset_snapshots.driver_version, "
        "dataset_versions.root_hash, dataset_snapshots.manifest_sha256, "
        "dataset_snapshots.file_count, dataset_snapshots.total_bytes, "
        "dataset_snapshots.created_at from dataset_snapshots "
        "join dataset_versions on dataset_versions.id = "
            "dataset_snapshots.dataset_version_id "
        "join datasets on datasets.id = dataset_versions.dataset_id where ")
        + whereColumn + QStringLiteral(" = :id"));
    query.bindValue(QStringLiteral(":id"), id);
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text() : notFoundMessage;
        }
        return false;
    }
    return parseSnapshot(query, result, error);
}

} // namespace aitrain
