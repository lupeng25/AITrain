#include "aitrain/storage/ModelCatalogRepository.h"

#include "aitrain/model/ModelManifest.h"
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

bool parseBinding(const QString& value, ModelSourceSnapshotBinding* result)
{
    if (value == QStringLiteral("project_snapshot")) {
        *result = ModelSourceSnapshotBinding::ProjectSnapshot;
        return true;
    }
    if (value == QStringLiteral("external_declared")) {
        *result = ModelSourceSnapshotBinding::ExternalDeclared;
        return true;
    }
    return false;
}

} // namespace

ModelCatalogRepository::ModelCatalogRepository(const ProjectDatabase& database)
    : database_(database)
{
}

bool ModelCatalogRepository::read(const ModelPackageId& modelPackageId,
    ModelPackageSnapshot* result, QString* error) const
{
    if (!database_.isOpen() || !modelPackageId.isValid() || !result) {
        if (error) {
            *error = QStringLiteral(
                "查询模型包需要已打开的数据库、有效 ID 和输出对象。");
        }
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "select source_artifact_id, manifest_json, created_at, "
        "source_snapshot_binding from model_packages where id = :id"));
    query.bindValue(QStringLiteral(":id"), modelPackageId.toString());
    if (!query.exec() || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text() : QStringLiteral("模型包不存在。");
        }
        return false;
    }

    ArtifactId sourceArtifactId;
    if (!ArtifactId::parse(query.value(0).toString(), &sourceArtifactId, error)) {
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        query.value(1).toString().toUtf8(), &parseError);
    ModelManifest manifest;
    if (parseError.error != QJsonParseError::NoError || !document.isObject()
        || !decodeModelManifest(document.object(), &manifest, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("已登记模型包的 Manifest JSON 无效。");
        }
        return false;
    }
    if (manifest.modelPackageId != modelPackageId) {
        if (error) *error = QStringLiteral("已登记模型包的主键与 Manifest 不一致。");
        return false;
    }
    ModelSourceSnapshotBinding binding;
    if (!parseBinding(query.value(3).toString(), &binding)) {
        if (error) {
            *error = QStringLiteral("已登记模型包的 Snapshot 来源绑定无效。");
        }
        return false;
    }
    result->manifest = manifest;
    result->sourceArtifactId = sourceArtifactId;
    result->createdAt = parseUtc(query.value(2).toString());
    result->sourceSnapshotBinding = binding;
    return true;
}

Page<ModelPackageSnapshot> ModelCatalogRepository::page(
    const PageRequest& request, QString* error) const
{
    using storage_internal::PageCursor;
    Page<ModelPackageSnapshot> result;
    PageCursor cursor;
    if (!database_.isOpen()
        || !storage_internal::validatePageRequest(request,
            QStringLiteral("model_packages"), &cursor, error)) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("查询模型包目录需要已打开的数据库。");
        }
        return result;
    }

    QSqlQuery query(database_.connection());
    QString sql = QStringLiteral("select id from model_packages ");
    if (!request.after.isEmpty()) {
        sql += QStringLiteral(
            "where (created_at < :after_time "
            "or (created_at = :after_time and id < :after_id)) ");
    }
    sql += QStringLiteral(
        "order by created_at desc, id desc limit :limit");
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
    while (query.next()) {
        ModelPackageId id;
        if (!ModelPackageId::parse(
                query.value(0).toString(), &id, error)) {
            return {};
        }
        ModelPackageSnapshot item;
        if (!read(id, &item, error)) return {};
        result.items.append(item);
    }
    if (result.items.size() > request.pageSize) {
        result.hasMore = true;
        result.items.removeLast();
    }
    if (result.hasMore && !result.items.isEmpty()) {
        const ModelPackageSnapshot& last = result.items.constLast();
        result.nextCursor = storage_internal::encodePageCursor(
            QStringLiteral("model_packages"),
            {last.createdAt.toUTC().toString(Qt::ISODateWithMs),
                last.manifest.modelPackageId.toString()});
    }
    return result;
}

} // namespace aitrain
