#include "aitrain/storage/ProjectMetaRepository.h"

#include "aitrain/storage/ProjectDatabase.h"
#include "aitrain/storage/ProjectStore.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

namespace aitrain {
namespace {

constexpr int kSchemaVersion = 13;

QDateTime parseUtc(const QString& value)
{
    return QDateTime::fromString(value, Qt::ISODateWithMs).toUTC();
}

QString utcText(const QDateTime& value)
{
    return value.toUTC().toString(Qt::ISODateWithMs);
}

} // namespace

ProjectMetaRepository::ProjectMetaRepository(const ProjectDatabase& database)
    : database_(database)
{
}

bool ProjectMetaRepository::read(ProjectMetaSnapshot* result, QString* error) const
{
    if (!database_.isOpen() || !result) {
        if (error) {
            *error = QStringLiteral("读取 project_meta 需要已打开的数据库和输出对象。");
        }
        return false;
    }

    QSqlQuery query(database_.connection());
    if (!query.exec(QStringLiteral(
            "select project_id, schema_version, display_name, open_generation, "
            "created_at, updated_at, last_opened_at "
            "from project_meta where singleton = 1"))
        || !query.next()) {
        if (error) {
            *error = query.lastError().isValid()
                ? query.lastError().text()
                : QStringLiteral("数据库缺少 project_meta 记录。");
        }
        return false;
    }

    ProjectMetaSnapshot parsed;
    if (!ProjectId::parse(query.value(0).toString(), &parsed.projectId, error)) {
        return false;
    }
    parsed.schemaVersion = query.value(1).toInt();
    parsed.displayName = query.value(2).toString();
    parsed.openGeneration = query.value(3).toLongLong();
    parsed.createdAt = parseUtc(query.value(4).toString());
    parsed.updatedAt = parseUtc(query.value(5).toString());
    parsed.lastOpenedAt = parseUtc(query.value(6).toString());
    if (parsed.schemaVersion != kSchemaVersion) {
        if (error) {
            *error = QStringLiteral(
                "SchemaRebuildRequired：项目数据库不是 Schema 13，请显式重建项目。");
        }
        return false;
    }
    if (parsed.displayName.trimmed().isEmpty()
        || !parsed.createdAt.isValid() || !parsed.updatedAt.isValid()
        || parsed.openGeneration < 0 || query.next()) {
        if (error) *error = QStringLiteral("project_meta 记录损坏或不唯一。");
        return false;
    }
    *result = parsed;
    return true;
}

bool ProjectMetaRepository::incrementOpenGeneration(const QDateTime& now,
    QString* error)
{
    if (!database_.isOpen()) {
        if (error) *error = QStringLiteral("更新 open_generation 需要已打开的数据库。");
        return false;
    }
    QSqlQuery query(database_.connection());
    query.prepare(QStringLiteral(
        "update project_meta set open_generation = open_generation + 1, "
        "last_opened_at = :now, updated_at = :now where singleton = 1"));
    query.bindValue(QStringLiteral(":now"), utcText(now));
    if (query.exec() && query.numRowsAffected() == 1) return true;
    if (error) {
        *error = query.lastError().isValid()
            ? query.lastError().text()
            : QStringLiteral("project_meta 单行更新没有命中记录。");
    }
    return false;
}

} // namespace aitrain
