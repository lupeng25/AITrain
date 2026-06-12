#include "ProjectRepositoryInternal.h"

#include <QSqlError>
#include <QVariant>

namespace aitrain {
namespace repository_internal {

QString sqlError(const QSqlQuery& query)
{
    return query.lastError().text();
}

QString nowIso()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
}

QString dateTimeToIso(const QDateTime& value)
{
    return value.isValid() ? value.toUTC().toString(Qt::ISODateWithMs) : QString();
}

QDateTime dateTimeFromIso(const QString& value)
{
    return value.isEmpty() ? QDateTime() : QDateTime::fromString(value, Qt::ISODateWithMs);
}

namespace {

bool tableHasColumn(QSqlDatabase& db, const QString& tableName, const QString& columnName, QString* error)
{
    QSqlQuery query(db);
    if (!query.exec(QStringLiteral("pragma table_info(%1)").arg(tableName))) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }

    while (query.next()) {
        if (query.value(1).toString() == columnName) {
            return true;
        }
    }
    return false;
}

} // namespace

bool ensureColumn(QSqlDatabase& db, const QString& tableName, const QString& columnDefinition, QString* error)
{
    const QString columnName = columnDefinition.section(QLatin1Char(' '), 0, 0);
    if (tableHasColumn(db, tableName, columnName, error)) {
        return true;
    }

    QSqlQuery query(db);
    if (!query.exec(QStringLiteral("alter table %1 add column %2").arg(tableName, columnDefinition))) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool execStatement(QSqlDatabase& db, const QString& statement, QString* error)
{
    QSqlQuery query(db);
    if (!query.exec(statement)) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ensureSchemaMigrationTable(QSqlDatabase& db, QString* error)
{
    return execStatement(
        db,
        QStringLiteral("create table if not exists schema_migrations ("
                       "version integer primary key,"
                       "name text not null,"
                       "applied_at text not null)"),
        error);
}

bool recordBaselineMigration(QSqlDatabase& db, QString* error)
{
    QSqlQuery query(db);
    query.prepare(QStringLiteral("insert or ignore into schema_migrations(version, name, applied_at) values(?, ?, ?)"));
    query.addBindValue(kCurrentSchemaVersion);
    query.addBindValue(QStringLiteral("baseline_current_schema"));
    query.addBindValue(nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

} // namespace repository_internal
} // namespace aitrain
