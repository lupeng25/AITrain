#pragma once

#include <QDateTime>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QString>

namespace aitrain {
namespace repository_internal {

constexpr int kCurrentSchemaVersion = 1;

QString sqlError(const QSqlQuery& query);
QString nowIso();
QString dateTimeToIso(const QDateTime& value);
QDateTime dateTimeFromIso(const QString& value);
bool ensureColumn(QSqlDatabase& db, const QString& tableName, const QString& columnDefinition, QString* error);
bool execStatement(QSqlDatabase& db, const QString& statement, QString* error);
bool ensureSchemaMigrationTable(QSqlDatabase& db, QString* error);
bool recordBaselineMigration(QSqlDatabase& db, QString* error);

} // namespace repository_internal
} // namespace aitrain
