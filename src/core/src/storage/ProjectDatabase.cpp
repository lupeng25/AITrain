#include "aitrain/storage/ProjectDatabase.h"

#include <QSqlError>
#include <QUuid>

#include <utility>

namespace aitrain {

ProjectDatabase::ProjectDatabase()
    : connectionName_(QStringLiteral("aitrain_%1")
          .arg(QUuid::createUuid().toString(QUuid::Id128)))
{
}

ProjectDatabase::~ProjectDatabase()
{
    close();
}

bool ProjectDatabase::open(const QString& databasePath, QString* error)
{
    close();
    connection_ = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName_);
    connection_.setDatabaseName(databasePath);
    if (connection_.open()) return true;
    if (error) *error = connection_.lastError().text();
    return false;
}

void ProjectDatabase::close()
{
    if (!connection_.isValid()) return;
    connection_.close();
    connection_ = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName_);
}

bool ProjectDatabase::isOpen() const
{
    return connection_.isValid() && connection_.isOpen();
}

void ProjectDatabase::swap(ProjectDatabase& other) noexcept
{
    using std::swap;
    swap(connectionName_, other.connectionName_);
    swap(connection_, other.connection_);
}

QSqlDatabase& ProjectDatabase::connection() { return connection_; }
const QSqlDatabase& ProjectDatabase::connection() const { return connection_; }

} // namespace aitrain
