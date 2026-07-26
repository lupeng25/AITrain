#pragma once

#include <QSqlDatabase>
#include <QString>

namespace aitrain {

class ProjectDatabase final {
public:
    ProjectDatabase();
    ~ProjectDatabase();
    ProjectDatabase(const ProjectDatabase&) = delete;
    ProjectDatabase& operator=(const ProjectDatabase&) = delete;

    bool open(const QString& databasePath, QString* error = nullptr);
    void close();
    bool isOpen() const;
    void swap(ProjectDatabase& other) noexcept;

    QSqlDatabase& connection();
    const QSqlDatabase& connection() const;

private:
    QString connectionName_;
    QSqlDatabase connection_;
};

} // namespace aitrain
