#pragma once

#include "aitrain/core/TaskModels.h"

#include <QSqlDatabase>
#include <QString>
#include <QVector>

namespace aitrain {

class ProjectRepository {
public:
    ProjectRepository();
    ~ProjectRepository();

    bool open(const QString& databasePath, QString* error = nullptr);
    void close();
    bool isOpen() const;

    bool initialize(QString* error = nullptr);
    bool upsertProject(const QString& name, const QString& rootPath, QString* error = nullptr);
    bool insertTask(const TaskRecord& task, QString* error = nullptr);
    bool updateTaskState(const QString& taskId, TaskState state, const QString& message, QString* error = nullptr);
    bool markInterruptedTasksFailed(const QString& message, QString* error = nullptr);
    bool insertMetric(const MetricPoint& metric, QString* error = nullptr);
    bool insertArtifact(const ArtifactRecord& artifact, QString* error = nullptr);
    bool insertEnvironmentCheck(const EnvironmentCheckRecord& check, QString* error = nullptr);

    QVector<TaskRecord> recentTasks(int limit, QString* error = nullptr) const;
    QVector<EnvironmentCheckRecord> recentEnvironmentChecks(int limit, QString* error = nullptr) const;

private:
    QString connectionName_;
    QSqlDatabase db_;
};

} // namespace aitrain
