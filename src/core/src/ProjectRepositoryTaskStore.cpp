#include "aitrain/core/ProjectRepository.h"

#include "ProjectRepositoryInternal.h"

#include <QSqlQuery>
#include <QVariant>

namespace aitrain {

using namespace repository_internal;

bool ProjectRepository::upsertProject(const QString& name, const QString& rootPath, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into projects(name, root_path, created_at, updated_at) "
                                 "values(?, ?, ?, ?) "
                                 "on conflict(root_path) do update set name = excluded.name, updated_at = excluded.updated_at"));
    const QString timestamp = nowIso();
    query.addBindValue(name);
    query.addBindValue(rootPath);
    query.addBindValue(timestamp);
    query.addBindValue(timestamp);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::insertTask(const TaskRecord& task, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into tasks(id, project_name, plugin_id, task_type, kind, state, work_dir, message, created_at, updated_at, started_at, finished_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
    const QString created = task.createdAt.isValid() ? task.createdAt.toUTC().toString(Qt::ISODateWithMs) : nowIso();
    const QString updated = task.updatedAt.isValid() ? task.updatedAt.toUTC().toString(Qt::ISODateWithMs) : created;
    query.addBindValue(task.id);
    query.addBindValue(task.projectName);
    query.addBindValue(task.pluginId);
    query.addBindValue(task.taskType);
    query.addBindValue(taskKindToString(task.kind));
    query.addBindValue(taskStateToString(task.state));
    query.addBindValue(task.workDir);
    query.addBindValue(task.message);
    query.addBindValue(created);
    query.addBindValue(updated);
    query.addBindValue(dateTimeToIso(task.startedAt));
    query.addBindValue(dateTimeToIso(task.finishedAt));
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::updateTaskState(const QString& taskId, TaskState state, const QString& message, QString* error)
{
    QSqlQuery readQuery(db_);
    readQuery.prepare(QStringLiteral("select state, started_at from tasks where id = ?"));
    readQuery.addBindValue(taskId);
    if (!readQuery.exec()) {
        if (error) {
            *error = sqlError(readQuery);
        }
        return false;
    }
    if (!readQuery.next()) {
        if (error) {
            *error = QStringLiteral("Task not found: %1").arg(taskId);
        }
        return false;
    }

    const TaskState currentState = taskStateFromString(readQuery.value(0).toString());
    if (!isValidTaskStateTransition(currentState, state)) {
        if (error) {
            *error = QStringLiteral("Invalid task state transition: %1 -> %2")
                .arg(taskStateToString(currentState), taskStateToString(state));
        }
        return false;
    }

    const QString timestamp = nowIso();
    const QString currentStartedAt = readQuery.value(1).toString();
    QString startedAt = currentStartedAt;
    QString finishedAt;
    if (state == TaskState::Running && currentStartedAt.isEmpty()) {
        startedAt = timestamp;
    }
    if (isTerminalTaskState(state)) {
        finishedAt = timestamp;
    }

    QSqlQuery query(db_);
    query.prepare(QStringLiteral("update tasks set state = ?, message = ?, updated_at = ?, started_at = ?, finished_at = coalesce(nullif(?, ''), finished_at) where id = ?"));
    query.addBindValue(taskStateToString(state));
    query.addBindValue(message);
    query.addBindValue(timestamp);
    query.addBindValue(startedAt);
    query.addBindValue(finishedAt);
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::insertMetric(const MetricPoint& metric, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into metrics(task_id, name, value, step, epoch, created_at) values(?, ?, ?, ?, ?, ?)"));
    query.addBindValue(metric.taskId);
    query.addBindValue(metric.name);
    query.addBindValue(metric.value);
    query.addBindValue(metric.step);
    query.addBindValue(metric.epoch);
    query.addBindValue(metric.createdAt.isValid() ? metric.createdAt.toUTC().toString(Qt::ISODateWithMs) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::markInterruptedTasksFailed(const QString& message, QString* error)
{
    QSqlQuery query(db_);
    const QString timestamp = nowIso();
    query.prepare(QStringLiteral("update tasks set state = ?, message = ?, updated_at = ?, finished_at = ? "
                                 "where state in (?, ?)"));
    query.addBindValue(taskStateToString(TaskState::Failed));
    query.addBindValue(message);
    query.addBindValue(timestamp);
    query.addBindValue(timestamp);
    query.addBindValue(taskStateToString(TaskState::Running));
    query.addBindValue(taskStateToString(TaskState::Paused));
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::insertArtifact(const ArtifactRecord& artifact, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into artifacts(task_id, kind, path, message, created_at) values(?, ?, ?, ?, ?)"));
    query.addBindValue(artifact.taskId);
    query.addBindValue(artifact.kind);
    query.addBindValue(artifact.path);
    query.addBindValue(artifact.message);
    query.addBindValue(artifact.createdAt.isValid() ? artifact.createdAt.toUTC().toString(Qt::ISODateWithMs) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::insertExport(const ExportRecord& exportRecord, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into exports(task_id, source_checkpoint_path, format, path, config_json, input_shape_json, output_shape_json, created_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?, ?)"));
    query.addBindValue(exportRecord.taskId);
    query.addBindValue(exportRecord.sourceCheckpointPath);
    query.addBindValue(exportRecord.format);
    query.addBindValue(exportRecord.path);
    query.addBindValue(exportRecord.configJson);
    query.addBindValue(exportRecord.inputShapeJson);
    query.addBindValue(exportRecord.outputShapeJson);
    query.addBindValue(exportRecord.createdAt.isValid() ? exportRecord.createdAt.toUTC().toString(Qt::ISODateWithMs) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

bool ProjectRepository::insertEnvironmentCheck(const EnvironmentCheckRecord& check, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into environment_checks(name, status, message, details_json, checked_at) values(?, ?, ?, ?, ?)"));
    query.addBindValue(check.name);
    query.addBindValue(check.status);
    query.addBindValue(check.message);
    query.addBindValue(check.detailsJson);
    query.addBindValue(check.checkedAt.isValid() ? check.checkedAt.toUTC().toString(Qt::ISODateWithMs) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

QVector<TaskRecord> ProjectRepository::recentTasks(int limit, QString* error) const
{
    QVector<TaskRecord> tasks;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, project_name, plugin_id, task_type, kind, state, work_dir, message, created_at, updated_at, started_at, finished_at "
                                 "from tasks order by updated_at desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return tasks;
    }

    while (query.next()) {
        TaskRecord task;
        task.id = query.value(0).toString();
        task.projectName = query.value(1).toString();
        task.pluginId = query.value(2).toString();
        task.taskType = query.value(3).toString();
        task.kind = taskKindFromString(query.value(4).toString());
        task.state = taskStateFromString(query.value(5).toString());
        task.workDir = query.value(6).toString();
        task.message = query.value(7).toString();
        task.createdAt = dateTimeFromIso(query.value(8).toString());
        task.updatedAt = dateTimeFromIso(query.value(9).toString());
        task.startedAt = dateTimeFromIso(query.value(10).toString());
        task.finishedAt = dateTimeFromIso(query.value(11).toString());
        tasks.append(task);
    }
    return tasks;
}

QVector<MetricPoint> ProjectRepository::metricsForTask(const QString& taskId, QString* error) const
{
    QVector<MetricPoint> metrics;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, task_id, name, value, step, epoch, created_at "
                                 "from metrics where task_id = ? order by step asc, id asc"));
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return metrics;
    }

    while (query.next()) {
        MetricPoint point;
        point.id = query.value(0).toInt();
        point.taskId = query.value(1).toString();
        point.name = query.value(2).toString();
        point.value = query.value(3).toDouble();
        point.step = query.value(4).toInt();
        point.epoch = query.value(5).toInt();
        point.createdAt = dateTimeFromIso(query.value(6).toString());
        metrics.append(point);
    }
    return metrics;
}

QVector<ArtifactRecord> ProjectRepository::artifactsForTask(const QString& taskId, QString* error) const
{
    QVector<ArtifactRecord> artifacts;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, task_id, kind, path, message, created_at "
                                 "from artifacts where task_id = ? order by created_at asc, id asc"));
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return artifacts;
    }

    while (query.next()) {
        ArtifactRecord artifact;
        artifact.id = query.value(0).toInt();
        artifact.taskId = query.value(1).toString();
        artifact.kind = query.value(2).toString();
        artifact.path = query.value(3).toString();
        artifact.message = query.value(4).toString();
        artifact.createdAt = dateTimeFromIso(query.value(5).toString());
        artifacts.append(artifact);
    }
    return artifacts;
}

QVector<ExportRecord> ProjectRepository::exportsForTask(const QString& taskId, QString* error) const
{
    QVector<ExportRecord> exports;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, task_id, source_checkpoint_path, format, path, config_json, input_shape_json, output_shape_json, created_at "
                                 "from exports where task_id = ? order by created_at asc, id asc"));
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return exports;
    }

    while (query.next()) {
        ExportRecord record;
        record.id = query.value(0).toInt();
        record.taskId = query.value(1).toString();
        record.sourceCheckpointPath = query.value(2).toString();
        record.format = query.value(3).toString();
        record.path = query.value(4).toString();
        record.configJson = query.value(5).toString();
        record.inputShapeJson = query.value(6).toString();
        record.outputShapeJson = query.value(7).toString();
        record.createdAt = dateTimeFromIso(query.value(8).toString());
        exports.append(record);
    }
    return exports;
}

QVector<ExportRecord> ProjectRepository::recentExports(int limit, QString* error) const
{
    QVector<ExportRecord> exports;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, task_id, source_checkpoint_path, format, path, config_json, input_shape_json, output_shape_json, created_at "
                                 "from exports order by created_at desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return exports;
    }

    while (query.next()) {
        ExportRecord record;
        record.id = query.value(0).toInt();
        record.taskId = query.value(1).toString();
        record.sourceCheckpointPath = query.value(2).toString();
        record.format = query.value(3).toString();
        record.path = query.value(4).toString();
        record.configJson = query.value(5).toString();
        record.inputShapeJson = query.value(6).toString();
        record.outputShapeJson = query.value(7).toString();
        record.createdAt = dateTimeFromIso(query.value(8).toString());
        exports.append(record);
    }
    return exports;
}

QVector<EnvironmentCheckRecord> ProjectRepository::recentEnvironmentChecks(int limit, QString* error) const
{
    QVector<EnvironmentCheckRecord> checks;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select name, status, message, details_json, checked_at "
                                 "from environment_checks order by checked_at desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return checks;
    }

    while (query.next()) {
        EnvironmentCheckRecord check;
        check.name = query.value(0).toString();
        check.status = query.value(1).toString();
        check.message = query.value(2).toString();
        check.detailsJson = query.value(3).toString();
        check.checkedAt = dateTimeFromIso(query.value(4).toString());
        checks.append(check);
    }
    return checks;
}

} // namespace aitrain
