#include "aitrain/core/ProjectRepository.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QUuid>
#include <QVariant>

namespace aitrain {

namespace {

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

} // namespace

ProjectRepository::ProjectRepository()
    : connectionName_(QStringLiteral("aitrain_%1").arg(QUuid::createUuid().toString(QUuid::Id128)))
{
}

ProjectRepository::~ProjectRepository()
{
    close();
}

bool ProjectRepository::open(const QString& databasePath, QString* error)
{
    close();
    db_ = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName_);
    db_.setDatabaseName(databasePath);
    if (!db_.open()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }
    return initialize(error);
}

void ProjectRepository::close()
{
    if (db_.isValid()) {
        db_.close();
    }
    db_ = QSqlDatabase();
    if (QSqlDatabase::contains(connectionName_)) {
        QSqlDatabase::removeDatabase(connectionName_);
    }
}

bool ProjectRepository::isOpen() const
{
    return db_.isValid() && db_.isOpen();
}

bool ProjectRepository::initialize(QString* error)
{
    QSqlQuery query(db_);
    const QStringList statements = {
        QStringLiteral("create table if not exists projects ("
                       "id integer primary key autoincrement,"
                       "name text not null,"
                       "root_path text not null unique,"
                       "created_at text not null,"
                       "updated_at text not null)"),
        QStringLiteral("create table if not exists tasks ("
                       "id text primary key,"
                       "project_name text not null,"
                       "plugin_id text not null,"
                       "task_type text not null,"
                       "kind text not null,"
                       "state text not null,"
                       "work_dir text not null,"
                       "message text,"
                       "created_at text not null,"
                       "updated_at text not null,"
                       "started_at text,"
                       "finished_at text)"),
        QStringLiteral("create table if not exists metrics ("
                       "id integer primary key autoincrement,"
                       "task_id text not null,"
                       "name text not null,"
                       "value real not null,"
                       "step integer not null,"
                       "epoch integer not null,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists artifacts ("
                       "id integer primary key autoincrement,"
                       "task_id text not null,"
                       "kind text not null,"
                       "path text not null,"
                       "message text,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists datasets ("
                       "id integer primary key autoincrement,"
                       "name text not null,"
                       "format text not null,"
                       "root_path text not null,"
                       "validation_status text,"
                       "sample_count integer not null default 0,"
                       "last_report_json text,"
                       "last_validated_at text,"
                       "created_at text not null,"
                       "updated_at text not null)"),
        QStringLiteral("create table if not exists dataset_versions ("
                       "id integer primary key autoincrement,"
                       "dataset_id integer not null,"
                       "version text not null,"
                       "root_path text not null,"
                       "metadata_json text,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists exports ("
                       "id integer primary key autoincrement,"
                       "task_id text not null,"
                       "source_checkpoint_path text,"
                       "format text not null,"
                       "path text not null,"
                       "config_json text,"
                       "input_shape_json text,"
                       "output_shape_json text,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists plugin_configs ("
                       "id integer primary key autoincrement,"
                       "plugin_id text not null,"
                       "name text not null,"
                       "config_json text not null,"
                       "created_at text not null,"
                       "updated_at text not null,"
                       "unique(plugin_id, name))"),
        QStringLiteral("create table if not exists environment_checks ("
                       "id integer primary key autoincrement,"
                       "name text not null,"
                       "status text not null,"
                       "message text,"
                       "details_json text,"
                       "checked_at text not null)")
    };

    for (const QString& statement : statements) {
        if (!query.exec(statement)) {
            if (error) {
                *error = sqlError(query);
            }
            return false;
        }
    }

    if (!ensureColumn(db_, QStringLiteral("tasks"), QStringLiteral("started_at text"), error)
        || !ensureColumn(db_, QStringLiteral("tasks"), QStringLiteral("finished_at text"), error)
        || !ensureColumn(db_, QStringLiteral("artifacts"), QStringLiteral("message text"), error)
        || !ensureColumn(db_, QStringLiteral("datasets"), QStringLiteral("validation_status text"), error)
        || !ensureColumn(db_, QStringLiteral("datasets"), QStringLiteral("sample_count integer not null default 0"), error)
        || !ensureColumn(db_, QStringLiteral("datasets"), QStringLiteral("last_report_json text"), error)
        || !ensureColumn(db_, QStringLiteral("datasets"), QStringLiteral("last_validated_at text"), error)
        || !ensureColumn(db_, QStringLiteral("exports"), QStringLiteral("source_checkpoint_path text"), error)
        || !ensureColumn(db_, QStringLiteral("exports"), QStringLiteral("input_shape_json text"), error)
        || !ensureColumn(db_, QStringLiteral("exports"), QStringLiteral("output_shape_json text"), error)) {
        return false;
    }
    return true;
}

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

bool ProjectRepository::upsertDatasetValidation(const DatasetRecord& dataset, QString* error)
{
    QSqlQuery findQuery(db_);
    findQuery.prepare(QStringLiteral("select id, created_at from datasets where root_path = ? limit 1"));
    findQuery.addBindValue(dataset.rootPath);
    if (!findQuery.exec()) {
        if (error) {
            *error = sqlError(findQuery);
        }
        return false;
    }

    const QString timestamp = nowIso();
    const QString validatedAt = dataset.lastValidatedAt.isValid()
        ? dataset.lastValidatedAt.toUTC().toString(Qt::ISODateWithMs)
        : timestamp;
    const bool exists = findQuery.next();
    const int datasetId = exists ? findQuery.value(0).toInt() : 0;
    const QString createdAt = exists ? findQuery.value(1).toString() : timestamp;

    QSqlQuery query(db_);
    if (exists) {
        query.prepare(QStringLiteral("update datasets set name = ?, format = ?, validation_status = ?, sample_count = ?, "
                                     "last_report_json = ?, last_validated_at = ?, updated_at = ? where id = ?"));
        query.addBindValue(dataset.name);
        query.addBindValue(dataset.format);
        query.addBindValue(dataset.validationStatus);
        query.addBindValue(dataset.sampleCount);
        query.addBindValue(dataset.lastReportJson);
        query.addBindValue(validatedAt);
        query.addBindValue(timestamp);
        query.addBindValue(datasetId);
    } else {
        query.prepare(QStringLiteral("insert into datasets(name, format, root_path, validation_status, sample_count, last_report_json, last_validated_at, created_at, updated_at) "
                                     "values(?, ?, ?, ?, ?, ?, ?, ?, ?)"));
        query.addBindValue(dataset.name);
        query.addBindValue(dataset.format);
        query.addBindValue(dataset.rootPath);
        query.addBindValue(dataset.validationStatus);
        query.addBindValue(dataset.sampleCount);
        query.addBindValue(dataset.lastReportJson);
        query.addBindValue(validatedAt);
        query.addBindValue(createdAt);
        query.addBindValue(timestamp);
    }
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }

    const int versionDatasetId = exists ? datasetId : query.lastInsertId().toInt();
    QSqlQuery versionQuery(db_);
    versionQuery.prepare(QStringLiteral("insert into dataset_versions(dataset_id, version, root_path, metadata_json, created_at) values(?, ?, ?, ?, ?)"));
    versionQuery.addBindValue(versionDatasetId);
    versionQuery.addBindValue(validatedAt);
    versionQuery.addBindValue(dataset.rootPath);
    versionQuery.addBindValue(dataset.lastReportJson);
    versionQuery.addBindValue(timestamp);
    if (!versionQuery.exec()) {
        if (error) {
            *error = sqlError(versionQuery);
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

QVector<DatasetRecord> ProjectRepository::recentDatasets(int limit, QString* error) const
{
    QVector<DatasetRecord> datasets;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, name, format, root_path, validation_status, sample_count, last_report_json, created_at, updated_at, last_validated_at "
                                 "from datasets order by updated_at desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return datasets;
    }

    while (query.next()) {
        DatasetRecord dataset;
        dataset.id = query.value(0).toInt();
        dataset.name = query.value(1).toString();
        dataset.format = query.value(2).toString();
        dataset.rootPath = query.value(3).toString();
        dataset.validationStatus = query.value(4).toString();
        dataset.sampleCount = query.value(5).toInt();
        dataset.lastReportJson = query.value(6).toString();
        dataset.createdAt = dateTimeFromIso(query.value(7).toString());
        dataset.updatedAt = dateTimeFromIso(query.value(8).toString());
        dataset.lastValidatedAt = dateTimeFromIso(query.value(9).toString());
        datasets.append(dataset);
    }
    return datasets;
}

DatasetRecord ProjectRepository::datasetByRootPath(const QString& rootPath, QString* error) const
{
    DatasetRecord dataset;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, name, format, root_path, validation_status, sample_count, last_report_json, created_at, updated_at, last_validated_at "
                                 "from datasets where root_path = ? order by updated_at desc limit 1"));
    query.addBindValue(rootPath);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return dataset;
    }
    if (!query.next()) {
        return dataset;
    }
    dataset.id = query.value(0).toInt();
    dataset.name = query.value(1).toString();
    dataset.format = query.value(2).toString();
    dataset.rootPath = query.value(3).toString();
    dataset.validationStatus = query.value(4).toString();
    dataset.sampleCount = query.value(5).toInt();
    dataset.lastReportJson = query.value(6).toString();
    dataset.createdAt = dateTimeFromIso(query.value(7).toString());
    dataset.updatedAt = dateTimeFromIso(query.value(8).toString());
    dataset.lastValidatedAt = dateTimeFromIso(query.value(9).toString());
    return dataset;
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
