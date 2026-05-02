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
                       "checked_at text not null)"),
        QStringLiteral("create table if not exists experiments ("
                       "id integer primary key autoincrement,"
                       "name text not null,"
                       "task_type text,"
                       "dataset_id integer,"
                       "notes text,"
                       "tags_json text,"
                       "created_at text not null,"
                       "updated_at text not null,"
                       "unique(name, task_type))"),
        QStringLiteral("create table if not exists experiment_runs ("
                       "id integer primary key autoincrement,"
                       "experiment_id integer not null,"
                       "task_id text,"
                       "training_backend text,"
                       "model_preset text,"
                       "dataset_snapshot_id integer,"
                       "request_json text,"
                       "environment_json text,"
                       "best_metrics_json text,"
                       "artifact_summary_json text,"
                       "created_at text not null,"
                       "updated_at text not null)"),
        QStringLiteral("create table if not exists dataset_snapshots ("
                       "id integer primary key autoincrement,"
                       "dataset_id integer,"
                       "name text,"
                       "root_path text not null,"
                       "manifest_path text not null,"
                       "content_hash text not null,"
                       "file_count integer not null default 0,"
                       "total_bytes integer not null default 0,"
                       "metadata_json text,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists model_versions ("
                       "id integer primary key autoincrement,"
                       "model_name text not null,"
                       "version text not null,"
                       "source_task_id text,"
                       "experiment_run_id integer,"
                       "dataset_snapshot_id integer,"
                       "checkpoint_path text,"
                       "onnx_path text,"
                       "tensorrt_engine_path text,"
                       "evaluation_report_id integer,"
                       "status text not null,"
                       "notes text,"
                       "metrics_json text,"
                       "created_at text not null,"
                       "updated_at text not null,"
                       "unique(model_name, version))"),
        QStringLiteral("create table if not exists evaluation_reports ("
                       "id integer primary key autoincrement,"
                       "task_id text,"
                       "model_path text not null,"
                       "task_type text,"
                       "dataset_snapshot_id integer,"
                       "report_path text not null,"
                       "summary_json text,"
                       "created_at text not null)"),
        QStringLiteral("create table if not exists pipeline_runs ("
                       "id integer primary key autoincrement,"
                       "name text not null,"
                       "template_id text not null,"
                       "task_ids_json text,"
                       "state text not null,"
                       "summary_json text,"
                       "created_at text not null,"
                       "updated_at text not null)")
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

int ProjectRepository::upsertExperiment(const ExperimentRecord& experiment, QString* error)
{
    QSqlQuery query(db_);
    const QString timestamp = nowIso();
    query.prepare(QStringLiteral("insert into experiments(name, task_type, dataset_id, notes, tags_json, created_at, updated_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?) "
                                 "on conflict(name, task_type) do update set "
                                 "dataset_id = excluded.dataset_id, notes = excluded.notes, tags_json = excluded.tags_json, updated_at = excluded.updated_at"));
    query.addBindValue(experiment.name);
    query.addBindValue(experiment.taskType);
    query.addBindValue(experiment.datasetId);
    query.addBindValue(experiment.notes);
    query.addBindValue(experiment.tagsJson);
    query.addBindValue(experiment.createdAt.isValid() ? dateTimeToIso(experiment.createdAt) : timestamp);
    query.addBindValue(timestamp);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }

    QSqlQuery readQuery(db_);
    readQuery.prepare(QStringLiteral("select id from experiments where name = ? and task_type = ? order by id desc limit 1"));
    readQuery.addBindValue(experiment.name);
    readQuery.addBindValue(experiment.taskType);
    if (!readQuery.exec()) {
        if (error) {
            *error = sqlError(readQuery);
        }
        return 0;
    }
    return readQuery.next() ? readQuery.value(0).toInt() : 0;
}

int ProjectRepository::insertExperimentRun(const ExperimentRunRecord& run, QString* error)
{
    if (!run.taskId.isEmpty()) {
        const ExperimentRunRecord existing = experimentRunForTask(run.taskId, error);
        if (existing.id > 0) {
            return existing.id;
        }
    }

    QSqlQuery query(db_);
    const QString timestamp = nowIso();
    query.prepare(QStringLiteral("insert into experiment_runs(experiment_id, task_id, training_backend, model_preset, dataset_snapshot_id, request_json, "
                                 "environment_json, best_metrics_json, artifact_summary_json, created_at, updated_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"));
    query.addBindValue(run.experimentId);
    query.addBindValue(run.taskId);
    query.addBindValue(run.trainingBackend);
    query.addBindValue(run.modelPreset);
    query.addBindValue(run.datasetSnapshotId);
    query.addBindValue(run.requestJson);
    query.addBindValue(run.environmentJson);
    query.addBindValue(run.bestMetricsJson);
    query.addBindValue(run.artifactSummaryJson);
    query.addBindValue(run.createdAt.isValid() ? dateTimeToIso(run.createdAt) : timestamp);
    query.addBindValue(run.updatedAt.isValid() ? dateTimeToIso(run.updatedAt) : timestamp);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }
    return query.lastInsertId().toInt();
}

bool ProjectRepository::updateExperimentRunSummary(const QString& taskId, const QString& bestMetricsJson, const QString& artifactSummaryJson, QString* error)
{
    if (taskId.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Task id is required to update experiment run summary");
        }
        return false;
    }

    QSqlQuery query(db_);
    query.prepare(QStringLiteral("update experiment_runs set best_metrics_json = ?, artifact_summary_json = ?, updated_at = ? where task_id = ?"));
    query.addBindValue(bestMetricsJson);
    query.addBindValue(artifactSummaryJson);
    query.addBindValue(nowIso());
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return false;
    }
    return true;
}

int ProjectRepository::insertDatasetSnapshot(const DatasetSnapshotRecord& snapshot, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into dataset_snapshots(dataset_id, name, root_path, manifest_path, content_hash, file_count, total_bytes, metadata_json, created_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?, ?, ?)"));
    query.addBindValue(snapshot.datasetId);
    query.addBindValue(snapshot.name);
    query.addBindValue(snapshot.rootPath);
    query.addBindValue(snapshot.manifestPath);
    query.addBindValue(snapshot.contentHash);
    query.addBindValue(snapshot.fileCount);
    query.addBindValue(snapshot.totalBytes);
    query.addBindValue(snapshot.metadataJson);
    query.addBindValue(snapshot.createdAt.isValid() ? dateTimeToIso(snapshot.createdAt) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }
    return query.lastInsertId().toInt();
}

int ProjectRepository::upsertModelVersion(const ModelVersionRecord& modelVersion, QString* error)
{
    QSqlQuery query(db_);
    const QString timestamp = nowIso();
    query.prepare(QStringLiteral("insert into model_versions(model_name, version, source_task_id, experiment_run_id, dataset_snapshot_id, checkpoint_path, onnx_path, "
                                 "tensorrt_engine_path, evaluation_report_id, status, notes, metrics_json, created_at, updated_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                                 "on conflict(model_name, version) do update set "
                                 "source_task_id = excluded.source_task_id, experiment_run_id = excluded.experiment_run_id, dataset_snapshot_id = excluded.dataset_snapshot_id, "
                                 "checkpoint_path = excluded.checkpoint_path, onnx_path = excluded.onnx_path, tensorrt_engine_path = excluded.tensorrt_engine_path, "
                                 "evaluation_report_id = excluded.evaluation_report_id, status = excluded.status, notes = excluded.notes, metrics_json = excluded.metrics_json, "
                                 "updated_at = excluded.updated_at"));
    query.addBindValue(modelVersion.modelName);
    query.addBindValue(modelVersion.version);
    query.addBindValue(modelVersion.sourceTaskId);
    query.addBindValue(modelVersion.experimentRunId);
    query.addBindValue(modelVersion.datasetSnapshotId);
    query.addBindValue(modelVersion.checkpointPath);
    query.addBindValue(modelVersion.onnxPath);
    query.addBindValue(modelVersion.tensorRtEnginePath);
    query.addBindValue(modelVersion.evaluationReportId);
    query.addBindValue(modelVersion.status.isEmpty() ? QStringLiteral("draft") : modelVersion.status);
    query.addBindValue(modelVersion.notes);
    query.addBindValue(modelVersion.metricsJson);
    query.addBindValue(modelVersion.createdAt.isValid() ? dateTimeToIso(modelVersion.createdAt) : timestamp);
    query.addBindValue(timestamp);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }

    QSqlQuery readQuery(db_);
    readQuery.prepare(QStringLiteral("select id from model_versions where model_name = ? and version = ? order by id desc limit 1"));
    readQuery.addBindValue(modelVersion.modelName);
    readQuery.addBindValue(modelVersion.version);
    if (!readQuery.exec()) {
        if (error) {
            *error = sqlError(readQuery);
        }
        return 0;
    }
    return readQuery.next() ? readQuery.value(0).toInt() : 0;
}

int ProjectRepository::insertEvaluationReport(const EvaluationReportRecord& report, QString* error)
{
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("insert into evaluation_reports(task_id, model_path, task_type, dataset_snapshot_id, report_path, summary_json, created_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?)"));
    query.addBindValue(report.taskId);
    query.addBindValue(report.modelPath);
    query.addBindValue(report.taskType);
    query.addBindValue(report.datasetSnapshotId);
    query.addBindValue(report.reportPath);
    query.addBindValue(report.summaryJson);
    query.addBindValue(report.createdAt.isValid() ? dateTimeToIso(report.createdAt) : nowIso());
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }
    return query.lastInsertId().toInt();
}

int ProjectRepository::insertPipelineRun(const PipelineRunRecord& pipelineRun, QString* error)
{
    QSqlQuery query(db_);
    const QString timestamp = nowIso();
    query.prepare(QStringLiteral("insert into pipeline_runs(name, template_id, task_ids_json, state, summary_json, created_at, updated_at) "
                                 "values(?, ?, ?, ?, ?, ?, ?)"));
    query.addBindValue(pipelineRun.name);
    query.addBindValue(pipelineRun.templateId);
    query.addBindValue(pipelineRun.taskIdsJson);
    query.addBindValue(pipelineRun.state.isEmpty() ? QStringLiteral("completed") : pipelineRun.state);
    query.addBindValue(pipelineRun.summaryJson);
    query.addBindValue(pipelineRun.createdAt.isValid() ? dateTimeToIso(pipelineRun.createdAt) : timestamp);
    query.addBindValue(pipelineRun.updatedAt.isValid() ? dateTimeToIso(pipelineRun.updatedAt) : timestamp);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }
    return query.lastInsertId().toInt();
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

QVector<DatasetVersionRecord> ProjectRepository::datasetVersions(int datasetId, QString* error) const
{
    QVector<DatasetVersionRecord> versions;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, dataset_id, version, root_path, metadata_json, created_at "
                                 "from dataset_versions where dataset_id = ? order by created_at desc, id desc"));
    query.addBindValue(datasetId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return versions;
    }

    while (query.next()) {
        DatasetVersionRecord version;
        version.id = query.value(0).toInt();
        version.datasetId = query.value(1).toInt();
        version.version = query.value(2).toString();
        version.rootPath = query.value(3).toString();
        version.metadataJson = query.value(4).toString();
        version.createdAt = dateTimeFromIso(query.value(5).toString());
        versions.append(version);
    }
    return versions;
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

QVector<ExperimentRecord> ProjectRepository::recentExperiments(int limit, QString* error) const
{
    QVector<ExperimentRecord> experiments;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, name, task_type, dataset_id, notes, tags_json, created_at, updated_at "
                                 "from experiments order by updated_at desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return experiments;
    }
    while (query.next()) {
        ExperimentRecord record;
        record.id = query.value(0).toInt();
        record.name = query.value(1).toString();
        record.taskType = query.value(2).toString();
        record.datasetId = query.value(3).toInt();
        record.notes = query.value(4).toString();
        record.tagsJson = query.value(5).toString();
        record.createdAt = dateTimeFromIso(query.value(6).toString());
        record.updatedAt = dateTimeFromIso(query.value(7).toString());
        experiments.append(record);
    }
    return experiments;
}

QVector<ExperimentRunRecord> ProjectRepository::experimentRuns(int experimentId, QString* error) const
{
    QVector<ExperimentRunRecord> runs;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, experiment_id, task_id, training_backend, model_preset, dataset_snapshot_id, request_json, "
                                 "environment_json, best_metrics_json, artifact_summary_json, created_at, updated_at "
                                 "from experiment_runs where experiment_id = ? order by created_at desc, id desc"));
    query.addBindValue(experimentId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return runs;
    }
    while (query.next()) {
        ExperimentRunRecord record;
        record.id = query.value(0).toInt();
        record.experimentId = query.value(1).toInt();
        record.taskId = query.value(2).toString();
        record.trainingBackend = query.value(3).toString();
        record.modelPreset = query.value(4).toString();
        record.datasetSnapshotId = query.value(5).toInt();
        record.requestJson = query.value(6).toString();
        record.environmentJson = query.value(7).toString();
        record.bestMetricsJson = query.value(8).toString();
        record.artifactSummaryJson = query.value(9).toString();
        record.createdAt = dateTimeFromIso(query.value(10).toString());
        record.updatedAt = dateTimeFromIso(query.value(11).toString());
        runs.append(record);
    }
    return runs;
}

ExperimentRunRecord ProjectRepository::experimentRunForTask(const QString& taskId, QString* error) const
{
    ExperimentRunRecord record;
    if (taskId.isEmpty()) {
        return record;
    }

    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, experiment_id, task_id, training_backend, model_preset, dataset_snapshot_id, request_json, "
                                 "environment_json, best_metrics_json, artifact_summary_json, created_at, updated_at "
                                 "from experiment_runs where task_id = ? order by created_at desc, id desc limit 1"));
    query.addBindValue(taskId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return record;
    }
    if (query.next()) {
        record.id = query.value(0).toInt();
        record.experimentId = query.value(1).toInt();
        record.taskId = query.value(2).toString();
        record.trainingBackend = query.value(3).toString();
        record.modelPreset = query.value(4).toString();
        record.datasetSnapshotId = query.value(5).toInt();
        record.requestJson = query.value(6).toString();
        record.environmentJson = query.value(7).toString();
        record.bestMetricsJson = query.value(8).toString();
        record.artifactSummaryJson = query.value(9).toString();
        record.createdAt = dateTimeFromIso(query.value(10).toString());
        record.updatedAt = dateTimeFromIso(query.value(11).toString());
    }
    return record;
}

QVector<DatasetSnapshotRecord> ProjectRepository::datasetSnapshots(int datasetId, QString* error) const
{
    QVector<DatasetSnapshotRecord> snapshots;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, dataset_id, name, root_path, manifest_path, content_hash, file_count, total_bytes, metadata_json, created_at "
                                 "from dataset_snapshots where dataset_id = ? order by created_at desc, id desc"));
    query.addBindValue(datasetId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return snapshots;
    }
    while (query.next()) {
        DatasetSnapshotRecord record;
        record.id = query.value(0).toInt();
        record.datasetId = query.value(1).toInt();
        record.name = query.value(2).toString();
        record.rootPath = query.value(3).toString();
        record.manifestPath = query.value(4).toString();
        record.contentHash = query.value(5).toString();
        record.fileCount = query.value(6).toInt();
        record.totalBytes = query.value(7).toLongLong();
        record.metadataJson = query.value(8).toString();
        record.createdAt = dateTimeFromIso(query.value(9).toString());
        snapshots.append(record);
    }
    return snapshots;
}

DatasetSnapshotRecord ProjectRepository::datasetSnapshotById(int snapshotId, QString* error) const
{
    DatasetSnapshotRecord record;
    if (snapshotId <= 0) {
        return record;
    }

    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, dataset_id, name, root_path, manifest_path, content_hash, file_count, total_bytes, metadata_json, created_at "
                                 "from dataset_snapshots where id = ? limit 1"));
    query.addBindValue(snapshotId);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return record;
    }
    if (query.next()) {
        record.id = query.value(0).toInt();
        record.datasetId = query.value(1).toInt();
        record.name = query.value(2).toString();
        record.rootPath = query.value(3).toString();
        record.manifestPath = query.value(4).toString();
        record.contentHash = query.value(5).toString();
        record.fileCount = query.value(6).toInt();
        record.totalBytes = query.value(7).toLongLong();
        record.metadataJson = query.value(8).toString();
        record.createdAt = dateTimeFromIso(query.value(9).toString());
    }
    return record;
}

DatasetSnapshotRecord ProjectRepository::latestDatasetSnapshot(int datasetId, QString* error) const
{
    const QVector<DatasetSnapshotRecord> snapshots = datasetSnapshots(datasetId, error);
    return snapshots.isEmpty() ? DatasetSnapshotRecord() : snapshots.first();
}

QVector<ModelVersionRecord> ProjectRepository::recentModelVersions(int limit, QString* error) const
{
    QVector<ModelVersionRecord> versions;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, model_name, version, source_task_id, experiment_run_id, dataset_snapshot_id, checkpoint_path, onnx_path, "
                                 "tensorrt_engine_path, evaluation_report_id, status, notes, metrics_json, created_at, updated_at "
                                 "from model_versions order by updated_at desc, id desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return versions;
    }
    while (query.next()) {
        ModelVersionRecord record;
        record.id = query.value(0).toInt();
        record.modelName = query.value(1).toString();
        record.version = query.value(2).toString();
        record.sourceTaskId = query.value(3).toString();
        record.experimentRunId = query.value(4).toInt();
        record.datasetSnapshotId = query.value(5).toInt();
        record.checkpointPath = query.value(6).toString();
        record.onnxPath = query.value(7).toString();
        record.tensorRtEnginePath = query.value(8).toString();
        record.evaluationReportId = query.value(9).toInt();
        record.status = query.value(10).toString();
        record.notes = query.value(11).toString();
        record.metricsJson = query.value(12).toString();
        record.createdAt = dateTimeFromIso(query.value(13).toString());
        record.updatedAt = dateTimeFromIso(query.value(14).toString());
        versions.append(record);
    }
    return versions;
}

QVector<EvaluationReportRecord> ProjectRepository::recentEvaluationReports(int limit, QString* error) const
{
    QVector<EvaluationReportRecord> reports;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, task_id, model_path, task_type, dataset_snapshot_id, report_path, summary_json, created_at "
                                 "from evaluation_reports order by created_at desc, id desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return reports;
    }
    while (query.next()) {
        EvaluationReportRecord record;
        record.id = query.value(0).toInt();
        record.taskId = query.value(1).toString();
        record.modelPath = query.value(2).toString();
        record.taskType = query.value(3).toString();
        record.datasetSnapshotId = query.value(4).toInt();
        record.reportPath = query.value(5).toString();
        record.summaryJson = query.value(6).toString();
        record.createdAt = dateTimeFromIso(query.value(7).toString());
        reports.append(record);
    }
    return reports;
}

QVector<PipelineRunRecord> ProjectRepository::recentPipelineRuns(int limit, QString* error) const
{
    QVector<PipelineRunRecord> runs;
    QSqlQuery query(db_);
    query.prepare(QStringLiteral("select id, name, template_id, task_ids_json, state, summary_json, created_at, updated_at "
                                 "from pipeline_runs order by updated_at desc, id desc limit ?"));
    query.addBindValue(limit);
    if (!query.exec()) {
        if (error) {
            *error = sqlError(query);
        }
        return runs;
    }
    while (query.next()) {
        PipelineRunRecord record;
        record.id = query.value(0).toInt();
        record.name = query.value(1).toString();
        record.templateId = query.value(2).toString();
        record.taskIdsJson = query.value(3).toString();
        record.state = query.value(4).toString();
        record.summaryJson = query.value(5).toString();
        record.createdAt = dateTimeFromIso(query.value(6).toString());
        record.updatedAt = dateTimeFromIso(query.value(7).toString());
        runs.append(record);
    }
    return runs;
}

} // namespace aitrain
