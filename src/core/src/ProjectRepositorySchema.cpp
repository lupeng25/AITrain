#include "aitrain/core/ProjectRepository.h"

#include "ProjectRepositoryInternal.h"

#include <QSqlError>
#include <QSqlQuery>
#include <QStringList>
#include <QUuid>
#include <QVariant>

namespace aitrain {

using namespace repository_internal;

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

    if (!db_.transaction()) {
        if (error) {
            *error = db_.lastError().text();
        }
        return false;
    }

    if (!ensureSchemaMigrationTable(db_, error)) {
        db_.rollback();
        return false;
    }

    for (const QString& statement : statements) {
        if (!execStatement(db_, statement, error)) {
            db_.rollback();
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
        db_.rollback();
        return false;
    }

    if (!recordBaselineMigration(db_, error)) {
        db_.rollback();
        return false;
    }

    if (!db_.commit()) {
        if (error) {
            *error = db_.lastError().text();
        }
        db_.rollback();
        return false;
    }
    return true;
}

int ProjectRepository::currentSchemaVersion()
{
    return kCurrentSchemaVersion;
}

int ProjectRepository::schemaVersion(QString* error) const
{
    QSqlQuery query(db_);
    if (!query.exec(QStringLiteral("select coalesce(max(version), 0) from schema_migrations"))) {
        if (error) {
            *error = sqlError(query);
        }
        return 0;
    }
    if (!query.next()) {
        return 0;
    }
    return query.value(0).toInt();
}

QVector<int> ProjectRepository::appliedSchemaMigrations(QString* error) const
{
    QVector<int> versions;
    QSqlQuery query(db_);
    if (!query.exec(QStringLiteral("select version from schema_migrations order by version asc"))) {
        if (error) {
            *error = sqlError(query);
        }
        return versions;
    }
    while (query.next()) {
        versions.append(query.value(0).toInt());
    }
    return versions;
}

} // namespace aitrain
