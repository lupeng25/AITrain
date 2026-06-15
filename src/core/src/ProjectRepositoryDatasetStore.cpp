#include "aitrain/core/ProjectRepository.h"

#include "ProjectRepositoryInternal.h"

#include <QSqlQuery>
#include <QVariant>

namespace aitrain {

using namespace repository_internal;

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

} // namespace aitrain
