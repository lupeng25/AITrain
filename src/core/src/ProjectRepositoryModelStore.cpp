#include "aitrain/core/ProjectRepository.h"

#include "ProjectRepositoryInternal.h"

#include <QSqlQuery>
#include <QVariant>

namespace aitrain {

using namespace repository_internal;

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
