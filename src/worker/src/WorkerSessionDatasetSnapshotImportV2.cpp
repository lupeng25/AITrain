#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSnapshotImportWorkflowV2(const QJsonObject& payload)
{
    if (running_ || datasetSnapshotImportWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Snapshot Import V2。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    aitrain::v2::DatasetSnapshotImportWorkflowRequestV2 request;
    request.sourcePath = payload.value(wp::field::sourcePath()).toString().trimmed();
    request.sourceFormat = payload.value(wp::field::sourceFormat()).toString().trimmed();
    request.targetDatasetName = payload.value(QStringLiteral("targetDatasetName")).toString().trimmed();
    request.options = payload.value(wp::field::options()).toObject();
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &datasetSnapshotImportTaskIdV2_, &error)
        || datasetSnapshotImportTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.sourcePath.isEmpty() || request.sourceFormat.isEmpty()
        || request.targetDatasetName.isEmpty()
        || !aitrain::v2::DatasetId::parse(
            payload.value(QStringLiteral("targetDatasetId")).toString(),
            &request.targetDatasetId, &error)) {
        datasetSnapshotImportTaskIdV2_ = {};
        fail(QStringLiteral("Dataset Snapshot Import V2 请求缺少项目、外部源、格式或目标 Dataset 身份：%1")
            .arg(error));
        return;
    }

    datasetSnapshotImportWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!datasetSnapshotImportWorkspaceV2_->open(projectRoot, &error)) {
        datasetSnapshotImportWorkspaceV2_.reset();
        datasetSnapshotImportTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Dataset Snapshot Import V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!datasetSnapshotImportWorkspaceV2_->startTask(datasetSnapshotImportTaskIdV2_,
            QStringLiteral("dataset.snapshot.import.v2"), QStringLiteral("dataset_snapshot_import"),
            &task, &error)) {
        datasetSnapshotImportWorkspaceV2_.reset();
        datasetSnapshotImportTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Dataset Snapshot Import V2 根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetSnapshotImportRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Snapshot Import V2 正在冻结并物化外部数据。")}});

    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 result;
    const bool executed = datasetSnapshotImportWorkspaceV2_->runDatasetSnapshotImportWorkflow(
        datasetSnapshotImportTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    datasetSnapshotImportRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (datasetSnapshotImportWorkspaceV2_->task(datasetSnapshotImportTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            aitrain::v2::Failure failure;
            failure.code = aitrain::v2::FailureCode::InternalError;
            failure.message = error.isEmpty()
                ? QStringLiteral("Dataset Snapshot Import V2 执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查外部源与 V2 工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetSnapshotImportWorkspaceV2_->finalizeTask(datasetSnapshotImportTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        datasetSnapshotImportWorkspaceV2_.reset();
        datasetSnapshotImportTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Snapshot Import V2 执行失败：%1").arg(error),
            QStringLiteral("dataset_snapshot_import_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("datasetId"), result.datasetSnapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("importPlanArtifactId"), result.importPlanArtifactId.toString()},
        {QStringLiteral("snapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Snapshot Import V2 已登记快照。"));
    }
    send(wp::event::datasetSnapshotImportWorkflowV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetSnapshotImportWorkspaceV2_.reset();
    datasetSnapshotImportTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_snapshot_import_v2_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Snapshot Import V2 completed")}});
        finishSession();
    }
}
