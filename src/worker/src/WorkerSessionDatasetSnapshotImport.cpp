#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSnapshotImportWorkflow(const wp::DatasetSnapshotImportCommand& command)
{
    if (running_ || datasetSnapshotImportWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Snapshot Import 。"));
        return;
    }
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DatasetSnapshotImportWorkflowRequest request;
    request.sourcePath = command.sourcePath.trimmed();
    request.sourceFormat = command.sourceFormat.trimmed();
    request.targetDatasetName = command.targetDatasetName.trimmed();
    request.options = command.options;
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &datasetSnapshotImportTaskId_, &error)
        || datasetSnapshotImportTaskId_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.sourcePath.isEmpty() || request.sourceFormat.isEmpty()
        || request.targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(
            command.targetDatasetId,
            &request.targetDatasetId, &error)) {
        datasetSnapshotImportTaskId_ = {};
        fail(QStringLiteral("Dataset Snapshot Import  请求缺少项目、外部源、格式或目标 Dataset 身份：%1")
            .arg(error));
        return;
    }

    datasetSnapshotImportWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!datasetSnapshotImportWorkspace_->openForWorkerChild(projectRoot, &error)) {
        datasetSnapshotImportWorkspace_.reset();
        datasetSnapshotImportTaskId_ = {};
        fail(QStringLiteral("无法打开 Dataset Snapshot Import  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!datasetSnapshotImportWorkspace_->startTask(datasetSnapshotImportTaskId_,
            QStringLiteral("dataset.snapshot.import"), QStringLiteral("dataset_snapshot_import"),
            &task, &error)) {
        datasetSnapshotImportWorkspace_.reset();
        datasetSnapshotImportTaskId_ = {};
        fail(QStringLiteral("无法创建 Dataset Snapshot Import  根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetSnapshotImportRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Snapshot Import  正在冻结并物化外部数据。")}});

    aitrain::DatasetSnapshotImportWorkflowResult result;
    const bool executed = datasetSnapshotImportWorkspace_->runDatasetSnapshotImportWorkflow(
        datasetSnapshotImportTaskId_, request, &result, &error, pollingCancellationCallback(0));
    datasetSnapshotImportRunning_ = false;
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (datasetSnapshotImportWorkspace_->task(datasetSnapshotImportTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty()
                ? QStringLiteral("Dataset Snapshot Import  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查外部源与  工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetSnapshotImportWorkspace_->finalizeTask(datasetSnapshotImportTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        datasetSnapshotImportWorkspace_.reset();
        datasetSnapshotImportTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Snapshot Import  执行失败：%1").arg(error),
            QStringLiteral("dataset_snapshot_import_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("datasetId"), result.datasetSnapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("importPlanArtifactId"), result.importPlanArtifactId.toString()},
        {QStringLiteral("snapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Snapshot Import  已登记快照。"));
    }
    send(wp::event::datasetSnapshotImportWorkflow(), response);

    const aitrain::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetSnapshotImportWorkspace_.reset();
    datasetSnapshotImportTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_snapshot_import_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Snapshot Import  completed")}});
        finishSession();
    }
}
