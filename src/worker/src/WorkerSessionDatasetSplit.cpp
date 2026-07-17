#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSplitWorkflow(const wp::DatasetSplitCommand& command)
{
    if (running_ || datasetSplitWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Split 。"));
        return;
    }
    const aitrain::TaskId taskId = command.context.taskId;
    const QString taskIdText = taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DatasetSplitWorkflowRequest request;
    request.targetDatasetName = command.targetDatasetName.trimmed();
    request.options = command.options;
    QString error;
    if (!taskId.isValid()) {
        error = QStringLiteral("TaskId 无效。");
    }
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(command.sourceDatasetId,
            &request.sourceDatasetId, &error)
        || !aitrain::DatasetVersionId::parse(command.sourceDatasetVersionId,
            &request.sourceDatasetVersionId, &error)
        || !aitrain::SnapshotId::parse(command.sourceSnapshotId,
            &request.sourceSnapshotId, &error)
        || !aitrain::ArtifactId::parse(command.sourceSnapshotArtifactId,
            &request.sourceSnapshotArtifactId, &error)
        || !aitrain::DatasetId::parse(command.targetDatasetId,
            &request.targetDatasetId, &error)) {
        datasetSplitTaskId_ = {};
        fail(QStringLiteral("Dataset Split  请求缺少项目、源四重身份或目标 Dataset 身份：%1").arg(error));
        return;
    }
    datasetSplitTaskId_ = taskId;

    datasetSplitWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!datasetSplitWorkspace_->open(projectRoot, &error)) {
        datasetSplitWorkspace_.reset();
        datasetSplitTaskId_ = {};
        fail(QStringLiteral("无法打开 Dataset Split  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!datasetSplitWorkspace_->startTask(datasetSplitTaskId_,
            QStringLiteral("dataset.split"), QStringLiteral("dataset_split"),
            &task, &error)) {
        datasetSplitWorkspace_.reset();
        datasetSplitTaskId_ = {};
        fail(QStringLiteral("无法创建 Dataset Split  根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetSplitRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Split  正在核对源快照并生成不可变计划。")}});

    aitrain::DatasetSplitWorkflowResult result;
    const bool executed = datasetSplitWorkspace_->runDatasetSplitWorkflow(
        datasetSplitTaskId_, request, &result, &error, pollingCancellationCallback(0));
    datasetSplitRunning_ = false;
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (datasetSplitWorkspace_->task(datasetSplitTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Dataset Split  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查源四重身份与  工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetSplitWorkspace_->finalizeTask(datasetSplitTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        datasetSplitWorkspace_.reset();
        datasetSplitTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Split  执行失败：%1").arg(error),
            QStringLiteral("dataset_split_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("datasetId"), result.datasetSnapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("splitPlanArtifactId"), result.splitPlanArtifactId.toString()},
        {QStringLiteral("splitArtifactId"), result.splitArtifactId.toString()},
        {QStringLiteral("snapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Split  已登记目标快照。"));
    }
    send(wp::event::datasetSplitWorkflow(), response);

    const aitrain::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetSplitWorkspace_.reset();
    datasetSplitTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_split_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Split  completed")}});
        finishSession();
    }
}
