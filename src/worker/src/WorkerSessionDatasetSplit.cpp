#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSplitWorkflow(const wp::DatasetSplitCommand& command)
{
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
        fail(QStringLiteral("Dataset Split  请求缺少项目、源四重身份或目标 Dataset 身份：%1").arg(error));
        return;
    }

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Dataset Split  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("dataset.split"), QStringLiteral("dataset_split"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Dataset Split  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Dataset Split 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();
    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Split  正在核对源快照并生成不可变计划。")}});

    aitrain::DatasetSplitWorkflowResult result;
    const bool executed = activeWorkspace->runDatasetSplitWorkflow(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Dataset Split  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查源四重身份与  工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            activeWorkspace->finalizeTask(taskId,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        publishPersistedTerminal(taskId);
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

    publishPersistedTerminal(taskId, QStringLiteral("Dataset Split  completed"));
}
