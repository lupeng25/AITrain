#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSnapshotImportWorkflow(const wp::DatasetSnapshotImportCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DatasetSnapshotImportWorkflowRequest request;
    request.sourcePath = command.sourcePath.trimmed();
    request.sourceFormat = command.sourceFormat.trimmed();
    request.targetDatasetName = command.targetDatasetName.trimmed();
    request.options = command.options;
    QString error;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.sourcePath.isEmpty() || request.sourceFormat.isEmpty()
        || request.targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(
            command.targetDatasetId,
            &request.targetDatasetId, &error)) {
        fail(QStringLiteral("Dataset Snapshot Import  请求缺少项目、外部源、格式或目标 Dataset 身份：%1")
            .arg(error));
        return;
    }

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Dataset Snapshot Import  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("dataset.snapshot.import"), QStringLiteral("dataset_snapshot_import"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Dataset Snapshot Import  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Dataset Snapshot Import 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();
    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Snapshot Import  正在冻结并物化外部数据。")}});

    aitrain::DatasetSnapshotImportWorkflowResult result;
    const bool executed = activeWorkspace->runDatasetSnapshotImportWorkflow(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty()
                ? QStringLiteral("Dataset Snapshot Import  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查外部源与  工作区后重试。");
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

    publishPersistedTerminal(taskId,
        QStringLiteral("Dataset Snapshot Import  completed"));
}
