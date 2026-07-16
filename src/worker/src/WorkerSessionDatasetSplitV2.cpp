#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetSplitWorkflowV2(const QJsonObject& payload)
{
    if (running_ || datasetSplitWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Split V2。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    aitrain::v2::DatasetSplitWorkflowRequestV2 request;
    request.targetDatasetName = payload.value(QStringLiteral("targetDatasetName")).toString().trimmed();
    request.options = payload.value(wp::field::options()).toObject();
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &datasetSplitTaskIdV2_, &error)
        || datasetSplitTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.targetDatasetName.isEmpty()
        || !aitrain::v2::DatasetId::parse(payload.value(QStringLiteral("sourceDatasetId")).toString(),
            &request.sourceDatasetId, &error)
        || !aitrain::v2::DatasetVersionId::parse(payload.value(QStringLiteral("sourceDatasetVersionId")).toString(),
            &request.sourceDatasetVersionId, &error)
        || !aitrain::v2::SnapshotId::parse(payload.value(QStringLiteral("sourceSnapshotId")).toString(),
            &request.sourceSnapshotId, &error)
        || !aitrain::v2::ArtifactId::parse(payload.value(QStringLiteral("sourceSnapshotArtifactId")).toString(),
            &request.sourceSnapshotArtifactId, &error)
        || !aitrain::v2::DatasetId::parse(payload.value(QStringLiteral("targetDatasetId")).toString(),
            &request.targetDatasetId, &error)) {
        datasetSplitTaskIdV2_ = {};
        fail(QStringLiteral("Dataset Split V2 请求缺少项目、源四重身份或目标 Dataset 身份：%1").arg(error));
        return;
    }

    datasetSplitWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!datasetSplitWorkspaceV2_->open(projectRoot, &error)) {
        datasetSplitWorkspaceV2_.reset();
        datasetSplitTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Dataset Split V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!datasetSplitWorkspaceV2_->startTask(datasetSplitTaskIdV2_,
            QStringLiteral("dataset.split.v2"), QStringLiteral("dataset_split"),
            &task, &error)) {
        datasetSplitWorkspaceV2_.reset();
        datasetSplitTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Dataset Split V2 根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetSplitRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Split V2 正在核对源快照并生成不可变计划。")}});

    aitrain::v2::DatasetSplitWorkflowResultV2 result;
    const bool executed = datasetSplitWorkspaceV2_->runDatasetSplitWorkflow(
        datasetSplitTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    datasetSplitRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (datasetSplitWorkspaceV2_->task(datasetSplitTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            aitrain::v2::Failure failure;
            failure.code = aitrain::v2::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Dataset Split V2 执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查源四重身份与 V2 工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetSplitWorkspaceV2_->finalizeTask(datasetSplitTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        datasetSplitWorkspaceV2_.reset();
        datasetSplitTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Split V2 执行失败：%1").arg(error),
            QStringLiteral("dataset_split_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
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
            aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Split V2 已登记目标快照。"));
    }
    send(wp::event::datasetSplitWorkflowV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetSplitWorkspaceV2_.reset();
    datasetSplitTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_split_v2_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Split V2 completed")}});
        finishSession();
    }
}
