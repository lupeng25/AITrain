#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetConversionWorkflowV2(const QJsonObject& payload)
{
    if (running_ || datasetConversionWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Conversion V2。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    aitrain::v2::DatasetConversionWorkflowRequestV2 request;
    request.sourcePath = payload.value(wp::field::sourcePath()).toString().trimmed();
    request.sourceFormat = payload.value(wp::field::sourceFormat()).toString().trimmed();
    request.targetFormat = payload.value(wp::field::targetFormat()).toString().trimmed();
    request.targetDatasetName = payload.value(QStringLiteral("targetDatasetName")).toString().trimmed();
    request.options = payload.value(wp::field::options()).toObject();
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &datasetConversionTaskIdV2_, &error)
        || datasetConversionTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.sourcePath.isEmpty() || request.sourceFormat.isEmpty()
        || request.targetFormat.isEmpty() || request.targetDatasetName.isEmpty()
        || !aitrain::v2::DatasetId::parse(payload.value(QStringLiteral("targetDatasetId")).toString(),
            &request.targetDatasetId, &error)) {
        datasetConversionTaskIdV2_ = {};
        fail(QStringLiteral("Dataset Conversion V2 请求缺少项目、外部源、格式或目标 Dataset 身份：%1").arg(error));
        return;
    }

    datasetConversionWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!datasetConversionWorkspaceV2_->open(projectRoot, &error)) {
        datasetConversionWorkspaceV2_.reset();
        datasetConversionTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Dataset Conversion V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!datasetConversionWorkspaceV2_->startTask(datasetConversionTaskIdV2_,
            QStringLiteral("dataset.conversion.v2"), QStringLiteral("dataset_conversion"),
            &task, &error)) {
        datasetConversionWorkspaceV2_.reset();
        datasetConversionTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Dataset Conversion V2 根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetConversionRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Conversion V2 正在冻结外部源并生成受控快照。")}});

    aitrain::v2::DatasetConversionWorkflowResultV2 result;
    const bool executed = datasetConversionWorkspaceV2_->runDatasetConversionWorkflow(
        datasetConversionTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    datasetConversionRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (datasetConversionWorkspaceV2_->task(datasetConversionTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            aitrain::v2::Failure failure;
            failure.code = aitrain::v2::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Dataset Conversion V2 执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查外部源与 V2 工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetConversionWorkspaceV2_->finalizeTask(datasetConversionTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        datasetConversionWorkspaceV2_.reset();
        datasetConversionTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Conversion V2 执行失败：%1").arg(error),
            QStringLiteral("dataset_conversion_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("datasetId"), result.datasetSnapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("conversionArtifactId"), result.conversionArtifactId.toString()},
        {QStringLiteral("snapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Conversion V2 已登记新快照。"));
    }
    send(wp::event::datasetConversionWorkflowV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetConversionWorkspaceV2_.reset();
    datasetConversionTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_conversion_v2_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Conversion V2 completed")}});
        finishSession();
    }
}
