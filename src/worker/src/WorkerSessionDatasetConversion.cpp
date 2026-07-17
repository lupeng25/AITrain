#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDatasetConversionWorkflow(const wp::DatasetConversionCommand& command)
{
    if (running_ || datasetConversionWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Dataset Conversion 。"));
        return;
    }
    const aitrain::TaskId taskId = command.context.taskId;
    const QString taskIdText = taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DatasetConversionWorkflowRequest request;
    request.sourcePath = command.sourcePath.trimmed();
    request.sourceFormat = command.sourceFormat.trimmed();
    request.targetFormat = command.targetFormat.trimmed();
    request.targetDatasetName = command.targetDatasetName.trimmed();
    request.options = command.options;
    QString error;
    if (!taskId.isValid()) {
        error = QStringLiteral("TaskId 无效。");
    }
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || request.sourcePath.isEmpty() || request.sourceFormat.isEmpty()
        || request.targetFormat.isEmpty() || request.targetDatasetName.isEmpty()
        || !aitrain::DatasetId::parse(command.targetDatasetId,
            &request.targetDatasetId, &error)) {
        datasetConversionTaskId_ = {};
        fail(QStringLiteral("Dataset Conversion  请求缺少项目、外部源、格式或目标 Dataset 身份：%1").arg(error));
        return;
    }
    datasetConversionTaskId_ = taskId;

    datasetConversionWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!datasetConversionWorkspace_->open(projectRoot, &error)) {
        datasetConversionWorkspace_.reset();
        datasetConversionTaskId_ = {};
        fail(QStringLiteral("无法打开 Dataset Conversion  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!datasetConversionWorkspace_->startTask(datasetConversionTaskId_,
            QStringLiteral("dataset.conversion"), QStringLiteral("dataset_conversion"),
            &task, &error)) {
        datasetConversionWorkspace_.reset();
        datasetConversionTaskId_ = {};
        fail(QStringLiteral("无法创建 Dataset Conversion  根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    datasetConversionRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Dataset Conversion  正在冻结外部源并生成受控快照。")}});

    aitrain::DatasetConversionWorkflowResult result;
    const bool executed = datasetConversionWorkspace_->runDatasetConversionWorkflow(
        datasetConversionTaskId_, request, &result, &error, pollingCancellationCallback(0));
    datasetConversionRunning_ = false;
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (datasetConversionWorkspace_->task(datasetConversionTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Dataset Conversion  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查外部源与  工作区后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            datasetConversionWorkspace_->finalizeTask(datasetConversionTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        datasetConversionWorkspace_.reset();
        datasetConversionTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Dataset Conversion  执行失败：%1").arg(error),
            QStringLiteral("dataset_conversion_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("datasetId"), result.datasetSnapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("conversionArtifactId"), result.conversionArtifactId.toString()},
        {QStringLiteral("snapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Dataset Conversion  已登记新快照。"));
    }
    send(wp::event::datasetConversionWorkflow(), response);

    const aitrain::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    datasetConversionWorkspace_.reset();
    datasetConversionTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("dataset_conversion_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Dataset Conversion  completed")}});
        finishSession();
    }
}
