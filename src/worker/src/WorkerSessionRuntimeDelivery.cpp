#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QStringList>

namespace wp = aitrain::worker_protocol;

namespace {

const QStringList kRuntimeDeliveryStepKinds{
    QStringLiteral("ImportOrResolveModel"),
    QStringLiteral("ValidateManifest"),
    QStringLiteral("RunInferenceSmoke"),
    QStringLiteral("Benchmark"),
    QStringLiteral("DeploymentValidate"),
    QStringLiteral("RenderDeliveryReport")};

QJsonArray initialSteps()
{
    QJsonArray steps;
    for (int ordinal = 0; ordinal < kRuntimeDeliveryStepKinds.size(); ++ordinal) {
        QJsonObject step;
        step.insert(QStringLiteral("ordinal"), ordinal);
        step.insert(QStringLiteral("kind"), kRuntimeDeliveryStepKinds.at(ordinal));
        step.insert(QStringLiteral("state"), QStringLiteral("pending"));
        steps.append(step);
    }
    return steps;
}

QJsonArray encodeSteps(const QVector<aitrain::WorkflowStepSnapshot>& values)
{
    QJsonArray steps;
    for (const aitrain::WorkflowStepSnapshot& value : values) {
        QJsonObject step;
        step.insert(QStringLiteral("ordinal"), value.ordinal);
        step.insert(QStringLiteral("kind"), value.kind);
        step.insert(QStringLiteral("state"), aitrain::workflowStepStateToString(value.state));
        step.insert(QStringLiteral("backend"), value.backend);
        step.insert(QStringLiteral("inputArtifactId"), value.inputArtifactId.toString());
        step.insert(QStringLiteral("outputArtifactId"), value.outputArtifactId.toString());
        if (value.failure.isFailure()) {
            step.insert(QStringLiteral("failureCode"), aitrain::failureCodeToString(value.failure.code));
            step.insert(QStringLiteral("message"), value.failure.message);
        }
        steps.append(step);
    }
    return steps;
}

} // namespace

void WorkerSession::runRuntimeDeliveryWorkflow(const QJsonObject& payload)
{
    if (running_ || runtimeDeliveryWorkspace_) {
        fail(QStringLiteral("Runtime Delivery Workflow 已在运行。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QString modelPackageIdText = payload.value(QStringLiteral("modelPackageId")).toString().trimmed();
    const QString runtimeRoute = payload.value(QStringLiteral("runtimeRoute")).toString().trimmed();
    const QString sampleImagePath = payload.value(wp::field::sampleImagePath()).toString().trimmed();
    const QJsonObject options = payload.value(wp::field::options()).toObject();
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &runtimeDeliveryTaskId_, &error)
        || runtimeDeliveryTaskId_ != controlTaskId_) {
        fail(QStringLiteral("Runtime Delivery Workflow 的 taskId 无效或与 Protocol  控制身份不一致。"));
        return;
    }
    aitrain::ModelPackageId modelPackageId;
    if (!aitrain::ModelPackageId::parse(modelPackageIdText, &modelPackageId, &error)
        || projectRoot.isEmpty() || runtimeRoute.isEmpty() || sampleImagePath.isEmpty()
        || !QFileInfo(projectRoot).isDir() || !QFileInfo(sampleImagePath).isFile()) {
        fail(QStringLiteral("Runtime Delivery Workflow 需要有效项目、ModelPackageId、runtime route 和常规样本图。"));
        return;
    }

    runtimeDeliveryWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!runtimeDeliveryWorkspace_->open(projectRoot, &error)) {
        runtimeDeliveryWorkspace_.reset();
        fail(QStringLiteral("无法打开 Runtime Delivery  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!runtimeDeliveryWorkspace_->startTask(runtimeDeliveryTaskId_,
            QStringLiteral("runtime.%1").arg(runtimeRoute), QStringLiteral("runtime_delivery"), &task, &error)) {
        runtimeDeliveryWorkspace_.reset();
        fail(QStringLiteral("无法启动 Runtime Delivery  根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    runtimeDeliveryRunning_ = true;
    QJsonObject started;
    started.insert(wp::field::taskId(), taskIdText);
    started.insert(QStringLiteral("percent"), 0);
    started.insert(QStringLiteral("steps"), initialSteps());
    started.insert(wp::field::message(), QStringLiteral(
        "Runtime Delivery 六步工作流已启动。ONNX Runtime 单次同步 infer 进入后不可中断，取消将在该次调用返回后收口。"));
    send(wp::event::progress(), started);

    aitrain::RuntimeDeliveryWorkflowRequest request;
    request.modelPackageId = modelPackageId;
    request.runtimeRoute = runtimeRoute;
    request.sampleImagePath = sampleImagePath;
    request.options = options;
    aitrain::RuntimeDeliveryWorkflowResult result;
    const bool executed = runtimeDeliveryWorkspace_->runRuntimeDeliveryWorkflow(
        runtimeDeliveryTaskId_, request, &result, &error, pollingCancellationCallback(0));
    runtimeDeliveryRunning_ = false;

    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (runtimeDeliveryWorkspace_->task(runtimeDeliveryTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            const aitrain::Failure failure{aitrain::FailureCode::InternalError,
                error.isEmpty() ? QStringLiteral("Runtime Delivery Workflow 启动或持久化失败。") : error,
                QStringLiteral("检查项目、模型包和 Artifact Store 后重试。"), QDateTime::currentDateTimeUtc()};
            runtimeDeliveryWorkspace_->finalizeTask(runtimeDeliveryTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        runtimeDeliveryWorkspace_.reset();
        runtimeDeliveryTaskId_ = {};
        failWithDetails(QStringLiteral("Runtime Delivery Workflow 执行失败：%1").arg(error),
            QStringLiteral("runtime_delivery_start_failed"));
        return;
    }

    QJsonObject response;
    response.insert(wp::field::taskId(), taskIdText);
    response.insert(QStringLiteral("workflowRunId"), result.workflowRunId.toString());
    response.insert(QStringLiteral("state"), aitrain::workflowStepStateToString(result.state));
    response.insert(QStringLiteral("runtimeStatusObserved"), result.runtimeStatusObserved);
    response.insert(QStringLiteral("runtimeStatus"), result.runtimeStatusObserved
            ? aitrain::runtimeStatusToString(result.runtimeStatus) : QStringLiteral("not_probed"));
    response.insert(QStringLiteral("finalOutputArtifactId"), result.finalOutputArtifactId.toString());
    response.insert(QStringLiteral("evidenceArtifactId"), result.evidence.artifactId.toString());
    response.insert(QStringLiteral("steps"), encodeSteps(
        runtimeDeliveryWorkspace_->workflowSteps(result.workflowRunId, nullptr)));
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"), aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Runtime Delivery Workflow 完成。"));
    }
    send(wp::event::runtimeDeliveryWorkflow(), response);

    runtimeDeliveryWorkspace_.reset();
    runtimeDeliveryTaskId_ = {};
    running_ = false;
    if (result.state == aitrain::WorkflowStepState::Canceled) {
        sendCanceledAndFinish(taskIdText, result.failure.message);
        return;
    }
    if (result.state == aitrain::WorkflowStepState::Failed) {
        failWithDetails(result.failure.message,
            aitrain::failureCodeToString(result.failure.code), response);
        return;
    }
    QJsonObject completed;
    completed.insert(wp::field::taskId(), taskIdText);
    completed.insert(wp::field::message(), QStringLiteral("Runtime Delivery Workflow completed"));
    send(wp::event::completed(), completed);
    finishSession();
}
