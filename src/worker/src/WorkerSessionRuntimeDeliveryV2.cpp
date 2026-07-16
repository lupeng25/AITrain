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

QJsonArray encodeSteps(const QVector<aitrain::v2::WorkflowStepSnapshotV2>& values)
{
    QJsonArray steps;
    for (const aitrain::v2::WorkflowStepSnapshotV2& value : values) {
        QJsonObject step;
        step.insert(QStringLiteral("ordinal"), value.ordinal);
        step.insert(QStringLiteral("kind"), value.kind);
        step.insert(QStringLiteral("state"), aitrain::v2::workflowStepStateToString(value.state));
        step.insert(QStringLiteral("backend"), value.backend);
        step.insert(QStringLiteral("inputArtifactId"), value.inputArtifactId.toString());
        step.insert(QStringLiteral("outputArtifactId"), value.outputArtifactId.toString());
        if (value.failure.isFailure()) {
            step.insert(QStringLiteral("failureCode"), aitrain::v2::failureCodeToString(value.failure.code));
            step.insert(QStringLiteral("message"), value.failure.message);
        }
        steps.append(step);
    }
    return steps;
}

} // namespace

void WorkerSession::runRuntimeDeliveryWorkflowV2(const QJsonObject& payload)
{
    if (running_ || runtimeDeliveryWorkspaceV2_) {
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
    if (!aitrain::v2::TaskId::parse(taskIdText, &runtimeDeliveryTaskIdV2_, &error)
        || runtimeDeliveryTaskIdV2_ != controlTaskId_) {
        fail(QStringLiteral("Runtime Delivery Workflow 的 taskId 无效或与 Protocol V2 控制身份不一致。"));
        return;
    }
    aitrain::v2::ModelPackageId modelPackageId;
    if (!aitrain::v2::ModelPackageId::parse(modelPackageIdText, &modelPackageId, &error)
        || projectRoot.isEmpty() || runtimeRoute.isEmpty() || sampleImagePath.isEmpty()
        || !QFileInfo(projectRoot).isDir() || !QFileInfo(sampleImagePath).isFile()) {
        fail(QStringLiteral("Runtime Delivery Workflow 需要有效项目、ModelPackageId、runtime route 和常规样本图。"));
        return;
    }

    runtimeDeliveryWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!runtimeDeliveryWorkspaceV2_->open(projectRoot, &error)) {
        runtimeDeliveryWorkspaceV2_.reset();
        fail(QStringLiteral("无法打开 Runtime Delivery V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!runtimeDeliveryWorkspaceV2_->startTask(runtimeDeliveryTaskIdV2_,
            QStringLiteral("runtime.%1").arg(runtimeRoute), QStringLiteral("runtime_delivery"), &task, &error)) {
        runtimeDeliveryWorkspaceV2_.reset();
        fail(QStringLiteral("无法启动 Runtime Delivery V2 根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    runtimeDeliveryRunningV2_ = true;
    QJsonObject started;
    started.insert(wp::field::taskId(), taskIdText);
    started.insert(QStringLiteral("percent"), 0);
    started.insert(QStringLiteral("steps"), initialSteps());
    started.insert(wp::field::message(), QStringLiteral(
        "Runtime Delivery 六步工作流已启动。ONNX Runtime 单次同步 infer 进入后不可中断，取消将在该次调用返回后收口。"));
    send(wp::event::progress(), started);

    aitrain::v2::RuntimeDeliveryWorkflowRequestV2 request;
    request.modelPackageId = modelPackageId;
    request.runtimeRoute = runtimeRoute;
    request.sampleImagePath = sampleImagePath;
    request.options = options;
    aitrain::v2::RuntimeDeliveryWorkflowResultV2 result;
    const bool executed = runtimeDeliveryWorkspaceV2_->runRuntimeDeliveryWorkflow(
        runtimeDeliveryTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    runtimeDeliveryRunningV2_ = false;

    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (runtimeDeliveryWorkspaceV2_->task(runtimeDeliveryTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            const aitrain::v2::Failure failure{aitrain::v2::FailureCode::InternalError,
                error.isEmpty() ? QStringLiteral("Runtime Delivery Workflow 启动或持久化失败。") : error,
                QStringLiteral("检查项目、模型包和 Artifact Store 后重试。"), QDateTime::currentDateTimeUtc()};
            runtimeDeliveryWorkspaceV2_->finalizeTask(runtimeDeliveryTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        runtimeDeliveryWorkspaceV2_.reset();
        runtimeDeliveryTaskIdV2_ = {};
        failWithDetails(QStringLiteral("Runtime Delivery Workflow 执行失败：%1").arg(error),
            QStringLiteral("runtime_delivery_start_failed"));
        return;
    }

    QJsonObject response;
    response.insert(wp::field::taskId(), taskIdText);
    response.insert(QStringLiteral("workflowRunId"), result.workflowRunId.toString());
    response.insert(QStringLiteral("state"), aitrain::v2::workflowStepStateToString(result.state));
    response.insert(QStringLiteral("runtimeStatusObserved"), result.runtimeStatusObserved);
    response.insert(QStringLiteral("runtimeStatus"), result.runtimeStatusObserved
            ? aitrain::v2::runtimeStatusV2ToString(result.runtimeStatus) : QStringLiteral("not_probed"));
    response.insert(QStringLiteral("finalOutputArtifactId"), result.finalOutputArtifactId.toString());
    response.insert(QStringLiteral("evidenceArtifactId"), result.evidence.artifactId.toString());
    response.insert(QStringLiteral("steps"), encodeSteps(
        runtimeDeliveryWorkspaceV2_->workflowSteps(result.workflowRunId, nullptr)));
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"), aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Runtime Delivery Workflow 完成。"));
    }
    send(wp::event::runtimeDeliveryWorkflowV2(), response);

    runtimeDeliveryWorkspaceV2_.reset();
    runtimeDeliveryTaskIdV2_ = {};
    running_ = false;
    if (result.state == aitrain::v2::WorkflowStepState::Canceled) {
        sendCanceledAndFinish(taskIdText, result.failure.message);
        return;
    }
    if (result.state == aitrain::v2::WorkflowStepState::Failed) {
        failWithDetails(result.failure.message,
            aitrain::v2::failureCodeToString(result.failure.code), response);
        return;
    }
    QJsonObject completed;
    completed.insert(wp::field::taskId(), taskIdText);
    completed.insert(wp::field::message(), QStringLiteral("Runtime Delivery Workflow completed"));
    send(wp::event::completed(), completed);
    finishSession();
}
