#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/runtime/OnnxRuntimeAdapter.h"
#include "aitrain/runtime/RuntimeInvocation.h"
#include "aitrain/workflow/TrainingWorkflowProfile.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSet>
#include <QTimer>
#include <QSaveFile>

namespace wp = aitrain::worker_protocol;
using namespace worker_support;

namespace {

QString defaultTrainersRoot()
{
    const QDir application(QCoreApplication::applicationDirPath());
    const QString packaged = application.filePath(QStringLiteral("python_trainers"));
    if (QFileInfo(packaged).isDir()) {
        return packaged;
    }
    const QString source = QDir::current().filePath(QStringLiteral("python_trainers"));
    return QFileInfo(source).isDir() ? source : QString();
}

aitrain::Failure workflowFailure(aitrain::FailureCode code, const QString& message)
{
    const QString suggestedAction = code == aitrain::FailureCode::Canceled
        ? QStringLiteral("如需继续，请重新启动训练任务。")
        : QStringLiteral("检查失败详情、输入数据和运行环境，修复后重新执行。");
    return {code, message, suggestedAction, QDateTime::currentDateTimeUtc()};
}

const QSet<QString>& allowedTrainingParameterKeys()
{
    static const QSet<QString> keys{
        QStringLiteral("epochs"), QStringLiteral("batchSize"), QStringLiteral("imageSize"),
        QStringLiteral("gridSize"), QStringLiteral("seed"), QStringLiteral("horizontalFlip"),
        QStringLiteral("colorJitter"), QStringLiteral("trainingBackend"),
        QStringLiteral("ultralyticsTrainArgs"), QStringLiteral("ultralyticsExportArgs"),
        QStringLiteral("device"), QStringLiteral("workers"), QStringLiteral("learningRate"),
        QStringLiteral("optimizer"), QStringLiteral("loss"), QStringLiteral("encoderWeights"),
        QStringLiteral("ignoreIndex"), QStringLiteral("modelFamily"), QStringLiteral("taskType"),
        QStringLiteral("exportOnnx"), QStringLiteral("thresholdStrategy"), QStringLiteral("quantile"),
        QStringLiteral("backbone"), QStringLiteral("layers"), QStringLiteral("coresetSamplingRatio"),
        QStringLiteral("numNeighbors"), QStringLiteral("modelSize"), QStringLiteral("lr"),
        QStringLiteral("weightDecay"), QStringLiteral("runtime"), QStringLiteral("exportFormats"),
        QStringLiteral("runOfficial"), QStringLiteral("prepareOnly"), QStringLiteral("trainingTemplate"),
        QStringLiteral("modelPreset"), QStringLiteral("model"), QStringLiteral("cancellationGraceMs"),
        QStringLiteral("runtimeOptions"), QStringLiteral("exportFormat")
    };
    return keys;
}

bool isForbiddenTrainingPathKey(const QString& key)
{
    const QString normalized = key.trimmed().toLower();
    static const QSet<QString> forbidden{
        QStringLiteral("datasetpath"), QStringLiteral("datasetformat"),
        QStringLiteral("sampleimagepath"), QStringLiteral("pythonexecutable"),
        QStringLiteral("trainersroot"), QStringLiteral("resumecheckpointpath"),
        QStringLiteral("imagenetdir"), QStringLiteral("pythonpathprepend"),
        QStringLiteral("paddleocrrepopath"), QStringLiteral("datasetsnapshotmanifest"),
        QStringLiteral("projectroot"), QStringLiteral("outputpath"),
        QStringLiteral("modelpath"), QStringLiteral("checkpointpath"),
        QStringLiteral("trainingpreflight")
    };
    return forbidden.contains(normalized)
        || normalized.endsWith(QStringLiteral("path"))
        || normalized.endsWith(QStringLiteral("root"))
        || normalized.endsWith(QStringLiteral("directory"))
        || normalized.endsWith(QStringLiteral("executable"));
}

bool validateTrainingParameterValue(const QJsonValue& value, const QString& location, QString* error)
{
    if (value.isString()) {
        const QString text = value.toString();
        if (text.contains(QLatin1Char('/')) || text.contains(QLatin1Char('\\'))
            || text.startsWith(QStringLiteral("file:"), Qt::CaseInsensitive)) {
            if (error) *error = QStringLiteral("训练参数 %1 不允许携带文件系统路径。").arg(location);
            return false;
        }
        return true;
    }
    if (value.isArray()) {
        const QJsonArray values = value.toArray();
        for (int index = 0; index < values.size(); ++index) {
            if (!validateTrainingParameterValue(values.at(index),
                    QStringLiteral("%1[%2]").arg(location).arg(index), error)) {
                return false;
            }
        }
        return true;
    }
    if (!value.isObject()) {
        return true;
    }
    const QJsonObject object = value.toObject();
    for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
        const QString nestedLocation = location.isEmpty()
            ? it.key() : QStringLiteral("%1.%2").arg(location, it.key());
        if (isForbiddenTrainingPathKey(it.key())) {
            if (error) *error = QStringLiteral("训练参数 %1 是禁止的原始路径字段。").arg(nestedLocation);
            return false;
        }
        if (!validateTrainingParameterValue(it.value(), nestedLocation, error)) {
            return false;
        }
    }
    return true;
}

bool validateTrainingParameters(const QJsonObject& parameters, QString* error)
{
    for (auto it = parameters.constBegin(); it != parameters.constEnd(); ++it) {
        if (!allowedTrainingParameterKeys().contains(it.key())) {
            if (error) *error = QStringLiteral("训练参数不在允许列表中：%1").arg(it.key());
            return false;
        }
        if (isForbiddenTrainingPathKey(it.key())) {
            if (error) *error = QStringLiteral("训练参数 %1 是禁止的原始路径字段。").arg(it.key());
            return false;
        }
        if (!validateTrainingParameterValue(it.value(), it.key(), error)) {
            return false;
        }
    }
    return true;
}

} // namespace

void WorkerSession::runTrainingWorkflow(const wp::TrainingCommand& command)
{
    if (running_ || trainingWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发启动  训练 Workflow。"));
        return;
    }
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QString capabilityId = command.capabilityId.trimmed();
    const QString taskType = command.taskType.trimmed();
    const QString trainingBackend = command.trainingBackend.trimmed();
    const QJsonObject parameters = command.parameters;
    const QString deploymentSampleRelativePath = QDir::fromNativeSeparators(
        command.deploymentSampleRelativePath.trimmed());
    QString error;
    aitrain::TrainingWorkflowProfile profile;
    aitrain::TrainingWorkflowRequest workflowRequest;
    if (taskIdText.isEmpty() || projectRoot.isEmpty() || capabilityId.isEmpty()
        || taskType.isEmpty() || trainingBackend.isEmpty()) {
        fail(QStringLiteral("runTrainingWorkflow 需要有效 taskId、项目、完整 Snapshot 身份、能力、任务类型和已注册训练后端。"));
        return;
    }
    const QString normalizedDeploymentSample = QDir::cleanPath(deploymentSampleRelativePath);
    if (!deploymentSampleRelativePath.isEmpty()
        && (!QDir::isRelativePath(deploymentSampleRelativePath)
            || normalizedDeploymentSample == QStringLiteral("..")
            || normalizedDeploymentSample.startsWith(QStringLiteral("../")))) {
        fail(QStringLiteral("runTrainingWorkflow 的 deploymentSampleRelativePath 必须是 Snapshot Artifact 内部相对路径。"));
        return;
    }
    if (!aitrain::resolveTrainingWorkflowProfile(trainingBackend, &profile, &error)
        || profile.capabilityTaskType != taskType) {
        fail(QStringLiteral("runTrainingWorkflow 请求与训练 Workflow Profile 不一致：%1")
            .arg(error.isEmpty() ? QStringLiteral("taskType 不匹配") : error));
        return;
    }
    if (!aitrain::BuiltinCapabilityRegistry::instance().supports(
            capabilityId, taskType, profile.datasetFormat, trainingBackend, &error)) {
        fail(QStringLiteral("runTrainingWorkflow 请求与内置能力矩阵不一致：%1").arg(error));
        return;
    }
    if (!aitrain::TaskId::parse(taskIdText, &trainingWorkflowTaskId_, &error)
        || trainingWorkflowTaskId_ != controlTaskId_
        || !aitrain::DatasetId::parse(command.datasetId,
            &workflowRequest.datasetId, &error)
        || !aitrain::DatasetVersionId::parse(command.datasetVersionId,
            &workflowRequest.datasetVersionId, &error)
        || !aitrain::SnapshotId::parse(command.snapshotId,
            &workflowRequest.snapshotId, &error)
        || !aitrain::ArtifactId::parse(command.snapshotArtifactId,
            &workflowRequest.snapshotArtifactId, &error)) {
        trainingWorkflowTaskId_ = {};
        fail(QStringLiteral("runTrainingWorkflow 要求控制任务一致且 Dataset/Version/Snapshot/Artifact 身份完整：%1").arg(error));
        return;
    }
    if (!validateTrainingParameters(parameters, &error)) {
        trainingWorkflowTaskId_ = {};
        fail(QStringLiteral("runTrainingWorkflow 参数合同无效：%1").arg(error));
        return;
    }
    if (!QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("runTrainingWorkflow 的项目目录不存在。"));
        return;
    }
    const QString pythonProgram = firstUsablePythonExecutable();
    const QString trainersRoot = defaultTrainersRoot();
    if (pythonProgram.isEmpty() || trainersRoot.isEmpty()) {
        fail(QStringLiteral(" 训练 Workflow 需要可用 Python 和 python_trainers 目录。"));
        return;
    }

    trainingWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    trainingWorkflowDeploymentSampleRelativePath_ = deploymentSampleRelativePath.isEmpty()
        ? QString() : normalizedDeploymentSample;
    trainingWorkflowAdapterConfig_.pythonProgram = pythonProgram;
    trainingWorkflowAdapterConfig_.trainersRoot = trainersRoot;
    // Core 将基于已验证 Snapshot Artifact inventory 解析该相对路径；在接口完成
    // 身份化前仅透传相对值，Worker 不再接受或解析宿主机样本绝对路径。
    trainingWorkflowAdapterConfig_.deploymentSampleRelativePath = trainingWorkflowDeploymentSampleRelativePath_;
    trainingWorkflowAdapterConfig_.environment = QProcessEnvironment::systemEnvironment();
    trainingWorkflowAdapterConfig_.environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    trainingWorkflowAdapterConfig_.environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    configurePackagedPythonEnvironment(&trainingWorkflowAdapterConfig_.environment);
    trainingWorkflowAdapterConfig_.cancellationGraceMs = qMax(1000,
        parameters.value(QStringLiteral("cancellationGraceMs")).toInt(5000));
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral(" 训练 Workflow：正在打开项目工作区。")} });
    if (!trainingWorkspace_->open(projectRoot, &error)) {
        trainingWorkspace_.reset();
        fail(QStringLiteral("无法打开  项目工作区：%1").arg(error));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral(" 训练 Workflow：项目工作区已打开，正在创建根任务。")} });

    aitrain::TaskSnapshot task;
    if (!trainingWorkspace_->startTask(trainingWorkflowTaskId_, capabilityId, taskType, &task, &error)) {
        trainingWorkspace_.reset();
        fail(QStringLiteral("无法创建  训练根任务：%1").arg(error));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral(" 训练 Workflow：根任务已创建。")} });

    workflowRequest.templateId = profile.templateId;
    workflowRequest.trainingBackend = profile.trainingBackend;
    workflowRequest.evaluationBackend = profile.evaluationBackend;
    workflowRequest.exportBackend = profile.exportBackend;
    workflowRequest.deploymentBackend = profile.deploymentBackend;
    workflowRequest.parameterSummary = parameters;
    workflowRequest.parameterSummary.insert(QStringLiteral("trainingBackend"), trainingBackend);
    workflowRequest.requireEvidenceBeforeTerminal = true;
    aitrain::TrainingWorkflowDispatch dispatch;
    if (!trainingWorkspace_->beginTrainingWorkflow(trainingWorkflowTaskId_, workflowRequest, &dispatch, &error)) {
        QString ignored;
        trainingWorkspace_->finalizeTask(trainingWorkflowTaskId_, aitrain::TaskState::Failed,
            workflowFailure(aitrain::FailureCode::InternalError,
                QStringLiteral("无法创建  训练 Workflow：%1").arg(error)), &ignored);
        trainingWorkspace_.reset();
        failWithDetails(QStringLiteral("无法创建  训练 Workflow：%1").arg(error),
            aitrain::failureCodeToString(aitrain::FailureCode::InternalError));
        return;
    }
    trainingWorkflowRunId_ = dispatch.workflowRunId;
    if (!dispatch.dispatch.hasStep || dispatch.dispatch.step.kind != QStringLiteral("Train")) {
        QString ignored;
        trainingWorkspace_->finalizeTask(trainingWorkflowTaskId_, aitrain::TaskState::Failed,
            workflowFailure(aitrain::FailureCode::InternalError,
                QStringLiteral("Snapshot 身份校验后未能直接派发 Train。")), &ignored);
        trainingWorkspace_.reset();
        failWithDetails(QStringLiteral(" 训练 Workflow 未从已登记 Snapshot 直接进入 Train。"),
            aitrain::failureCodeToString(aitrain::FailureCode::InternalError));
        return;
    }
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 3},
        {wp::field::message(), QStringLiteral(" 训练 Workflow：Snapshot 身份与完整性已校验，开始训练。")}});
    dispatchTrainingWorkflow(dispatch);
}

void WorkerSession::dispatchTrainingWorkflow(const aitrain::TrainingWorkflowDispatch& dispatch)
{
    if (!trainingWorkspace_ || finishingSession_) {
        return;
    }
    if (!dispatch.dispatch.hasStep) {
        finishTrainingWorkflow(dispatch);
        return;
    }
    const aitrain::WorkflowStepSnapshot& step = dispatch.dispatch.step;
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
        {wp::field::message(), QStringLiteral(" Workflow 开始步骤 %1：%2（%3）。")
            .arg(step.ordinal + 1).arg(step.kind, step.backend)}});
    if (canceled_) {
        aitrain::TrainingWorkflowDispatch completed;
        QString error;
        const aitrain::WorkflowStepExecutionResult execution{
            aitrain::WorkflowStepState::Canceled, {},
            workflowFailure(aitrain::FailureCode::Canceled, QStringLiteral("用户取消  训练 Workflow。"))};
        if (!trainingWorkspace_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &error)) {
            failWithDetails(QStringLiteral("无法收口已取消的  Workflow 步骤：%1").arg(error), QStringLiteral("workflow_cancel_failed"));
            return;
        }
        dispatchTrainingWorkflow(completed);
        return;
    }
    aitrain::TrainingWorkflowProfile profile;
    const bool hasProfile = aitrain::resolveTrainingWorkflowProfile(
        step.parameterSummary.value(QStringLiteral("trainingBackend")).toString(), &profile, nullptr);
    const auto routeIt = hasProfile
        ? std::find_if(profile.steps.cbegin(), profile.steps.cend(), [&step](const auto& route) {
            return route.kind == step.kind;
        })
        : profile.steps.cend();
    const bool usesAdapter = routeIt != profile.steps.cend() && !routeIt->script.isEmpty();
    if (usesAdapter) {
        aitrain::TrainingWorkflowAdapterLaunch launch;
        QString error;
        if (!trainingWorkspace_->prepareTrainingWorkflowAdapterLaunch(dispatch.workflowRunId, step.id,
                trainingWorkflowAdapterConfig_, &launch, &error)
            || !trainingWorkspace_->startTrainingWorkflowAdapterStep(dispatch.workflowRunId, step.id, launch.launch,
                [this](const aitrain::TrainingWorkflowDispatch& next) {
                    // 只有 Adapter 终态已经提交 Artifact、下一步骤已经引用该不可变
                    // Artifact 后，才向 GUI 暴露文件路径；禁止转发暂存区 candidate。
                    if (next.dispatch.hasStep) {
                        aitrain::VerifiedTrainingWorkflowInput committed;
                        QString resolutionError;
                        if (trainingWorkspace_->resolveTrainingWorkflowStepInput(next.workflowRunId,
                                next.dispatch.step.id, &committed, &resolutionError)) {
                            for (const aitrain::VerifiedWorkflowArtifactFile& file : committed.files) {
                                const QString kind = file.relativePath.section(QLatin1Char('/'), 0, 0);
                                send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                                    {QStringLiteral("kind"), kind}, {QStringLiteral("artifactId"), committed.artifactId.toString()},
                                    {QStringLiteral("relativePath"), file.relativePath},
                                    {wp::field::message(), QStringLiteral("训练 Workflow Adapter Artifact 已原子提交并验证。")}});
                            }
                        }
                    }
                    // Adapter settled 回调仍位于 PythonAdapterHost 的调用栈中。
                    // 排队派发，避免下一 Adapter 重入 start() 或终态路径销毁当前 Host。
                    QTimer::singleShot(0, this, [this, next]() { dispatchTrainingWorkflow(next); });
                },
                &error, [this](const aitrain::ProtocolEnvelope& event) { forwardTrainingWorkflowAdapterEvent(event); })) {
            aitrain::TrainingWorkflowDispatch completed;
            const aitrain::WorkflowStepExecutionResult execution{
                aitrain::WorkflowStepState::Failed, {}, workflowFailure(aitrain::FailureCode::ProcessCrashed,
                    QStringLiteral("无法启动  训练 Workflow %1 Adapter：%2").arg(step.kind, error))};
            QString completionError;
            if (trainingWorkspace_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &completionError)) {
                dispatchTrainingWorkflow(completed);
            } else {
                failWithDetails(QStringLiteral(" Workflow Adapter 启动失败且无法收口步骤：%1；%2").arg(error, completionError),
                    QStringLiteral("workflow_adapter_start_failed"));
            }
        }
        return;
    }
    QTimer::singleShot(0, this, [this, dispatch]() { runTrainingWorkflowLocalStep(dispatch); });
}

void WorkerSession::runTrainingWorkflowLocalStep(const aitrain::TrainingWorkflowDispatch& dispatch)
{
    if (!trainingWorkspace_ || !dispatch.dispatch.hasStep || finishingSession_) {
        return;
    }
    const aitrain::WorkflowStepSnapshot& step = dispatch.dispatch.step;
    aitrain::WorkflowStepExecutionResult execution;
    QString error;
    if (canceled_) {
        execution.state = aitrain::WorkflowStepState::Canceled;
        execution.failure = workflowFailure(aitrain::FailureCode::Canceled, QStringLiteral("用户取消  训练 Workflow。"));
    } else if (step.kind == QStringLiteral("DeploymentValidate")) {
        aitrain::TrainingDeploymentInvocation prepared;
        aitrain::RuntimeInvocation invocation;
        if (!trainingWorkspace_->prepareTrainingWorkflowDeploymentInvocation(dispatch.workflowRunId, step.id,
                trainingWorkflowDeploymentSampleRelativePath_, &prepared, &error)
            || !aitrain::decodeRuntimeInvocation(prepared.invocation, &invocation, &error)) {
            execution.failure = workflowFailure(aitrain::FailureCode::ArtifactIncomplete,
                QStringLiteral("无法准备  部署验证：%1").arg(error));
        } else {
            aitrain::OnnxRuntimeAdapter adapter;
            const aitrain::RuntimeOperationResult runtime = adapter.deploymentValidate(invocation.model,
                QJsonObject{{QStringLiteral("imagePath"), invocation.imagePath}, {QStringLiteral("outputPath"), invocation.outputPath},
                    {QStringLiteral("options"), invocation.options}});
            if (canceled_) {
                execution.state = aitrain::WorkflowStepState::Canceled;
                execution.failure = workflowFailure(aitrain::FailureCode::Canceled, QStringLiteral("用户在部署验证期间取消  训练 Workflow。"));
            } else if (runtime.status != aitrain::RuntimeStatus::Available) {
                execution.failure = workflowFailure(aitrain::FailureCode::ArtifactIncompatible,
                    QStringLiteral(" ONNX Runtime 部署验证失败（%1）：%2")
                        .arg(aitrain::runtimeStatusToString(runtime.status), runtime.message));
            } else {
                const QString predictionsPath = runtime.details.value(QStringLiteral("predictionsPath")).toString();
                const QString overlayPath = runtime.details.value(QStringLiteral("overlayPath")).toString();
                const QString reportPath = QDir(invocation.outputPath).filePath(QStringLiteral("deployment_validation_report.json"));
                QSaveFile report(reportPath);
                const QJsonObject contents{{QStringLiteral("schemaVersion"), 2}, {QStringLiteral("status"), QStringLiteral("passed")},
                    {QStringLiteral("runtime"), invocation.runtimeRoute}, {QStringLiteral("sourceArtifactId"), prepared.sourceArtifactId.toString()},
                    {QStringLiteral("imagePath"), invocation.imagePath}, {QStringLiteral("predictionsPath"), predictionsPath},
                    {QStringLiteral("overlayPath"), overlayPath}, {QStringLiteral("details"), runtime.details}};
                if (predictionsPath.isEmpty() || overlayPath.isEmpty() || !report.open(QIODevice::WriteOnly)
                    || report.write(QJsonDocument(contents).toJson(QJsonDocument::Indented)) < 0 || !report.commit()) {
                    execution.failure = workflowFailure(aitrain::FailureCode::ArtifactIncomplete,
                        QStringLiteral("无法写入  部署验证报告或运行时产物不完整。"));
                } else {
                    aitrain::RuntimeArtifactBundle bundle;
                    const QVector<aitrain::RuntimeArtifactCandidate> candidates{{QStringLiteral("deployment_validation_report"), reportPath},
                        {QStringLiteral("deployment_predictions"), predictionsPath}, {QStringLiteral("deployment_overlay"), overlayPath}};
                    if (!trainingWorkspace_->commitRuntimeArtifacts(trainingWorkflowTaskId_, QStringLiteral("deployment_validation"),
                            candidates, &bundle, &error)) {
                        execution.failure = workflowFailure(aitrain::FailureCode::ArtifactIncomplete,
                            QStringLiteral("无法提交  部署验证 Artifact：%1").arg(error));
                    } else {
                        execution.state = aitrain::WorkflowStepState::Succeeded;
                        execution.outputArtifactId = bundle.artifactId;
                        for (auto it = bundle.pathsByKind.cbegin(); it != bundle.pathsByKind.cend(); ++it) {
                            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), it.key()},
                                {QStringLiteral("artifactId"), bundle.artifactId.toString()},
                                {QStringLiteral("relativePath"), QDir(QDir(trainingWorkspace_->workspacePath())
                                    .filePath(QStringLiteral("artifacts/artifacts/%1").arg(bundle.artifactId.toString())))
                                    .relativeFilePath(it.value())},
                                {wp::field::message(), QStringLiteral(" 部署验证 Artifact。")}});
                        }
                    }
                }
            }
        }
    } else if (step.kind == QStringLiteral("RegisterModel")) {
        aitrain::TrainingModelRegistration registered;
        if (!trainingWorkspace_->registerTrainingWorkflowModel(dispatch.workflowRunId, step.id, &registered, &error)) {
            execution.failure = workflowFailure(aitrain::FailureCode::ArtifactIncomplete,
                QStringLiteral("无法登记官方 YOLO 模型包：%1").arg(error));
        } else {
            execution.state = aitrain::WorkflowStepState::Succeeded;
            execution.outputArtifactId = registered.registrationArtifact.artifactId;
            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), QStringLiteral("model_manifest")},
                {QStringLiteral("artifactId"), registered.registrationArtifact.artifactId.toString()},
                {QStringLiteral("relativePath"), QStringLiteral("model_manifest.json")},
                {wp::field::message(), QStringLiteral("已登记  模型包。")}});
        }
    } else if (step.kind == QStringLiteral("RenderDeliveryReport")) {
        aitrain::RuntimeArtifactBundle report;
        if (!trainingWorkspace_->renderTrainingWorkflowDeliveryReport(dispatch.workflowRunId, step.id, &report, &error)) {
            execution.failure = workflowFailure(aitrain::FailureCode::InternalError,
                QStringLiteral("无法渲染  训练交付报告：%1").arg(error));
        } else {
            execution.state = aitrain::WorkflowStepState::Succeeded;
            execution.outputArtifactId = report.artifactId;
            for (auto it = report.pathsByKind.cbegin(); it != report.pathsByKind.cend(); ++it) {
                send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), it.key()},
                    {QStringLiteral("artifactId"), report.artifactId.toString()},
                    {QStringLiteral("relativePath"), QDir(QDir(trainingWorkspace_->workspacePath())
                        .filePath(QStringLiteral("artifacts/artifacts/%1").arg(report.artifactId.toString())))
                        .relativeFilePath(it.value())},
                    {wp::field::message(), QStringLiteral(" 训练交付报告 Artifact。")}});
            }
        }
    } else {
        execution.failure = workflowFailure(aitrain::FailureCode::BackendUnsupported,
            QStringLiteral(" 训练 Workflow 不支持步骤：%1").arg(step.kind));
    }

    if (execution.state != aitrain::WorkflowStepState::Succeeded
        && execution.state != aitrain::WorkflowStepState::Canceled) {
        execution.state = aitrain::WorkflowStepState::Failed;
    }
    aitrain::TrainingWorkflowDispatch completed;
    if (!trainingWorkspace_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &error)) {
        failWithDetails(QStringLiteral("无法收口  Workflow 步骤 %1：%2").arg(step.kind, error),
            QStringLiteral("workflow_step_complete_failed"));
        return;
    }
    dispatchTrainingWorkflow(completed);
}

void WorkerSession::forwardTrainingWorkflowAdapterEvent(const aitrain::ProtocolEnvelope& event)
{
    QJsonObject payload = event.payload;
    payload.insert(wp::field::taskId(), activeTaskId_);
    payload.insert(QStringLiteral("workflowEventKind"), event.kind);
    if (event.kind == QStringLiteral("event.log")) {
        send(wp::event::log(), payload);
    } else if (event.kind == QStringLiteral("event.progress")) {
        send(wp::event::progress(), payload);
    } else if (event.kind == QStringLiteral("event.metric")) {
        send(wp::event::metric(), payload);
    } else if (event.kind == QStringLiteral("event.artifact_candidate")) {
        payload.insert(wp::field::message(), QStringLiteral("官方 Adapter 已验证候选产物，等待步骤原子提交。"));
        payload.remove(QStringLiteral("path"));
        send(wp::event::log(), payload);
    } else if (event.kind == QStringLiteral("event.failed")) {
        const QJsonObject details = payload.value(QStringLiteral("details")).toObject();
        QString message = payload.value(wp::field::message()).toString();
        if (!details.isEmpty()) {
            message.append(QStringLiteral(" 详细信息：%1")
                .arg(QString::fromUtf8(QJsonDocument(details).toJson(QJsonDocument::Compact))));
        }
        payload.insert(wp::field::message(), message);
        send(wp::event::log(), payload);
    }
}

void WorkerSession::cancelTrainingWorkflow()
{
    canceled_ = true;
    QString error;
    if (!trainingWorkspace_->requestTaskCancellation(trainingWorkflowTaskId_, &error)) {
        failWithDetails(QStringLiteral("无法请求取消  训练 Workflow：%1").arg(error), QStringLiteral("workflow_cancel_failed"));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
        {wp::field::message(), QStringLiteral("已请求取消  训练 Workflow，等待当前步骤清理并收口。")}});
}

void WorkerSession::finishTrainingWorkflow(const aitrain::TrainingWorkflowDispatch& dispatch)
{
    if (finishingSession_) {
        return;
    }
    const aitrain::WorkflowRunExecutionResult& result = dispatch.dispatch.result;
    bool evidenceCommitted = false;
    QString evidenceError;
    if (trainingWorkspace_) {
        aitrain::EvidenceBundle evidence;
        aitrain::EvidenceArtifactBundle committed;
        if (trainingWorkspace_->buildWorkflowEvidenceBundle(dispatch.workflowRunId, &evidence, &evidenceError)
            && trainingWorkspace_->commitEvidenceBundle(evidence, &committed, &evidenceError)) {
            evidenceCommitted = true;
            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                {QStringLiteral("kind"), QStringLiteral("evidence_bundle")},
                {QStringLiteral("artifactId"), committed.artifactId.toString()},
                {QStringLiteral("relativePath"), QStringLiteral("evidence.json")},
                {wp::field::message(), QStringLiteral(" Evidence Bundle Artifact。")}});
        } else {
            send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                {wp::field::message(), QStringLiteral("Evidence Bundle 提交失败，根任务不会发布成功终态：%1").arg(evidenceError)}});
        }
    }
    if (!evidenceCommitted && trainingWorkspace_) {
        const aitrain::Failure failure = workflowFailure(aitrain::FailureCode::ArtifactIncomplete,
            QStringLiteral("Evidence Bundle 提交失败：%1").arg(evidenceError));
        QString recordError;
        trainingWorkspace_->recordWorkflowEvidenceFailure(dispatch.workflowRunId, failure, &recordError);
        const QString message = recordError.isEmpty() ? failure.message
            : QStringLiteral("%1；记录 Evidence 失败尝试也失败：%2").arg(failure.message, recordError);
        trainingWorkspace_.reset();
        failWithDetails(message, aitrain::failureCodeToString(failure.code));
        return;
    }
    if (trainingWorkspace_) {
        QString finalizationError;
        if (!trainingWorkspace_->closeWorkflowTerminalization(dispatch.workflowRunId,
                &finalizationError)) {
            trainingWorkspace_.reset();
            failWithDetails(QStringLiteral("Evidence 已提交，但根任务终态持久化失败：%1").arg(finalizationError),
                QStringLiteral("terminal_persistence_failed"));
            return;
        }
    }
    if (result.state == aitrain::WorkflowStepState::Succeeded && trainingWorkspace_) {
        send(wp::event::progress(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("percent"), 100},
            {wp::field::message(), QStringLiteral(" 官方 YOLO 训练、评估、导出、部署验证、模型登记与交付报告已完成。")}});
        running_ = false;
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), activeTaskId_},
            {wp::field::message(), QStringLiteral(" training workflow completed")}});
        trainingWorkspace_.reset();
        finishSession();
        return;
    }
    const QString message = result.failure.message.isEmpty()
        ? QStringLiteral(" 训练 Workflow 未成功完成。") : result.failure.message;
    trainingWorkspace_.reset();
    if (result.state == aitrain::WorkflowStepState::Canceled) {
        sendCanceledAndFinish(activeTaskId_, message);
    } else {
        failWithDetails(message, aitrain::failureCodeToString(result.failure.code));
    }
}
