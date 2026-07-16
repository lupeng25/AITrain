#include "WorkerSession.h"
#include "WorkerSessionSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/v2/OnnxRuntimeAdapterV2.h"
#include "aitrain/v2/RuntimeInvocationV2.h"
#include "aitrain/v2/TrainingWorkflowProfileV2.h"

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

aitrain::v2::Failure workflowFailure(aitrain::v2::FailureCode code, const QString& message)
{
    const QString suggestedAction = code == aitrain::v2::FailureCode::Canceled
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

void WorkerSession::runTrainingWorkflowV2(const QJsonObject& payload)
{
    if (running_ || trainingWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发启动 V2 训练 Workflow。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QString capabilityId = payload.value(QStringLiteral("capabilityId")).toString().trimmed();
    const QString taskType = payload.value(wp::field::taskType()).toString().trimmed();
    const QString trainingBackend = payload.value(QStringLiteral("trainingBackend")).toString().trimmed();
    const QJsonObject parameters = payload.value(QStringLiteral("parameters")).toObject();
    const QString deploymentSampleRelativePath = QDir::fromNativeSeparators(
        payload.value(QStringLiteral("deploymentSampleRelativePath")).toString().trimmed());
    QString error;
    aitrain::v2::TrainingWorkflowProfileV2 profile;
    aitrain::v2::TrainingWorkflowRequestV2 workflowRequest;
    if (payload.contains(wp::field::datasetPath()) || payload.contains(wp::field::format())
        || payload.contains(wp::field::sampleImagePath())
        || payload.contains(QStringLiteral("pythonExecutable"))
        || payload.contains(QStringLiteral("trainersRoot"))) {
        fail(QStringLiteral("runTrainingWorkflowV2 不接受数据集、部署样本或运行环境原始路径；请使用已登记 Snapshot 身份、包内相对样本路径和 Worker 运行时配置。"));
        return;
    }
    if (taskIdText.isEmpty() || projectRoot.isEmpty() || capabilityId.isEmpty()
        || taskType.isEmpty() || trainingBackend.isEmpty()) {
        fail(QStringLiteral("runTrainingWorkflowV2 需要有效 taskId、项目、完整 Snapshot 身份、能力、任务类型和已注册训练后端。"));
        return;
    }
    const QString normalizedDeploymentSample = QDir::cleanPath(deploymentSampleRelativePath);
    if (!deploymentSampleRelativePath.isEmpty()
        && (!QDir::isRelativePath(deploymentSampleRelativePath)
            || normalizedDeploymentSample == QStringLiteral("..")
            || normalizedDeploymentSample.startsWith(QStringLiteral("../")))) {
        fail(QStringLiteral("runTrainingWorkflowV2 的 deploymentSampleRelativePath 必须是 Snapshot Artifact 内部相对路径。"));
        return;
    }
    if (!aitrain::v2::resolveTrainingWorkflowProfileV2(trainingBackend, &profile, &error)
        || profile.capabilityTaskType != taskType) {
        fail(QStringLiteral("runTrainingWorkflowV2 请求与训练 Workflow Profile 不一致：%1")
            .arg(error.isEmpty() ? QStringLiteral("taskType 不匹配") : error));
        return;
    }
    if (!aitrain::BuiltinCapabilityRegistry::instance().supports(
            capabilityId, taskType, profile.datasetFormat, trainingBackend, &error)) {
        fail(QStringLiteral("runTrainingWorkflowV2 请求与内置能力矩阵不一致：%1").arg(error));
        return;
    }
    if (!aitrain::v2::TaskId::parse(taskIdText, &trainingWorkflowTaskIdV2_, &error)
        || trainingWorkflowTaskIdV2_ != controlTaskId_
        || !aitrain::v2::DatasetId::parse(payload.value(QStringLiteral("datasetId")).toString(),
            &workflowRequest.datasetId, &error)
        || !aitrain::v2::DatasetVersionId::parse(payload.value(QStringLiteral("datasetVersionId")).toString(),
            &workflowRequest.datasetVersionId, &error)
        || !aitrain::v2::SnapshotId::parse(payload.value(QStringLiteral("snapshotId")).toString(),
            &workflowRequest.snapshotId, &error)
        || !aitrain::v2::ArtifactId::parse(payload.value(QStringLiteral("snapshotArtifactId")).toString(),
            &workflowRequest.snapshotArtifactId, &error)) {
        trainingWorkflowTaskIdV2_ = {};
        fail(QStringLiteral("runTrainingWorkflowV2 要求控制任务一致且 Dataset/Version/Snapshot/Artifact 身份完整：%1").arg(error));
        return;
    }
    if (!validateTrainingParameters(parameters, &error)) {
        trainingWorkflowTaskIdV2_ = {};
        fail(QStringLiteral("runTrainingWorkflowV2 参数合同无效：%1").arg(error));
        return;
    }
    if (!QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("runTrainingWorkflowV2 的项目目录不存在。"));
        return;
    }
    const QString pythonProgram = firstUsablePythonExecutable();
    const QString trainersRoot = defaultTrainersRoot();
    if (pythonProgram.isEmpty() || trainersRoot.isEmpty()) {
        fail(QStringLiteral("V2 训练 Workflow 需要可用 Python 和 python_trainers 目录。"));
        return;
    }

    trainingWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    trainingWorkflowDeploymentSampleRelativePath_ = deploymentSampleRelativePath.isEmpty()
        ? QString() : normalizedDeploymentSample;
    trainingWorkflowAdapterConfigV2_.pythonProgram = pythonProgram;
    trainingWorkflowAdapterConfigV2_.trainersRoot = trainersRoot;
    // Core 将基于已验证 Snapshot Artifact inventory 解析该相对路径；在接口完成
    // 身份化前仅透传相对值，Worker 不再接受或解析宿主机样本绝对路径。
    trainingWorkflowAdapterConfigV2_.deploymentSampleRelativePath = trainingWorkflowDeploymentSampleRelativePath_;
    trainingWorkflowAdapterConfigV2_.environment = QProcessEnvironment::systemEnvironment();
    trainingWorkflowAdapterConfigV2_.environment.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    trainingWorkflowAdapterConfigV2_.environment.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    configurePackagedPythonEnvironment(&trainingWorkflowAdapterConfigV2_.environment);
    trainingWorkflowAdapterConfigV2_.cancellationGraceMs = qMax(1000,
        parameters.value(QStringLiteral("cancellationGraceMs")).toInt(5000));
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral("V2 训练 Workflow：正在打开项目工作区。")} });
    if (!trainingWorkspaceV2_->open(projectRoot, &error)) {
        trainingWorkspaceV2_.reset();
        fail(QStringLiteral("无法打开 V2 项目工作区：%1").arg(error));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral("V2 训练 Workflow：项目工作区已打开，正在创建根任务。")} });

    aitrain::v2::TaskSnapshot task;
    if (!trainingWorkspaceV2_->startTask(trainingWorkflowTaskIdV2_, capabilityId, taskType, &task, &error)) {
        trainingWorkspaceV2_.reset();
        fail(QStringLiteral("无法创建 V2 训练根任务：%1").arg(error));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral("V2 训练 Workflow：根任务已创建。")} });

    workflowRequest.templateId = profile.templateId;
    workflowRequest.trainingBackend = profile.trainingBackend;
    workflowRequest.evaluationBackend = profile.evaluationBackend;
    workflowRequest.exportBackend = profile.exportBackend;
    workflowRequest.deploymentBackend = profile.deploymentBackend;
    workflowRequest.parameterSummary = parameters;
    workflowRequest.parameterSummary.insert(QStringLiteral("trainingBackend"), trainingBackend);
    workflowRequest.requireEvidenceBeforeTerminal = true;
    aitrain::v2::TrainingWorkflowDispatchV2 dispatch;
    if (!trainingWorkspaceV2_->beginTrainingWorkflow(trainingWorkflowTaskIdV2_, workflowRequest, &dispatch, &error)) {
        QString ignored;
        trainingWorkspaceV2_->finalizeTask(trainingWorkflowTaskIdV2_, aitrain::v2::TaskState::Failed,
            workflowFailure(aitrain::v2::FailureCode::InternalError,
                QStringLiteral("无法创建 V2 训练 Workflow：%1").arg(error)), &ignored);
        trainingWorkspaceV2_.reset();
        failWithDetails(QStringLiteral("无法创建 V2 训练 Workflow：%1").arg(error),
            aitrain::v2::failureCodeToString(aitrain::v2::FailureCode::InternalError));
        return;
    }
    trainingWorkflowRunIdV2_ = dispatch.workflowRunId;
    if (!dispatch.dispatch.hasStep || dispatch.dispatch.step.kind != QStringLiteral("Train")) {
        QString ignored;
        trainingWorkspaceV2_->finalizeTask(trainingWorkflowTaskIdV2_, aitrain::v2::TaskState::Failed,
            workflowFailure(aitrain::v2::FailureCode::InternalError,
                QStringLiteral("Snapshot 身份校验后未能直接派发 Train。")), &ignored);
        trainingWorkspaceV2_.reset();
        failWithDetails(QStringLiteral("V2 训练 Workflow 未从已登记 Snapshot 直接进入 Train。"),
            aitrain::v2::failureCodeToString(aitrain::v2::FailureCode::InternalError));
        return;
    }
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 3},
        {wp::field::message(), QStringLiteral("V2 训练 Workflow：Snapshot 身份与完整性已校验，开始训练。")}});
    dispatchTrainingWorkflowV2(dispatch);
}

void WorkerSession::dispatchTrainingWorkflowV2(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch)
{
    if (!trainingWorkspaceV2_ || finishingSession_) {
        return;
    }
    if (!dispatch.dispatch.hasStep) {
        finishTrainingWorkflowV2(dispatch);
        return;
    }
    const aitrain::v2::WorkflowStepSnapshotV2& step = dispatch.dispatch.step;
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
        {wp::field::message(), QStringLiteral("V2 Workflow 开始步骤 %1：%2（%3）。")
            .arg(step.ordinal + 1).arg(step.kind, step.backend)}});
    if (canceled_) {
        aitrain::v2::TrainingWorkflowDispatchV2 completed;
        QString error;
        const aitrain::v2::WorkflowStepExecutionResultV2 execution{
            aitrain::v2::WorkflowStepState::Canceled, {},
            workflowFailure(aitrain::v2::FailureCode::Canceled, QStringLiteral("用户取消 V2 训练 Workflow。"))};
        if (!trainingWorkspaceV2_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &error)) {
            failWithDetails(QStringLiteral("无法收口已取消的 V2 Workflow 步骤：%1").arg(error), QStringLiteral("v2_workflow_cancel_failed"));
            return;
        }
        dispatchTrainingWorkflowV2(completed);
        return;
    }
    aitrain::v2::TrainingWorkflowProfileV2 profile;
    const bool hasProfile = aitrain::v2::resolveTrainingWorkflowProfileV2(
        step.parameterSummary.value(QStringLiteral("trainingBackend")).toString(), &profile, nullptr);
    const auto routeIt = hasProfile
        ? std::find_if(profile.steps.cbegin(), profile.steps.cend(), [&step](const auto& route) {
            return route.kind == step.kind;
        })
        : profile.steps.cend();
    const bool usesAdapter = routeIt != profile.steps.cend() && !routeIt->script.isEmpty();
    if (usesAdapter) {
        aitrain::v2::TrainingWorkflowAdapterLaunchV2 launch;
        QString error;
        if (!trainingWorkspaceV2_->prepareTrainingWorkflowAdapterLaunch(dispatch.workflowRunId, step.id,
                trainingWorkflowAdapterConfigV2_, &launch, &error)
            || !trainingWorkspaceV2_->startTrainingWorkflowAdapterStep(dispatch.workflowRunId, step.id, launch.launch,
                [this](const aitrain::v2::TrainingWorkflowDispatchV2& next) {
                    // 只有 Adapter 终态已经提交 Artifact、下一步骤已经引用该不可变
                    // Artifact 后，才向 GUI 暴露文件路径；禁止转发暂存区 candidate。
                    if (next.dispatch.hasStep) {
                        aitrain::v2::VerifiedTrainingWorkflowInputV2 committed;
                        QString resolutionError;
                        if (trainingWorkspaceV2_->resolveTrainingWorkflowStepInput(next.workflowRunId,
                                next.dispatch.step.id, &committed, &resolutionError)) {
                            for (const aitrain::v2::VerifiedWorkflowArtifactFileV2& file : committed.files) {
                                const QString kind = file.relativePath.section(QLatin1Char('/'), 0, 0);
                                send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                                    {QStringLiteral("kind"), kind}, {QStringLiteral("artifactId"), committed.artifactId.toString()},
                                    {QStringLiteral("relativePath"), file.relativePath},
                                    {wp::field::message(), QStringLiteral("训练 Workflow Adapter Artifact 已原子提交并验证。")}});
                            }
                        }
                    }
                    // Adapter settled 回调仍位于 PythonAdapterHostV2 的调用栈中。
                    // 排队派发，避免下一 Adapter 重入 start() 或终态路径销毁当前 Host。
                    QTimer::singleShot(0, this, [this, next]() { dispatchTrainingWorkflowV2(next); });
                },
                &error, [this](const aitrain::v2::ProtocolEnvelope& event) { forwardTrainingWorkflowAdapterEvent(event); })) {
            aitrain::v2::TrainingWorkflowDispatchV2 completed;
            const aitrain::v2::WorkflowStepExecutionResultV2 execution{
                aitrain::v2::WorkflowStepState::Failed, {}, workflowFailure(aitrain::v2::FailureCode::ProcessCrashed,
                    QStringLiteral("无法启动 V2 训练 Workflow %1 Adapter：%2").arg(step.kind, error))};
            QString completionError;
            if (trainingWorkspaceV2_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &completionError)) {
                dispatchTrainingWorkflowV2(completed);
            } else {
                failWithDetails(QStringLiteral("V2 Workflow Adapter 启动失败且无法收口步骤：%1；%2").arg(error, completionError),
                    QStringLiteral("v2_workflow_adapter_start_failed"));
            }
        }
        return;
    }
    QTimer::singleShot(0, this, [this, dispatch]() { runTrainingWorkflowV2LocalStep(dispatch); });
}

void WorkerSession::runTrainingWorkflowV2LocalStep(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch)
{
    if (!trainingWorkspaceV2_ || !dispatch.dispatch.hasStep || finishingSession_) {
        return;
    }
    const aitrain::v2::WorkflowStepSnapshotV2& step = dispatch.dispatch.step;
    aitrain::v2::WorkflowStepExecutionResultV2 execution;
    QString error;
    if (canceled_) {
        execution.state = aitrain::v2::WorkflowStepState::Canceled;
        execution.failure = workflowFailure(aitrain::v2::FailureCode::Canceled, QStringLiteral("用户取消 V2 训练 Workflow。"));
    } else if (step.kind == QStringLiteral("DeploymentValidate")) {
        aitrain::v2::TrainingDeploymentInvocationV2 prepared;
        aitrain::v2::RuntimeInvocationV2 invocation;
        if (!trainingWorkspaceV2_->prepareTrainingWorkflowDeploymentInvocation(dispatch.workflowRunId, step.id,
                trainingWorkflowDeploymentSampleRelativePath_, &prepared, &error)
            || !aitrain::v2::decodeRuntimeInvocationV2(prepared.invocation, &invocation, &error)) {
            execution.failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncomplete,
                QStringLiteral("无法准备 V2 部署验证：%1").arg(error));
        } else {
            aitrain::v2::OnnxRuntimeAdapterV2 adapter;
            const aitrain::v2::RuntimeOperationResultV2 runtime = adapter.deploymentValidate(invocation.model,
                QJsonObject{{QStringLiteral("imagePath"), invocation.imagePath}, {QStringLiteral("outputPath"), invocation.outputPath},
                    {QStringLiteral("options"), invocation.options}});
            if (canceled_) {
                execution.state = aitrain::v2::WorkflowStepState::Canceled;
                execution.failure = workflowFailure(aitrain::v2::FailureCode::Canceled, QStringLiteral("用户在部署验证期间取消 V2 训练 Workflow。"));
            } else if (runtime.status != aitrain::v2::RuntimeStatusV2::Available) {
                execution.failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncompatible,
                    QStringLiteral("V2 ONNX Runtime 部署验证失败（%1）：%2")
                        .arg(aitrain::v2::runtimeStatusV2ToString(runtime.status), runtime.message));
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
                    execution.failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncomplete,
                        QStringLiteral("无法写入 V2 部署验证报告或运行时产物不完整。"));
                } else {
                    aitrain::v2::RuntimeArtifactBundleV2 bundle;
                    const QVector<aitrain::v2::RuntimeArtifactCandidateV2> candidates{{QStringLiteral("deployment_validation_report"), reportPath},
                        {QStringLiteral("deployment_predictions"), predictionsPath}, {QStringLiteral("deployment_overlay"), overlayPath}};
                    if (!trainingWorkspaceV2_->commitRuntimeArtifacts(trainingWorkflowTaskIdV2_, QStringLiteral("deployment_validation_v2"),
                            candidates, &bundle, &error)) {
                        execution.failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncomplete,
                            QStringLiteral("无法提交 V2 部署验证 Artifact：%1").arg(error));
                    } else {
                        execution.state = aitrain::v2::WorkflowStepState::Succeeded;
                        execution.outputArtifactId = bundle.artifactId;
                        for (auto it = bundle.pathsByKind.cbegin(); it != bundle.pathsByKind.cend(); ++it) {
                            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), it.key()},
                                {QStringLiteral("artifactId"), bundle.artifactId.toString()},
                                {QStringLiteral("relativePath"), QDir(QDir(trainingWorkspaceV2_->workspacePath())
                                    .filePath(QStringLiteral("artifact-store/artifacts/%1").arg(bundle.artifactId.toString())))
                                    .relativeFilePath(it.value())},
                                {wp::field::message(), QStringLiteral("V2 部署验证 Artifact。")}});
                        }
                    }
                }
            }
        }
    } else if (step.kind == QStringLiteral("RegisterModel")) {
        aitrain::v2::TrainingModelRegistrationV2 registered;
        if (!trainingWorkspaceV2_->registerTrainingWorkflowModel(dispatch.workflowRunId, step.id, &registered, &error)) {
            execution.failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncomplete,
                QStringLiteral("无法登记官方 YOLO 模型包：%1").arg(error));
        } else {
            execution.state = aitrain::v2::WorkflowStepState::Succeeded;
            execution.outputArtifactId = registered.registrationArtifact.artifactId;
            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), QStringLiteral("model_manifest_v2")},
                {QStringLiteral("artifactId"), registered.registrationArtifact.artifactId.toString()},
                {QStringLiteral("relativePath"), QStringLiteral("model_manifest_v2.json")},
                {wp::field::message(), QStringLiteral("已登记 V2 模型包。")}});
        }
    } else if (step.kind == QStringLiteral("RenderDeliveryReport")) {
        aitrain::v2::RuntimeArtifactBundleV2 report;
        if (!trainingWorkspaceV2_->renderTrainingWorkflowDeliveryReport(dispatch.workflowRunId, step.id, &report, &error)) {
            execution.failure = workflowFailure(aitrain::v2::FailureCode::InternalError,
                QStringLiteral("无法渲染 V2 训练交付报告：%1").arg(error));
        } else {
            execution.state = aitrain::v2::WorkflowStepState::Succeeded;
            execution.outputArtifactId = report.artifactId;
            for (auto it = report.pathsByKind.cbegin(); it != report.pathsByKind.cend(); ++it) {
                send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("kind"), it.key()},
                    {QStringLiteral("artifactId"), report.artifactId.toString()},
                    {QStringLiteral("relativePath"), QDir(QDir(trainingWorkspaceV2_->workspacePath())
                        .filePath(QStringLiteral("artifact-store/artifacts/%1").arg(report.artifactId.toString())))
                        .relativeFilePath(it.value())},
                    {wp::field::message(), QStringLiteral("V2 训练交付报告 Artifact。")}});
            }
        }
    } else {
        execution.failure = workflowFailure(aitrain::v2::FailureCode::BackendUnsupported,
            QStringLiteral("V2 训练 Workflow 不支持步骤：%1").arg(step.kind));
    }

    if (execution.state != aitrain::v2::WorkflowStepState::Succeeded
        && execution.state != aitrain::v2::WorkflowStepState::Canceled) {
        execution.state = aitrain::v2::WorkflowStepState::Failed;
    }
    aitrain::v2::TrainingWorkflowDispatchV2 completed;
    if (!trainingWorkspaceV2_->completeTrainingWorkflowStep(dispatch.workflowRunId, step.id, execution, &completed, &error)) {
        failWithDetails(QStringLiteral("无法收口 V2 Workflow 步骤 %1：%2").arg(step.kind, error),
            QStringLiteral("v2_workflow_step_complete_failed"));
        return;
    }
    dispatchTrainingWorkflowV2(completed);
}

void WorkerSession::forwardTrainingWorkflowAdapterEvent(const aitrain::v2::ProtocolEnvelope& event)
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

void WorkerSession::cancelTrainingWorkflowV2()
{
    canceled_ = true;
    QString error;
    if (!trainingWorkspaceV2_->requestTaskCancellation(trainingWorkflowTaskIdV2_, &error)) {
        failWithDetails(QStringLiteral("无法请求取消 V2 训练 Workflow：%1").arg(error), QStringLiteral("v2_workflow_cancel_failed"));
        return;
    }
    send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
        {wp::field::message(), QStringLiteral("已请求取消 V2 训练 Workflow，等待当前步骤清理并收口。")}});
}

void WorkerSession::finishTrainingWorkflowV2(const aitrain::v2::TrainingWorkflowDispatchV2& dispatch)
{
    if (finishingSession_) {
        return;
    }
    const aitrain::v2::WorkflowRunExecutionResultV2& result = dispatch.dispatch.result;
    bool evidenceCommitted = false;
    QString evidenceError;
    if (trainingWorkspaceV2_) {
        aitrain::v2::EvidenceBundleV2 evidence;
        aitrain::v2::EvidenceArtifactBundleV2 committed;
        if (trainingWorkspaceV2_->buildWorkflowEvidenceBundle(dispatch.workflowRunId, &evidence, &evidenceError)
            && trainingWorkspaceV2_->commitEvidenceBundle(evidence, &committed, &evidenceError)) {
            evidenceCommitted = true;
            send(wp::event::artifact(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                {QStringLiteral("kind"), QStringLiteral("evidence_bundle_v2")},
                {QStringLiteral("artifactId"), committed.artifactId.toString()},
                {QStringLiteral("relativePath"), QStringLiteral("evidence.json")},
                {wp::field::message(), QStringLiteral("V2 Evidence Bundle Artifact。")}});
        } else {
            send(wp::event::log(), QJsonObject{{wp::field::taskId(), activeTaskId_},
                {wp::field::message(), QStringLiteral("Evidence Bundle 提交失败，根任务不会发布成功终态：%1").arg(evidenceError)}});
        }
    }
    if (!evidenceCommitted && trainingWorkspaceV2_) {
        const aitrain::v2::Failure failure = workflowFailure(aitrain::v2::FailureCode::ArtifactIncomplete,
            QStringLiteral("Evidence Bundle 提交失败：%1").arg(evidenceError));
        QString recordError;
        trainingWorkspaceV2_->recordWorkflowEvidenceFailure(dispatch.workflowRunId, failure, &recordError);
        const QString message = recordError.isEmpty() ? failure.message
            : QStringLiteral("%1；记录 Evidence 失败尝试也失败：%2").arg(failure.message, recordError);
        trainingWorkspaceV2_.reset();
        failWithDetails(message, aitrain::v2::failureCodeToString(failure.code));
        return;
    }
    if (trainingWorkspaceV2_) {
        QString finalizationError;
        if (!trainingWorkspaceV2_->closeWorkflowTerminalization(dispatch.workflowRunId,
                &finalizationError)) {
            trainingWorkspaceV2_.reset();
            failWithDetails(QStringLiteral("Evidence 已提交，但根任务终态持久化失败：%1").arg(finalizationError),
                QStringLiteral("v2_terminal_persistence_failed"));
            return;
        }
    }
    if (result.state == aitrain::v2::WorkflowStepState::Succeeded && trainingWorkspaceV2_) {
        send(wp::event::progress(), QJsonObject{{wp::field::taskId(), activeTaskId_}, {QStringLiteral("percent"), 100},
            {wp::field::message(), QStringLiteral("V2 官方 YOLO 训练、评估、导出、部署验证、模型登记与交付报告已完成。")}});
        running_ = false;
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), activeTaskId_},
            {wp::field::message(), QStringLiteral("V2 training workflow completed")}});
        trainingWorkspaceV2_.reset();
        finishSession();
        return;
    }
    const QString message = result.failure.message.isEmpty()
        ? QStringLiteral("V2 训练 Workflow 未成功完成。") : result.failure.message;
    trainingWorkspaceV2_.reset();
    if (result.state == aitrain::v2::WorkflowStepState::Canceled) {
        sendCanceledAndFinish(activeTaskId_, message);
    } else {
        failWithDetails(message, aitrain::v2::failureCodeToString(result.failure.code));
    }
}
