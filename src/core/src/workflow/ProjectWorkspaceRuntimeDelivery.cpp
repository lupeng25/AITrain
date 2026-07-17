#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/runtime/NcnnRuntimeAdapter.h"
#include "aitrain/runtime/OnnxRuntimeAdapter.h"
#include "aitrain/runtime/TensorRtRuntimeAdapter.h"

#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <algorithm>
#include <memory>

namespace aitrain {
namespace {

const QStringList kRuntimeDeliverySteps{
    QStringLiteral("ImportOrResolveModel"),
    QStringLiteral("ValidateManifest"),
    QStringLiteral("RunInferenceSmoke"),
    QStringLiteral("Benchmark"),
    QStringLiteral("DeploymentValidate"),
    QStringLiteral("RenderDeliveryReport")};

FailureCode failureCodeForRuntimeStatus(RuntimeStatus status)
{
    switch (status) {
    case RuntimeStatus::RuntimeNotImplemented: return FailureCode::RuntimeNotImplemented;
    case RuntimeStatus::SdkMissing: return FailureCode::SdkMissing;
    case RuntimeStatus::DependencyMissing: return FailureCode::DependencyMissing;
    case RuntimeStatus::HardwareUnsupported: return FailureCode::HardwareUnsupported;
    case RuntimeStatus::ArtifactIncompatible: return FailureCode::ArtifactIncompatible;
    case RuntimeStatus::Available: return FailureCode::None;
    }
    return FailureCode::InternalError;
}

Failure runtimeFailure(RuntimeStatus status, const QString& message)
{
    QString action = QStringLiteral("检查 Model Manifest、已提交 Artifact 和目标 Runtime 能力状态后重试。");
    if (status == RuntimeStatus::SdkMissing) action = QStringLiteral("安装并重新配置目标 Runtime SDK 后重试。");
    if (status == RuntimeStatus::DependencyMissing) action = QStringLiteral("补齐目标 Runtime DLL/依赖并重新执行环境检查。");
    if (status == RuntimeStatus::HardwareUnsupported) action = QStringLiteral("改用满足产品要求的硬件，不能把软件缺失记为硬件不支持。");
    if (status == RuntimeStatus::RuntimeNotImplemented) action = QStringLiteral("选择已实现的 Runtime/decoder 组合；不要隐式降级到其他后端。");
    return {failureCodeForRuntimeStatus(status),
        message.isEmpty() ? QStringLiteral("Runtime 操作失败：%1").arg(runtimeStatusToString(status)) : message,
        action, QDateTime::currentDateTimeUtc()};
}

WorkflowStepExecutionResult failedExecution(const Failure& failure)
{
    return {failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {}, failure};
}

WorkflowStepExecutionResult canceledExecution(const QString& message)
{
    return failedExecution({FailureCode::Canceled, message,
        QStringLiteral("确认取消原因后可从新的任务重新执行。"), QDateTime::currentDateTimeUtc()});
}

std::unique_ptr<RuntimeAdapter> runtimeAdapter(const QString& route)
{
    if (route == QStringLiteral("aitrain_onnxruntime")) return std::make_unique<OnnxRuntimeAdapter>();
    if (route == QStringLiteral("aitrain_ncnn")) return std::make_unique<NcnnRuntimeAdapter>();
    if (route == QStringLiteral("aitrain_tensorrt")) return std::make_unique<TensorRtRuntimeAdapter>();
    return {};
}

bool writeBytes(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("无法创建 Runtime Workflow 暂存目录：%1").arg(QFileInfo(path).absolutePath());
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 Runtime Workflow 暂存文件：%1").arg(path);
        return false;
    }
    return true;
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    return writeBytes(path, QJsonDocument(object).toJson(QJsonDocument::Indented), error);
}

QString fileSha256(const QString& path, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取推理样本：%1").arg(path);
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray bytes = file.read(1024 * 1024);
        if (bytes.isEmpty() && file.error() != QFile::NoError) {
            if (error) *error = QStringLiteral("读取推理样本失败：%1").arg(path);
            return {};
        }
        hash.addData(bytes);
    }
    return QString::fromLatin1(hash.result().toHex());
}

QJsonObject persistedRuntimeDetails(const RuntimeOperationResult& operation)
{
    QJsonObject details = operation.details;
    details.remove(QStringLiteral("predictionsPath"));
    details.remove(QStringLiteral("overlayPath"));
    details.remove(QStringLiteral("enginePath"));
    details.remove(QStringLiteral("binPath"));
    return details;
}

TaskState terminalTaskState(WorkflowStepState state)
{
    if (state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

QString currentError(const QString* error, const QString& fallback)
{
    return error && !error->isEmpty() ? *error : fallback;
}

RuntimeArtifactCandidate runtimeCandidate(const QString& kind, const QString& sourcePath)
{
    RuntimeArtifactCandidate candidate;
    candidate.kind = kind;
    candidate.sourcePath = sourcePath;
    return candidate;
}

bool isChildPath(const QString& parentPath, const QString& candidatePath)
{
    const QString parent = QDir::cleanPath(QDir(parentPath).absolutePath());
    const QString candidate = QDir::cleanPath(QFileInfo(candidatePath).absoluteFilePath());
    return candidate.startsWith(parent + QLatin1Char('/'), Qt::CaseInsensitive);
}

QString normalizedArtifactRelativePath(const QString& value, QString* error)
{
    const QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(value.trimmed()));
    if (normalized.isEmpty() || normalized == QStringLiteral(".")
        || normalized == QStringLiteral("..")
        || normalized.startsWith(QStringLiteral("../"))
        || normalized.contains(QStringLiteral("/../"))
        || QDir::isAbsolutePath(normalized)) {
        if (error) *error = QStringLiteral("Runtime 样本必须是 Snapshot Artifact 包内相对路径。");
        return {};
    }
    return normalized;
}

bool copyFileIntoStaging(const QString& sourcePath, const QString& destinationPath, QString* error)
{
    if (!QDir().mkpath(QFileInfo(destinationPath).absolutePath())) {
        if (error) *error = QStringLiteral("无法创建 Runtime 样本暂存目录：%1")
            .arg(QFileInfo(destinationPath).absolutePath());
        return false;
    }
    QFile source(sourcePath);
    QSaveFile destination(destinationPath);
    if (!source.open(QIODevice::ReadOnly) || !destination.open(QIODevice::WriteOnly)) {
        if (error) *error = QStringLiteral("无法打开 Runtime 样本或暂存文件。");
        return false;
    }
    while (!source.atEnd()) {
        const QByteArray block = source.read(1024 * 1024);
        if (block.isEmpty() && source.error() != QFile::NoError) {
            if (error) *error = QStringLiteral("读取 Runtime 样本失败：%1").arg(sourcePath);
            return false;
        }
        if (!block.isEmpty() && destination.write(block) != block.size()) {
            if (error) *error = QStringLiteral("写入 Runtime 样本暂存失败：%1").arg(destinationPath);
            return false;
        }
    }
    if (!destination.commit()) {
        if (error) *error = QStringLiteral("提交 Runtime 样本暂存失败：%1").arg(destinationPath);
        return false;
    }
    return true;
}

double percentile(const QVector<double>& sortedSamples, double fraction)
{
    if (sortedSamples.isEmpty()) return 0.0;
    const double position = fraction * static_cast<double>(sortedSamples.size() - 1);
    const int lower = static_cast<int>(position);
    const int upper = qMin(lower + 1, sortedSamples.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return sortedSamples.at(lower) * (1.0 - weight) + sortedSamples.at(upper) * weight;
}

} // namespace

bool ProjectWorkspace::runRuntimeDeliveryWorkflow(const TaskId& taskId,
    const RuntimeDeliveryWorkflowRequest& request,
    RuntimeDeliveryWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation,
    RuntimeAdapterFactory adapterFactory)
{
    if (error) error->clear();
    if (!isOpen() || !artifactStore_ || !taskId.isValid() || !request.modelPackageId.isValid()
        || !request.sampleDatasetId.isValid() || !request.sampleDatasetVersionId.isValid()
        || !request.sampleSnapshotId.isValid() || !request.sampleSnapshotArtifactId.isValid()
        || request.runtimeRoute.trimmed().isEmpty() || request.sampleRelativePath.trimmed().isEmpty()
        || !result) {
        if (error) *error = QStringLiteral("运行 Runtime Delivery Workflow 需要已打开工作区、运行中任务、ModelPackageId、Runtime 路由和 Snapshot 样本身份。");
        return false;
    }
    *result = {};
    TaskSnapshot task;
    ModelPackageSnapshot modelPackage;
    ArtifactSnapshot sourceArtifact;
    DatasetSnapshotRecord sampleSnapshot;
    ArtifactSnapshot sampleArtifact;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running
        || !storage_.modelPackage(request.modelPackageId, &modelPackage, error)
        || !storage_.artifact(modelPackage.sourceArtifactId, &sourceArtifact, error)
        || !storage_.datasetSnapshot(request.sampleSnapshotId, &sampleSnapshot, error)
        || !storage_.artifact(request.sampleSnapshotArtifactId, &sampleArtifact, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("Runtime Delivery Workflow 只能使用运行中任务、已登记模型包和已登记 Dataset Snapshot Artifact。");
        return false;
    }
    if (sampleSnapshot.datasetId != request.sampleDatasetId
        || sampleSnapshot.datasetVersionId != request.sampleDatasetVersionId
        || sampleSnapshot.artifactId != request.sampleSnapshotArtifactId
        || sampleArtifact.kind != QStringLiteral("dataset_snapshot")
        || sampleArtifact.taskId != sampleSnapshot.taskId) {
        if (error) *error = QStringLiteral("Runtime Delivery Workflow 样本 Snapshot 身份与 Artifact lineage 不一致。");
        return false;
    }
    const QString sampleRelativePath = normalizedArtifactRelativePath(request.sampleRelativePath, error);
    if (sampleRelativePath.isEmpty()) return false;
    ArtifactFileSnapshot expectedSample;
    bool sampleListed = false;
    for (const ArtifactFileSnapshot& file : sampleArtifact.files) {
        if (file.relativePath == sampleRelativePath) {
            expectedSample = file;
            sampleListed = true;
            break;
        }
    }
    if (!sampleListed || expectedSample.byteCount < 0 || expectedSample.sha256.size() != 64) {
        if (error) *error = QStringLiteral("Runtime Delivery Workflow 样本不属于 Snapshot Artifact 文件清单：%1")
            .arg(sampleRelativePath);
        return false;
    }
    const QString sampleArtifactRoot = artifactStore_->artifactPath(sampleSnapshot.artifactId);
    const QString sampleSourcePath = QDir(sampleArtifactRoot).filePath(sampleRelativePath);
    const QFileInfo sample(sampleSourcePath);
    if (!isChildPath(sampleArtifactRoot, sample.absoluteFilePath())
        || !sample.exists() || !sample.isFile() || sample.isSymLink()
        || sample.size() != expectedSample.byteCount) {
        if (error) *error = QStringLiteral("Runtime Delivery Workflow 样本文件越界、丢失或已被修改：%1")
            .arg(sampleRelativePath);
        return false;
    }
    const QString sampleSha256 = fileSha256(sample.absoluteFilePath(), error);
    if (sampleSha256.isEmpty()) return false;
    if (sampleSha256 != expectedSample.sha256) {
        if (error) *error = QStringLiteral("Runtime Delivery Workflow 样本 SHA-256 与 Snapshot Artifact 清单不一致：%1")
            .arg(sampleRelativePath);
        return false;
    }
    const QString stagingRoot = runtimeStagingPath(taskId);
    const QString sampleWorkingPath = QDir(stagingRoot).filePath(
        QStringLiteral("00-sample/%1").arg(sample.fileName()));
    if (!copyFileIntoStaging(sample.absoluteFilePath(), sampleWorkingPath, error)) {
        cleanupRuntimeStaging(taskId, nullptr);
        return false;
    }

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("runtime-delivery");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    QJsonObject parameters;
    parameters.insert(QStringLiteral("modelPackageId"), request.modelPackageId.toString());
    parameters.insert(QStringLiteral("runtimeRoute"), request.runtimeRoute.trimmed());
    parameters.insert(QStringLiteral("sampleDatasetId"), request.sampleDatasetId.toString());
    parameters.insert(QStringLiteral("sampleDatasetVersionId"), request.sampleDatasetVersionId.toString());
    parameters.insert(QStringLiteral("sampleSnapshotId"), request.sampleSnapshotId.toString());
    parameters.insert(QStringLiteral("sampleSnapshotArtifactId"), request.sampleSnapshotArtifactId.toString());
    parameters.insert(QStringLiteral("sampleRelativePath"), sampleRelativePath);
    parameters.insert(QStringLiteral("sampleImageSha256"), sampleSha256);
    parameters.insert(QStringLiteral("options"), request.options);
    QVector<WorkflowStepSnapshot> steps;
    for (int ordinal = 0; ordinal < kRuntimeDeliverySteps.size(); ++ordinal) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = kRuntimeDeliverySteps.at(ordinal);
        step.backend = ordinal == 0 ? QStringLiteral("model_package_registry")
            : (ordinal == kRuntimeDeliverySteps.size() - 1 ? QStringLiteral("evidence_renderer")
                                                           : request.runtimeRoute.trimmed());
        step.parameterSummary = parameters;
        if (ordinal == 0) step.inputArtifactId = modelPackage.sourceArtifactId;
        steps.append(step);
    }
    WorkflowInputBinding input;
    input.workflowRunId = workflow.id;
    input.role = QStringLiteral("model_package");
    input.sourceArtifactId = modelPackage.sourceArtifactId;
    input.sourceTaskId = modelPackage.manifest.sourceTaskId;
    input.sourceArtifactKind = sourceArtifact.kind;
    input.modelPackageId = modelPackage.manifest.modelPackageId;
    input.boundAt = QDateTime::currentDateTimeUtc();
    if (!storage_.createWorkflowRunWithInput(workflow, steps, input, error)) {
        cleanupRuntimeStaging(taskId, nullptr);
        return false;
    }
    result->workflowRunId = workflow.id;

    ModelPackageRuntimeService resolver(&storage_, artifactStore_->rootPath());
    RuntimeModelLocation modelLocation;
    RuntimeCapability capability;
    std::unique_ptr<RuntimeAdapter> adapter;
    RuntimeStatus observedStatus = RuntimeStatus::RuntimeNotImplemented;
    bool statusObserved = false;
    QString observedMessage;
    QJsonObject benchmarkFacts;
    benchmarkFacts.insert(QStringLiteral("available"), false);
    QJsonArray deliveryFacts;

    const auto observe = [&](RuntimeStatus status, const QString& message) {
        observedStatus = status;
        observedMessage = message;
        statusObserved = true;
    };
    const auto commitFiles = [&](const WorkflowStepSnapshot& step,
                                 const QString& bundleKind,
                                 const QVector<RuntimeArtifactCandidate>& candidates,
                                 ArtifactId* artifactId,
                                 QString* commitError) -> bool {
        RuntimeArtifactBundle bundle;
        if (!commitRuntimeArtifacts(taskId, bundleKind, candidates, &bundle, commitError)) return false;
        if (artifactId) *artifactId = bundle.artifactId;
        QJsonObject fact;
        fact.insert(QStringLiteral("ordinal"), step.ordinal);
        fact.insert(QStringLiteral("kind"), step.kind);
        fact.insert(QStringLiteral("backend"), step.backend);
        fact.insert(QStringLiteral("status"), runtimeStatusToString(observedStatus));
        fact.insert(QStringLiteral("outputArtifactId"), bundle.artifactId.toString());
        deliveryFacts.append(fact);
        return true;
    };

    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult executionResult;
    const bool ran = runner.run(workflow.id,
        [&](const WorkflowStepSnapshot& step,
            const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
            QString stepError;
            if (aitrain::isCancellationRequested(stepCancellation)) {
                return canceledExecution(QStringLiteral("Runtime Delivery Workflow 在步骤执行前收到取消请求。"));
            }
            if (step.kind == QStringLiteral("ImportOrResolveModel")) {
                bool exists = false;
                if (!storage_.artifactExists(modelPackage.sourceArtifactId, &exists, &stepError) || !exists) {
                    observe(RuntimeStatus::ArtifactIncompatible,
                        currentError(&stepError, QStringLiteral("Model Package 引用的来源 Artifact 未提交或已丢失。")));
                    return failedExecution(runtimeFailure(observedStatus, observedMessage));
                }
                QJsonObject fact;
                fact.insert(QStringLiteral("ordinal"), step.ordinal);
                fact.insert(QStringLiteral("kind"), step.kind);
                fact.insert(QStringLiteral("backend"), step.backend);
                fact.insert(QStringLiteral("status"), QStringLiteral("resolved"));
                fact.insert(QStringLiteral("outputArtifactId"), modelPackage.sourceArtifactId.toString());
                deliveryFacts.append(fact);
                return {WorkflowStepState::Succeeded, modelPackage.sourceArtifactId, {}};
            }
            if (step.kind == QStringLiteral("ValidateManifest")) {
                QString resolveError;
                if (!resolver.resolve(request.modelPackageId, request.runtimeRoute, &modelLocation, &capability, &resolveError)) {
                    RuntimeStatus status = capability.runtimeStatus;
                    if (status == RuntimeStatus::Available) status = RuntimeStatus::RuntimeNotImplemented;
                    observe(status, resolveError);
                    return failedExecution(runtimeFailure(status, resolveError));
                }
                adapter = adapterFactory ? adapterFactory(request.runtimeRoute) : runtimeAdapter(request.runtimeRoute);
                if (!adapter) {
                    observe(RuntimeStatus::RuntimeNotImplemented,
                        QStringLiteral("该 Runtime 路由没有 AITrain C++  Adapter；外部官方路线不能伪装成本地 Runtime。"));
                    return failedExecution(runtimeFailure(observedStatus, observedMessage));
                }
                RuntimeOperationResult operation = adapter->validateModel(modelLocation);
                if (operation.status == RuntimeStatus::Available) operation = adapter->probe(modelLocation);
                observe(operation.status, operation.message);
                if (operation.status != RuntimeStatus::Available) {
                    return failedExecution(runtimeFailure(operation.status, operation.message));
                }
                const QString directory = QDir(stagingRoot).filePath(QStringLiteral("01-validate-manifest"));
                const QString reportPath = QDir(directory).filePath(QStringLiteral("manifest_validation_report.json"));
                QString manifestError;
                const QJsonObject manifest = encodeModelManifest(modelLocation.manifest, &manifestError);
                QJsonObject report;
                report.insert(QStringLiteral("schemaVersion"), 2);
                report.insert(QStringLiteral("reportKind"), QStringLiteral("manifest_validation"));
                report.insert(QStringLiteral("modelPackageId"), request.modelPackageId.toString());
                report.insert(QStringLiteral("sourceArtifactId"), modelPackage.sourceArtifactId.toString());
                report.insert(QStringLiteral("runtimeRoute"), request.runtimeRoute);
                report.insert(QStringLiteral("sampleDatasetId"), request.sampleDatasetId.toString());
                report.insert(QStringLiteral("sampleDatasetVersionId"), request.sampleDatasetVersionId.toString());
                report.insert(QStringLiteral("sampleSnapshotId"), request.sampleSnapshotId.toString());
                report.insert(QStringLiteral("sampleSnapshotArtifactId"), request.sampleSnapshotArtifactId.toString());
                report.insert(QStringLiteral("sampleRelativePath"), sampleRelativePath);
                report.insert(QStringLiteral("runtimeStatus"), runtimeStatusToString(operation.status));
                report.insert(QStringLiteral("capability"), capability.toJson());
                report.insert(QStringLiteral("manifest"), manifest);
                if (manifest.isEmpty() || !writeJson(reportPath, report, &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete,
                        manifestError.isEmpty() ? currentError(&stepError, QStringLiteral("Manifest 报告生成失败。")) : manifestError,
                        QStringLiteral("修复 Manifest 报告生成或 Artifact Store 后重试。"), QDateTime::currentDateTimeUtc()});
                }
                ArtifactId output;
                QVector<RuntimeArtifactCandidate> candidates;
                candidates.append(runtimeCandidate(QStringLiteral("manifest_validation_report"), reportPath));
                if (!commitFiles(step, QStringLiteral("runtime_manifest_validation"),
                        candidates, &output, &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete, currentError(&stepError, QStringLiteral("Manifest 校验 Artifact 提交失败。")),
                        QStringLiteral("修复 Artifact Store 后重试。"), QDateTime::currentDateTimeUtc()});
                }
                return {WorkflowStepState::Succeeded, output, {}};
            }
            if (step.kind == QStringLiteral("RunInferenceSmoke")
                || step.kind == QStringLiteral("Benchmark")
                || step.kind == QStringLiteral("DeploymentValidate")) {
                if (!adapter) {
                    observe(RuntimeStatus::RuntimeNotImplemented, QStringLiteral("Runtime Adapter 尚未通过 ValidateManifest。"));
                    return failedExecution(runtimeFailure(observedStatus, observedMessage));
                }
                const QString slug = step.kind == QStringLiteral("RunInferenceSmoke") ? QStringLiteral("02-inference-smoke")
                    : (step.kind == QStringLiteral("Benchmark") ? QStringLiteral("03-benchmark") : QStringLiteral("04-deployment-validate"));
                const QString outputPath = QDir(stagingRoot).filePath(slug);
                RuntimeOperationResult operation;
                double elapsedMs = 0.0;
                QJsonObject benchmarkMeasurement;
                if (step.kind == QStringLiteral("Benchmark")) {
                    const int warmupIterations = qBound(0,
                        request.options.value(QStringLiteral("benchmarkWarmup")).toInt(1), 5);
                    const int measuredIterations = qBound(1,
                        request.options.value(QStringLiteral("benchmarkIterations")).toInt(5), 20);
                    for (int iteration = 0; iteration < warmupIterations; ++iteration) {
                        if (aitrain::isCancellationRequested(stepCancellation)) {
                            return canceledExecution(QStringLiteral("Runtime Benchmark 预热阶段收到取消请求，暂存产物不会提交。"));
                        }
                        QJsonObject invocation;
                        invocation.insert(QStringLiteral("imagePath"), sampleWorkingPath);
                        invocation.insert(QStringLiteral("outputPath"), QDir(outputPath).filePath(
                            QStringLiteral("warmup-%1").arg(iteration + 1)));
                        invocation.insert(QStringLiteral("options"), request.options);
                        operation = adapter->infer(modelLocation, invocation);
                        observe(operation.status, operation.message);
                        if (operation.status != RuntimeStatus::Available) {
                            return failedExecution(runtimeFailure(operation.status, operation.message));
                        }
                    }
                    QVector<double> samplesMs;
                    QJsonArray sampleValues;
                    samplesMs.reserve(measuredIterations);
                    for (int iteration = 0; iteration < measuredIterations; ++iteration) {
                        if (aitrain::isCancellationRequested(stepCancellation)) {
                            return canceledExecution(QStringLiteral("Runtime Benchmark 采样阶段收到取消请求，暂存产物不会提交。"));
                        }
                        QJsonObject invocation;
                        invocation.insert(QStringLiteral("imagePath"), sampleWorkingPath);
                        invocation.insert(QStringLiteral("outputPath"), QDir(outputPath).filePath(
                            QStringLiteral("iteration-%1").arg(iteration + 1)));
                        invocation.insert(QStringLiteral("options"), request.options);
                        QElapsedTimer sampleTimer;
                        sampleTimer.start();
                        operation = adapter->infer(modelLocation, invocation);
                        const double sampleMs = operation.details.value(QStringLiteral("elapsedMs"))
                                                    .toDouble(static_cast<double>(sampleTimer.elapsed()));
                        observe(operation.status, operation.message);
                        if (operation.status != RuntimeStatus::Available) {
                            return failedExecution(runtimeFailure(operation.status, operation.message));
                        }
                        samplesMs.append(sampleMs);
                        sampleValues.append(sampleMs);
                    }
                    std::sort(samplesMs.begin(), samplesMs.end());
                    elapsedMs = percentile(samplesMs, 0.50);
                    benchmarkMeasurement.insert(QStringLiteral("benchmarkKind"), QStringLiteral("smoke_timing"));
                    benchmarkMeasurement.insert(QStringLiteral("warmupIterations"), warmupIterations);
                    benchmarkMeasurement.insert(QStringLiteral("measuredIterations"), measuredIterations);
                    benchmarkMeasurement.insert(QStringLiteral("samplesMs"), sampleValues);
                    benchmarkMeasurement.insert(QStringLiteral("minMs"), samplesMs.first());
                    benchmarkMeasurement.insert(QStringLiteral("p50Ms"), elapsedMs);
                    benchmarkMeasurement.insert(QStringLiteral("p95Ms"), percentile(samplesMs, 0.95));
                    benchmarkMeasurement.insert(QStringLiteral("maxMs"), samplesMs.last());
                } else {
                    QJsonObject invocation;
                    invocation.insert(QStringLiteral("imagePath"), sampleWorkingPath);
                    invocation.insert(QStringLiteral("outputPath"), outputPath);
                    invocation.insert(QStringLiteral("options"), request.options);
                    QElapsedTimer timer;
                    timer.start();
                    if (step.kind == QStringLiteral("RunInferenceSmoke")) {
                        operation = adapter->infer(modelLocation, invocation);
                    } else {
                        operation = adapter->deploymentValidate(modelLocation, invocation);
                    }
                    elapsedMs = operation.details.value(QStringLiteral("elapsedMs"))
                                    .toDouble(static_cast<double>(timer.elapsed()));
                }
                if (aitrain::isCancellationRequested(stepCancellation)) {
                    return canceledExecution(QStringLiteral("Runtime 操作结束后检测到取消请求，暂存产物不会提交。"));
                }
                observe(operation.status, operation.message);
                if (operation.status != RuntimeStatus::Available) {
                    return failedExecution(runtimeFailure(operation.status, operation.message));
                }
                const QString operationReportPath = QDir(outputPath).filePath(QStringLiteral("runtime_operation_report.json"));
                QJsonObject operationReport;
                operationReport.insert(QStringLiteral("schemaVersion"), 2);
                operationReport.insert(QStringLiteral("operation"), step.kind);
                operationReport.insert(QStringLiteral("modelPackageId"), request.modelPackageId.toString());
                operationReport.insert(QStringLiteral("runtimeRoute"), request.runtimeRoute);
                operationReport.insert(QStringLiteral("runtimeStatus"), runtimeStatusToString(operation.status));
                operationReport.insert(QStringLiteral("sampleDatasetId"), request.sampleDatasetId.toString());
                operationReport.insert(QStringLiteral("sampleDatasetVersionId"), request.sampleDatasetVersionId.toString());
                operationReport.insert(QStringLiteral("sampleSnapshotId"), request.sampleSnapshotId.toString());
                operationReport.insert(QStringLiteral("sampleSnapshotArtifactId"), request.sampleSnapshotArtifactId.toString());
                operationReport.insert(QStringLiteral("sampleRelativePath"), sampleRelativePath);
                operationReport.insert(QStringLiteral("sampleImageSha256"), sampleSha256);
                operationReport.insert(QStringLiteral("elapsedMs"), elapsedMs);
                operationReport.insert(QStringLiteral("details"), persistedRuntimeDetails(operation));
                if (!benchmarkMeasurement.isEmpty()) {
                    operationReport.insert(QStringLiteral("benchmark"), benchmarkMeasurement);
                }
                if (!writeJson(operationReportPath, operationReport, &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete, currentError(&stepError, QStringLiteral("Runtime 操作报告暂存失败。")),
                        QStringLiteral("修复 Runtime 报告暂存后重试。"), QDateTime::currentDateTimeUtc()});
                }
                const QString predictionsPath = operation.details.value(QStringLiteral("predictionsPath")).toString();
                const QString overlayPath = operation.details.value(QStringLiteral("overlayPath")).toString();
                const QString reportKind = step.kind == QStringLiteral("RunInferenceSmoke") ? QStringLiteral("inference_smoke_report")
                    : (step.kind == QStringLiteral("Benchmark") ? QStringLiteral("benchmark_report") : QStringLiteral("deployment_validation_report"));
                ArtifactId output;
                const QString bundleKind = step.kind == QStringLiteral("RunInferenceSmoke") ? QStringLiteral("runtime_inference_smoke")
                    : (step.kind == QStringLiteral("Benchmark") ? QStringLiteral("runtime_benchmark") : QStringLiteral("runtime_deployment_validation"));
                QVector<RuntimeArtifactCandidate> candidates;
                candidates.append(runtimeCandidate(QStringLiteral("predictions"), predictionsPath));
                candidates.append(runtimeCandidate(QStringLiteral("overlay"), overlayPath));
                candidates.append(runtimeCandidate(reportKind, operationReportPath));
                if (!commitFiles(step, bundleKind, candidates, &output, &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete, currentError(&stepError, QStringLiteral("Runtime Artifact 提交失败。")),
                        QStringLiteral("修复 Runtime Artifact 提交后重试。"), QDateTime::currentDateTimeUtc()});
                }
                if (step.kind == QStringLiteral("Benchmark")) {
                    benchmarkFacts = benchmarkMeasurement;
                    benchmarkFacts.insert(QStringLiteral("available"), true);
                    benchmarkFacts.insert(QStringLiteral("runtimeStatus"), runtimeStatusToString(operation.status));
                    benchmarkFacts.insert(QStringLiteral("artifactId"), output.toString());
                    benchmarkFacts.insert(QStringLiteral("limitation"),
                        QStringLiteral("这是固定样本、本机、小次数 smoke timing，不构成统计性能验收。"));
                }
                return {WorkflowStepState::Succeeded, output, {}};
            }
            if (step.kind == QStringLiteral("RenderDeliveryReport")) {
                const QString directory = QDir(stagingRoot).filePath(QStringLiteral("05-delivery-report"));
                const QString jsonPath = QDir(directory).filePath(QStringLiteral("delivery_report.json"));
                const QString markdownPath = QDir(directory).filePath(QStringLiteral("delivery_report.md"));
                QJsonObject report;
                report.insert(QStringLiteral("schemaVersion"), 2);
                report.insert(QStringLiteral("reportKind"), QStringLiteral("workflow_delivery_summary"));
                report.insert(QStringLiteral("modelPackageId"), request.modelPackageId.toString());
                report.insert(QStringLiteral("runtimeRoute"), request.runtimeRoute);
                report.insert(QStringLiteral("runtimeStatus"),
                    statusObserved ? runtimeStatusToString(observedStatus) : QStringLiteral("not_probed"));
                report.insert(QStringLiteral("sampleDatasetId"), request.sampleDatasetId.toString());
                report.insert(QStringLiteral("sampleDatasetVersionId"), request.sampleDatasetVersionId.toString());
                report.insert(QStringLiteral("sampleSnapshotId"), request.sampleSnapshotId.toString());
                report.insert(QStringLiteral("sampleSnapshotArtifactId"), request.sampleSnapshotArtifactId.toString());
                report.insert(QStringLiteral("sampleRelativePath"), sampleRelativePath);
                report.insert(QStringLiteral("sampleImageSha256"), sampleSha256);
                report.insert(QStringLiteral("benchmark"), benchmarkFacts);
                report.insert(QStringLiteral("steps"), deliveryFacts);
                QJsonArray limitations;
                limitations.append(QStringLiteral("本报告只证明已登记模型包在当前本机 Runtime 的工作流事实，不构成客户域精度验收。"));
                limitations.append(QStringLiteral("TensorRT decoder 未实现时必须保持 RuntimeNotImplemented，不得声明 GPU 推理成功。"));
                report.insert(QStringLiteral("limitations"), limitations);
                const QString markdown = QStringLiteral("# Runtime Workflow Delivery Summary \n\n"
                    "- Model Package：`%1`\n- Runtime：`%2`\n- Runtime Status：`%3`\n- Sample SHA-256：`%4`\n\n"
                    "本文件是工作流内交付摘要。终态后还会由 EvidenceRenderer 生成完整 JSON、Markdown、HTML 与 Model Card。\n\n"
                    "本报告仅陈述当前工作流事实，不构成客户域精度或外部机器验收。\n")
                    .arg(request.modelPackageId.toString(), request.runtimeRoute,
                        statusObserved ? runtimeStatusToString(observedStatus) : QStringLiteral("not_probed"), sampleSha256);
                if (!writeJson(jsonPath, report, &stepError) || !writeBytes(markdownPath, markdown.toUtf8(), &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete, currentError(&stepError, QStringLiteral("交付报告暂存失败。")),
                        QStringLiteral("修复交付报告暂存后重试。"), QDateTime::currentDateTimeUtc()});
                }
                ArtifactId output;
                QVector<RuntimeArtifactCandidate> candidates;
                candidates.append(runtimeCandidate(QStringLiteral("delivery_report_json"), jsonPath));
                candidates.append(runtimeCandidate(QStringLiteral("delivery_report_markdown"), markdownPath));
                if (!commitFiles(step, QStringLiteral("runtime_delivery_report"),
                        candidates, &output, &stepError)) {
                    return failedExecution({FailureCode::ArtifactIncomplete, currentError(&stepError, QStringLiteral("交付报告 Artifact 提交失败。")),
                        QStringLiteral("修复交付报告 Artifact 提交后重试。"), QDateTime::currentDateTimeUtc()});
                }
                return {WorkflowStepState::Succeeded, output, {}};
            }
            return failedExecution({FailureCode::InternalError,
                QStringLiteral("Runtime Delivery Workflow 出现未注册步骤：%1").arg(step.kind),
                QStringLiteral("修复固定 Workflow 模板。"), QDateTime::currentDateTimeUtc()});
        }, &executionResult, error, cancellation);
    if (!ran) {
        QString ignored;
        cleanupRuntimeStaging(taskId, &ignored);
        return false;
    }

    QString cleanupError;
    const bool stagingCleaned = cleanupRuntimeStaging(taskId, &cleanupError);
    TaskState terminalState = terminalTaskState(executionResult.state);
    Failure terminalFailure = executionResult.failure;
    if (!stagingCleaned) {
        terminalState = TaskState::Failed;
        terminalFailure = {FailureCode::ArtifactIncomplete, cleanupError,
            QStringLiteral("清理 .runtime-staging 后重新执行。"), QDateTime::currentDateTimeUtc()};
    } else if (terminalState != TaskState::Succeeded) {
        const FailureCode expectedCode = terminalState == TaskState::Canceled
            ? FailureCode::Canceled : (terminalFailure.isFailure() && terminalFailure.code != FailureCode::Canceled
                    ? terminalFailure.code : FailureCode::InternalError);
        terminalFailure.code = expectedCode;
        if (terminalFailure.message.trimmed().isEmpty()) {
            terminalFailure.message = QStringLiteral("Runtime Delivery Workflow 未返回完整终态 Failure。");
        }
        if (terminalFailure.suggestedAction.trimmed().isEmpty()) {
            terminalFailure.suggestedAction = terminalState == TaskState::Canceled
                ? QStringLiteral("确认取消原因后可从新的任务重新执行。")
                : QStringLiteral("检查 Workflow Step 终态和 Runtime 状态后重试。");
        }
        if (!terminalFailure.occurredAt.isValid()) {
            terminalFailure.occurredAt = QDateTime::currentDateTimeUtc();
        }
    }
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot currentTask;
        if (!storage_.task(taskId, &currentTask, error)) return false;
        if (currentTask.state == TaskState::Running
            && !taskCoordinator_->requestCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, terminalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;

    EvidenceBundle evidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)) return false;
    evidence.runtimeStatus.insert(QStringLiteral("modelPackageId"), request.modelPackageId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("modelProducerTaskId"), input.sourceTaskId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("modelSourceArtifactId"), input.sourceArtifactId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("modelSourceArtifactKind"), input.sourceArtifactKind);
    evidence.runtimeStatus.insert(QStringLiteral("runtimeRoute"), request.runtimeRoute);
    evidence.runtimeStatus.insert(QStringLiteral("sampleDatasetId"), request.sampleDatasetId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("sampleDatasetVersionId"), request.sampleDatasetVersionId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("sampleSnapshotId"), request.sampleSnapshotId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("sampleSnapshotArtifactId"), request.sampleSnapshotArtifactId.toString());
    evidence.runtimeStatus.insert(QStringLiteral("sampleRelativePath"), sampleRelativePath);
    evidence.runtimeStatus.insert(QStringLiteral("sampleImageSha256"), sampleSha256);
    evidence.runtimeStatus.insert(QStringLiteral("statusObserved"), statusObserved);
    evidence.runtimeStatus.insert(QStringLiteral("status"),
        statusObserved ? runtimeStatusToString(observedStatus) : QStringLiteral("not_probed"));
    evidence.runtimeStatus.insert(QStringLiteral("message"), observedMessage);
    evidence.runtimeStatus.insert(QStringLiteral("capability"), capability.toJson());
    evidence.benchmark = benchmarkFacts;
    evidence.evaluation = QJsonObject{};
    evidence.evaluation.insert(QStringLiteral("available"), false);
    evidence.evaluation.insert(QStringLiteral("reason"),
        QStringLiteral("-503 是推理/部署 Workflow，不重新执行训练评估。"));
    evidence.limitations.append(QStringLiteral("Runtime Delivery Workflow 不接受裸模型路径，只解析已登记 ModelPackageId 和已提交 Artifact。"));
    if (request.runtimeRoute == QStringLiteral("aitrain_tensorrt")) {
        evidence.limitations.append(QStringLiteral("当前 TensorRT 官方 YOLO decoder 未实现，不得声明真实 TensorRT 推理或 GPU 验收成功。"));
    }
    EvidenceArtifactBundle evidenceArtifact;
    if (!commitEvidenceBundle(evidence, &evidenceArtifact, error)) {
        const Failure evidenceFailure{FailureCode::ArtifactIncomplete,
            currentError(error, QStringLiteral("Evidence Bundle 提交失败。")),
            QStringLiteral("保留封存终态并在恢复流程中重试 Evidence 提交。"), QDateTime::currentDateTimeUtc()};
        QString ignored;
        recordWorkflowEvidenceFailure(workflow.id, evidenceFailure, &ignored);
        return false;
    }
    if (!closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->state = terminalState == TaskState::Succeeded ? WorkflowStepState::Succeeded
        : (terminalState == TaskState::Canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed);
    result->finalOutputArtifactId = executionResult.finalOutputArtifactId;
    result->evidence = evidenceArtifact;
    result->runtimeStatus = observedStatus;
    result->runtimeStatusObserved = statusObserved;
    result->failure = terminalFailure;
    return true;
}

} // namespace aitrain
