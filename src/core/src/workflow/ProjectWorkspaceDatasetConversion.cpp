#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/dataset/BuiltinDatasetDrivers.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QSet>

namespace aitrain {
namespace {

Failure conversionFailure(const QString& message)
{
    Failure failure;
    failure.code = message == QStringLiteral("dataset_conversion_canceled")
            || message.contains(QStringLiteral("canceled"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("已取消"))
        ? FailureCode::Canceled
        : message.startsWith(QStringLiteral("dataset_conversion_backend_unsupported:"))
        ? FailureCode::BackendUnsupported
        : message.contains(QStringLiteral("source_changed"))
        ? FailureCode::ArtifactIncompatible
        : message.contains(QStringLiteral("source_"))
        ? FailureCode::InvalidDataset
        : FailureCode::ArtifactIncomplete;
    failure.message = message.isEmpty() ? QStringLiteral("dataset_conversion_failed") : message;
    failure.suggestedAction = failure.code == FailureCode::BackendUnsupported
        ? QStringLiteral("选择已实现且具有目标 Dataset Driver 的转换路线。")
        : QStringLiteral("检查外部源数据、目标格式和 Evidence 后重新运行转换。");
    failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

bool hashFile(const QString& path, QString* result, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_conversion_committed_file_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("dataset_conversion_committed_file_read_failed:%1").arg(path);
            return false;
        }
        hash.addData(block);
    }
    *result = QString::fromLatin1(hash.result().toHex());
    return true;
}

bool copyVerifiedTargetTree(const DatasetArtifactConversion& conversion,
    const ArtifactSnapshot& stored,
    const QString& staging,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    QHash<QString, ArtifactFileSnapshot> inventory;
    for (const ArtifactFileSnapshot& file : stored.files) inventory.insert(file.relativePath, file);
    const QSet<QString> metadata{QStringLiteral("conversion_plan.json"),
        QStringLiteral("dataset_conversion_report.json"),
        QStringLiteral("conversion_commit_report.json")};
    int copied = 0;
    for (const QJsonValue& value : conversion.plan.value(QStringLiteral("plannedOutputs")).toArray()) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (error) *error = QStringLiteral("dataset_conversion_canceled");
            return false;
        }
        const QString relative = QDir::cleanPath(value.toString());
        if (metadata.contains(relative)) continue;
        if (relative.isEmpty() || QDir::isAbsolutePath(relative)
            || relative == QStringLiteral("..") || relative.startsWith(QStringLiteral("../"))
            || !inventory.contains(relative)) {
            if (error) *error = QStringLiteral("dataset_conversion_snapshot_inventory_mismatch:%1").arg(relative);
            return false;
        }
        const QString source = QDir(conversion.artifactPath).filePath(relative);
        const QFileInfo sourceInfo(source);
        QString actualHash;
        const ArtifactFileSnapshot expected = inventory.value(relative);
        if (!sourceInfo.isFile() || sourceInfo.isSymLink() || sourceInfo.size() != expected.byteCount
            || !hashFile(source, &actualHash, error) || actualHash != expected.sha256) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("dataset_conversion_snapshot_source_tampered:%1").arg(relative);
            }
            return false;
        }
        const QString destination = QDir(staging).filePath(relative);
        if (!QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(source, destination)) {
            if (error) *error = QStringLiteral("dataset_conversion_snapshot_copy_failed:%1").arg(relative);
            return false;
        }
        ++copied;
    }
    if (copied == 0) {
        if (error) *error = QStringLiteral("dataset_conversion_snapshot_empty");
        return false;
    }
    return true;
}

} // namespace

bool ProjectWorkspace::runDatasetConversionWorkflow(const TaskId& taskId,
    const DatasetConversionWorkflowRequest& request,
    DatasetConversionWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result || request.sourcePath.trimmed().isEmpty()
        || request.sourceFormat.trimmed().isEmpty() || request.targetFormat.trimmed().isEmpty()
        || !request.targetDatasetId.isValid() || request.targetDatasetName.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("dataset_conversion_invalid_workflow_request");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) return false;

    DatasetDriverRegistry drivers;
    if (!registerBuiltinDatasetDrivers(&drivers, error)) return false;
    const DatasetDriver* targetDriver = drivers.driverForFormat(request.targetFormat);

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("dataset_conversion");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QJsonObject parameters{{QStringLiteral("sourceFormat"), request.sourceFormat},
        {QStringLiteral("targetFormat"), request.targetFormat},
        {QStringLiteral("targetDatasetId"), request.targetDatasetId.toString()},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("options"), request.options}};
    QVector<WorkflowStepSnapshot> steps;
    for (int index = 0; index < 2; ++index) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = index;
        step.kind = index == 0 ? QStringLiteral("Convert") : QStringLiteral("RegisterSnapshot");
        step.backend = index == 0 ? QStringLiteral("dataset_conversion_service")
                                  : QStringLiteral("dataset_snapshot_registration");
        step.parameterSummary = parameters;
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;

    DatasetArtifactConversion conversion;
    DatasetSnapshotArtifactBundle snapshot;
    QString executionError;
    DatasetConversionService service(artifactStore_.get(), &storage_, &drivers);
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult runResult;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) {
        if (aitrain::isCancellationRequested(stepCancellation)) {
            return WorkflowStepExecutionResult{WorkflowStepState::Canceled, {},
                conversionFailure(QStringLiteral("dataset_conversion_canceled"))};
        }
        if (step.kind == QStringLiteral("Convert")) {
            DatasetArtifactConversionRequest conversionRequest;
            conversionRequest.sourcePath = request.sourcePath;
            conversionRequest.sourceFormat = request.sourceFormat;
            conversionRequest.targetFormat = request.targetFormat;
            conversionRequest.options = request.options;
            if (!service.convert(taskId, conversionRequest, &conversion, &executionError,
                    stepCancellation)) {
                const Failure failure = conversionFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded,
                conversion.artifactId, {}};
        }
        if (step.kind == QStringLiteral("RegisterSnapshot")) {
            if (!targetDriver || !conversion.artifactId.isValid()) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    conversionFailure(QStringLiteral("dataset_conversion_target_driver_missing"))};
            }
            ArtifactSnapshot stored;
            ArtifactId snapshotArtifactId;
            QString staging;
            if (!storage_.artifact(conversion.artifactId, &stored, &executionError)
                || stored.kind != QStringLiteral("dataset_conversion")
                || !artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot"),
                    &snapshotArtifactId, &staging, &executionError)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    conversionFailure(executionError)};
            }
            const auto abort = [&]() { QString ignored; artifactStore_->abort(staging, &ignored); };
            DatasetOperationContext context;
            context.isCancellationRequested = stepCancellation;
            DatasetInspection inspection;
            DatasetDriverValidationResult validation;
            DatasetSnapshotOptions snapshotOptions;
            snapshotOptions.isCancellationRequested = stepCancellation;
            DatasetSnapshotResult snapshotResult;
            const QString manifestPath = QDir(staging).filePath(QStringLiteral("dataset_snapshot.json"));
            if (!copyVerifiedTargetTree(conversion, stored, staging, stepCancellation, &executionError)
                || !targetDriver->inspect(staging, request.targetFormat, &inspection, context, &executionError)
                || !targetDriver->validate(inspection, &validation, context, &executionError)
                || !validation.valid
                || !createDatasetSnapshot(staging, manifestPath, request.targetFormat,
                    targetDriver->id(), targetDriver->version(), snapshotOptions,
                    &snapshotResult, &executionError)) {
                abort();
                const Failure failure = conversionFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            QString finalPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(snapshotArtifactId, taskId, QStringLiteral("dataset_snapshot"),
                    staging, &storage_, &finalPath, &executionError, stepCancellation,
                    &commitCanceled)) {
                if (QFileInfo::exists(staging)) abort();
                const Failure failure = commitCanceled
                    ? conversionFailure(QStringLiteral("dataset_conversion_canceled"))
                    : conversionFailure(executionError);
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, failure};
            }
            ArtifactSnapshot committed;
            if (!storage_.artifact(snapshotArtifactId, &committed, &executionError)) {
                QString ignored;
                artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    conversionFailure(executionError)};
            }
            QString manifestSha256;
            for (const ArtifactFileSnapshot& file : committed.files) {
                if (file.relativePath == QStringLiteral("dataset_snapshot.json")) {
                    manifestSha256 = file.sha256;
                    break;
                }
            }
            DatasetSnapshotRecord record;
            record.datasetId = request.targetDatasetId;
            record.id = snapshotResult.snapshotId;
            record.taskId = taskId;
            record.artifactId = snapshotArtifactId;
            record.rootPath = finalPath;
            record.datasetFormat = request.targetFormat;
            record.driverId = targetDriver->id();
            record.driverVersion = targetDriver->version();
            record.rootHash = snapshotResult.rootHash;
            record.manifestSha256 = manifestSha256;
            record.fileCount = snapshotResult.fileCount;
            record.totalBytes = snapshotResult.totalBytes;
            if (aitrain::isCancellationRequested(stepCancellation)) {
                QString ignored;
                artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                return WorkflowStepExecutionResult{WorkflowStepState::Canceled, {},
                    conversionFailure(QStringLiteral("dataset_conversion_canceled"))};
            }
            if (manifestSha256.isEmpty() || !storage_.registerDatasetSnapshot(&record, &executionError)) {
                QString ignored;
                artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    conversionFailure(executionError)};
            }
            snapshot.snapshot = record;
            snapshot.artifactPath = finalPath;
            snapshot.manifestPath = QDir(finalPath).filePath(QStringLiteral("dataset_snapshot.json"));
            snapshot.manifest = snapshotResult.manifest;
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded,
                snapshotArtifactId, {}};
        }
        return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
            conversionFailure(QStringLiteral("dataset_conversion_unknown_step"))};
    };
    if (!runner.run(workflow.id, executor, &runResult, error, cancellation)) return false;

    TaskState terminalState = TaskState::Failed;
    Failure finalFailure;
    if (runResult.state == WorkflowStepState::Succeeded) terminalState = TaskState::Succeeded;
    else if (runResult.state == WorkflowStepState::Canceled) terminalState = TaskState::Canceled;
    finalFailure = runResult.failure;
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, finalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)
        || !commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->terminalState = terminalState;
    result->conversionArtifactId = conversion.artifactId;
    result->datasetSnapshot = snapshot.snapshot;
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->failure = finalFailure;
    result->summary = QJsonObject{{QStringLiteral("sourceFormat"), request.sourceFormat},
        {QStringLiteral("targetFormat"), request.targetFormat},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("targetDatasetNamePersisted"), false},
        {QStringLiteral("convertedSampleCount"),
            conversion.conversionReport.value(QStringLiteral("convertedSampleCount"))}};
    return true;
}

} // namespace aitrain
