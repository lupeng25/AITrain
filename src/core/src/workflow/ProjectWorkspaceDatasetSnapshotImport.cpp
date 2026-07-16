#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/dataset/BuiltinDatasetDrivers.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QTemporaryDir>

#include <algorithm>

namespace aitrain {
namespace {

Failure importFailure(const QString& message)
{
    Failure failure;
    failure.code = message.contains(QStringLiteral("canceled"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("已取消"))
        ? FailureCode::Canceled
        : message.contains(QStringLiteral("unsupported"), Qt::CaseInsensitive)
        ? FailureCode::BackendUnsupported
        : message.contains(QStringLiteral("changed"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("hash"), Qt::CaseInsensitive)
        ? FailureCode::ArtifactIncompatible
        : FailureCode::InvalidDataset;
    failure.message = message.isEmpty()
        ? QStringLiteral("dataset_snapshot_import_failed") : message;
    failure.suggestedAction = failure.code == FailureCode::Canceled
        ? QStringLiteral("如需继续，请重新发起数据集快照导入。")
        : QStringLiteral("检查外部源数据、格式和 Evidence 后重新导入。");
    failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

bool safeRelativePath(const QString& value)
{
    const QString clean = QDir::cleanPath(value);
    return !clean.isEmpty() && clean != QStringLiteral(".") && clean != QStringLiteral("..")
        && !QDir::isAbsolutePath(clean) && !clean.startsWith(QStringLiteral("../"));
}

bool hashFile(const QString& path, QString* hash, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_snapshot_import_source_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_source_read_failed:%1").arg(path);
            return false;
        }
        digest.addData(block);
    }
    *hash = QString::fromLatin1(digest.result().toHex());
    return true;
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    QSaveFile file(path);
    const QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Indented);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("dataset_snapshot_import_plan_write_failed:%1")
            .arg(file.errorString());
        return false;
    }
    return true;
}

bool snapshotSource(const DatasetDriver* driver,
    const DatasetInspection& inspection,
    const DatasetSnapshotOptions& options,
    DatasetSnapshotResult* result,
    QString* error)
{
    QTemporaryDir temporary;
    if (!temporary.isValid()) {
        if (error) *error = QStringLiteral("dataset_snapshot_import_temp_unavailable");
        return false;
    }
    return driver->snapshot(inspection, temporary.filePath(QStringLiteral("snapshot.json")),
        options, result, error);
}

bool copyFrozenInventory(const QString& sourceRoot,
    const QJsonObject& frozenManifest,
    const QString& staging,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    const QDir source(sourceRoot);
    const QDir destination(staging);
    for (const QJsonValue& value : frozenManifest.value(QStringLiteral("files")).toArray()) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_canceled");
            return false;
        }
        const QJsonObject item = value.toObject();
        const QString relative = QDir::cleanPath(item.value(QStringLiteral("relativePath")).toString());
        const qint64 expectedBytes = item.value(QStringLiteral("bytes")).toString().toLongLong();
        const QString expectedHash = item.value(QStringLiteral("sha256")).toString();
        const QString sourcePath = source.filePath(relative);
        const QFileInfo sourceInfo(sourcePath);
        QString actualHash;
        if (!safeRelativePath(relative) || relative == QStringLiteral("dataset_snapshot.json")
            || !sourceInfo.isFile() || sourceInfo.isSymLink()
            || sourceInfo.size() != expectedBytes || expectedHash.size() != 64
            || !hashFile(sourcePath, &actualHash, error) || actualHash != expectedHash) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("dataset_snapshot_import_source_changed:%1").arg(relative);
            }
            return false;
        }
        const QString target = destination.filePath(relative);
        if (!QDir().mkpath(QFileInfo(target).absolutePath()) || QFileInfo::exists(target)
            || !QFile::copy(sourcePath, target)) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_copy_failed:%1").arg(relative);
            return false;
        }
    }
    return true;
}

} // namespace

bool ProjectWorkspace::runDatasetSnapshotImportWorkflow(const TaskId& taskId,
    const DatasetSnapshotImportWorkflowRequest& request,
    DatasetSnapshotImportWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result || request.sourcePath.trimmed().isEmpty()
        || request.sourceFormat.trimmed().isEmpty() || !request.targetDatasetId.isValid()
        || request.targetDatasetName.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("dataset_snapshot_import_invalid_request");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) return false;

    DatasetDriverRegistry drivers;
    if (!registerBuiltinDatasetDrivers(&drivers, error)) return false;
    const DatasetDriver* driver = drivers.driverForFormat(request.sourceFormat);
    const qsizetype maxFileCount = request.options.contains(QStringLiteral("maxFiles"))
        ? qMax<qsizetype>(1, request.options.value(QStringLiteral("maxFiles")).toInt())
        : 1000000;

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("dataset_snapshot_import");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QJsonObject parameters{{QStringLiteral("sourceFormat"), request.sourceFormat},
        {QStringLiteral("targetDatasetId"), request.targetDatasetId.toString()},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("options"), request.options}};
    QVector<WorkflowStepSnapshot> steps;
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = ordinal == 0 ? QStringLiteral("PlanSnapshotImport")
                                 : QStringLiteral("MaterializeAndRegisterSnapshot");
        step.backend = ordinal == 0 ? QStringLiteral("dataset_driver")
                                    : QStringLiteral("artifact_store");
        step.parameterSummary = parameters;
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;

    DatasetInspection sourceInspection;
    DatasetSnapshotResult frozenSnapshot;
    QJsonObject importPlan;
    ArtifactId planArtifactId;
    DatasetSnapshotArtifactBundle snapshotBundle;
    QString executionError;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult run;
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) {
        if (aitrain::isCancellationRequested(stepCancellation)) {
            return WorkflowStepExecutionResult{WorkflowStepState::Canceled, {},
                importFailure(QStringLiteral("dataset_snapshot_import_canceled"))};
        }
        if (!driver) {
            return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                importFailure(QStringLiteral("dataset_snapshot_import_backend_unsupported"))};
        }
        DatasetOperationContext context;
        context.isCancellationRequested = stepCancellation;
        if (step.kind == QStringLiteral("PlanSnapshotImport")) {
            DatasetDriverValidationResult validation;
            DatasetSnapshotOptions options;
            options.maxFileCount = maxFileCount;
            options.isCancellationRequested = stepCancellation;
            if (!driver->inspect(request.sourcePath, request.sourceFormat,
                    &sourceInspection, context, &executionError)
                || !driver->validate(sourceInspection, &validation, context, &executionError)
                || !validation.valid
                || !snapshotSource(driver, sourceInspection, options, &frozenSnapshot, &executionError)) {
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_snapshot_import_invalid_dataset");
                const Failure failure = importFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            importPlan = QJsonObject{{QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("kind"), QStringLiteral("dataset_snapshot_import_plan")},
                {QStringLiteral("sourceFormat"), request.sourceFormat},
                {QStringLiteral("driverId"), driver->id()},
                {QStringLiteral("driverVersion"), driver->version()},
                {QStringLiteral("sourceRootHash"), frozenSnapshot.rootHash},
                {QStringLiteral("fileCount"), QString::number(frozenSnapshot.fileCount)},
                {QStringLiteral("totalBytes"), QString::number(frozenSnapshot.totalBytes)},
                {QStringLiteral("classDefinitions"), frozenSnapshot.manifest.value(QStringLiteral("classDefinitions"))},
                {QStringLiteral("files"), frozenSnapshot.manifest.value(QStringLiteral("files"))}};
            importPlan.insert(QStringLiteral("planHash"), QString::fromLatin1(
                QCryptographicHash::hash(QJsonDocument(importPlan).toJson(QJsonDocument::Compact),
                    QCryptographicHash::Sha256).toHex()));
            QString staging;
            if (!artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot_import_plan"),
                    &planArtifactId, &staging, &executionError)
                || !writeJson(QDir(staging).filePath(QStringLiteral("snapshot_import_plan.json")),
                    importPlan, &executionError)) {
                if (!staging.isEmpty()) { QString ignored; artifactStore_->abort(staging, &ignored); }
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, importFailure(executionError)};
            }
            QString planPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(planArtifactId, taskId,
                    QStringLiteral("dataset_snapshot_import_plan"), staging, &storage_,
                    &planPath, &executionError, stepCancellation, &commitCanceled)) {
                if (QFileInfo::exists(staging)) { QString ignored; artifactStore_->abort(staging, &ignored); }
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    importFailure(commitCanceled ? QStringLiteral("dataset_snapshot_import_canceled")
                                                 : executionError)};
            }
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded, planArtifactId, {}};
        }
        if (step.kind == QStringLiteral("MaterializeAndRegisterSnapshot")) {
            ArtifactSnapshot storedPlan;
            if (!storage_.artifact(planArtifactId, &storedPlan, &executionError)
                || storedPlan.kind != QStringLiteral("dataset_snapshot_import_plan")) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    importFailure(executionError.isEmpty()
                        ? QStringLiteral("dataset_snapshot_import_plan_missing") : executionError)};
            }
            const auto planFile = std::find_if(storedPlan.files.cbegin(), storedPlan.files.cend(),
                [](const ArtifactFileSnapshot& file) {
                    return file.relativePath == QStringLiteral("snapshot_import_plan.json");
                });
            const QString committedPlanPath = QDir(artifactStore_->rootPath()).filePath(
                QStringLiteral("artifacts/%1/snapshot_import_plan.json").arg(planArtifactId.toString()));
            QString actualPlanHash;
            QFile committedPlanFile(committedPlanPath);
            if (planFile == storedPlan.files.cend() || !hashFile(committedPlanPath, &actualPlanHash, &executionError)
                || actualPlanHash != planFile->sha256 || !committedPlanFile.open(QIODevice::ReadOnly)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    importFailure(executionError.isEmpty()
                        ? QStringLiteral("dataset_snapshot_import_plan_tampered") : executionError)};
            }
            const QJsonDocument planDocument = QJsonDocument::fromJson(committedPlanFile.readAll());
            QJsonObject committedPlan = planDocument.object();
            const QString declaredPlanHash = committedPlan.take(QStringLiteral("planHash")).toString();
            const QString calculatedPlanHash = QString::fromLatin1(QCryptographicHash::hash(
                QJsonDocument(committedPlan).toJson(QJsonDocument::Compact),
                QCryptographicHash::Sha256).toHex());
            if (!planDocument.isObject() || declaredPlanHash != calculatedPlanHash
                || committedPlan.value(QStringLiteral("sourceRootHash")).toString()
                    != frozenSnapshot.rootHash) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
                    importFailure(QStringLiteral("dataset_snapshot_import_plan_tampered"))};
            }
            committedPlan.insert(QStringLiteral("planHash"), declaredPlanHash);
            importPlan = committedPlan;
            DatasetSnapshotResult currentSource;
            DatasetSnapshotOptions checkOptions;
            checkOptions.maxFileCount = maxFileCount;
            checkOptions.isCancellationRequested = stepCancellation;
            if (!snapshotSource(driver, sourceInspection, checkOptions, &currentSource, &executionError)
                || currentSource.rootHash != importPlan.value(QStringLiteral("sourceRootHash")).toString()
                || QString::number(currentSource.fileCount)
                    != importPlan.value(QStringLiteral("fileCount")).toString()) {
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_snapshot_import_source_changed");
                const Failure failure = importFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed, {}, failure};
            }
            ArtifactId snapshotArtifactId;
            QString staging;
            if (!artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot"),
                    &snapshotArtifactId, &staging, &executionError)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, importFailure(executionError)};
            }
            const auto abort = [&]() { QString ignored; artifactStore_->abort(staging, &ignored); };
            DatasetInspection stagedInspection;
            DatasetDriverValidationResult stagedValidation;
            DatasetSnapshotOptions snapshotOptions;
            snapshotOptions.maxFileCount = maxFileCount;
            snapshotOptions.isCancellationRequested = stepCancellation;
            snapshotOptions.classDefinitions = importPlan.value(QStringLiteral("classDefinitions")).toArray();
            DatasetSnapshotResult snapshotResult;
            const QString manifestPath = QDir(staging).filePath(QStringLiteral("dataset_snapshot.json"));
            if (!copyFrozenInventory(request.sourcePath, importPlan, staging,
                    stepCancellation, &executionError)
                || !driver->inspect(staging, request.sourceFormat, &stagedInspection, context, &executionError)
                || !driver->validate(stagedInspection, &stagedValidation, context, &executionError)
                || !stagedValidation.valid
                || !createDatasetSnapshot(staging, manifestPath, request.sourceFormat,
                    driver->id(), driver->version(), snapshotOptions, &snapshotResult, &executionError)
                || snapshotResult.rootHash
                    != importPlan.value(QStringLiteral("sourceRootHash")).toString()) {
                abort();
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_snapshot_import_materialized_mismatch");
                const Failure failure = importFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed, {}, failure};
            }
            QString committedPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(snapshotArtifactId, taskId, QStringLiteral("dataset_snapshot"),
                    staging, &storage_, &committedPath, &executionError, stepCancellation,
                    &commitCanceled)) {
                if (QFileInfo::exists(staging)) abort();
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    importFailure(commitCanceled ? QStringLiteral("dataset_snapshot_import_canceled")
                                                 : executionError)};
            }
            ArtifactSnapshot committed;
            if (!storage_.artifact(snapshotArtifactId, &committed, &executionError)) {
                QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, importFailure(executionError)};
            }
            QString manifestHash;
            for (const ArtifactFileSnapshot& file : committed.files) {
                if (file.relativePath == QStringLiteral("dataset_snapshot.json")) manifestHash = file.sha256;
            }
            DatasetSnapshotRecord record;
            record.datasetId = request.targetDatasetId;
            record.id = snapshotResult.snapshotId;
            record.taskId = taskId;
            record.artifactId = snapshotArtifactId;
            record.rootPath = committedPath;
            record.datasetFormat = request.sourceFormat;
            record.driverId = driver->id();
            record.driverVersion = driver->version();
            record.rootHash = snapshotResult.rootHash;
            record.manifestSha256 = manifestHash;
            record.fileCount = snapshotResult.fileCount;
            record.totalBytes = snapshotResult.totalBytes;
            if (aitrain::isCancellationRequested(stepCancellation)
                || manifestHash.isEmpty() || !storage_.registerDatasetSnapshot(&record, &executionError)) {
                QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                const bool wasCanceled = aitrain::isCancellationRequested(stepCancellation);
                return WorkflowStepExecutionResult{
                    wasCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed, {},
                    importFailure(wasCanceled ? QStringLiteral("dataset_snapshot_import_canceled")
                                              : executionError)};
            }
            snapshotBundle.snapshot = record;
            snapshotBundle.artifactPath = committedPath;
            snapshotBundle.manifestPath = QDir(committedPath).filePath(QStringLiteral("dataset_snapshot.json"));
            snapshotBundle.manifest = snapshotResult.manifest;
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded, snapshotArtifactId, {}};
        }
        return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
            importFailure(QStringLiteral("dataset_snapshot_import_unknown_step"))};
    };
    if (!runner.run(workflow.id, executor, &run, error, cancellation)) return false;

    TaskState terminalState = run.state == WorkflowStepState::Succeeded ? TaskState::Succeeded
        : run.state == WorkflowStepState::Canceled ? TaskState::Canceled : TaskState::Failed;
    const Failure terminalFailure = run.failure;
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, terminalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)
        || !commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->terminalState = terminalState;
    result->importPlanArtifactId = planArtifactId;
    result->datasetSnapshot = snapshotBundle.snapshot;
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->failure = terminalFailure;
    result->summary = QJsonObject{{QStringLiteral("sourceFormat"), request.sourceFormat},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("targetDatasetNamePersisted"), false},
        {QStringLiteral("fileCount"), QString::number(snapshotBundle.snapshot.fileCount)},
        {QStringLiteral("totalBytes"), QString::number(snapshotBundle.snapshot.totalBytes)}};
    return true;
}

} // namespace aitrain
