#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/dataset/BuiltinDatasetDrivers.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

namespace aitrain {
namespace {

Failure splitFailure(const QString& message)
{
    Failure failure;
    failure.code = message.contains(QStringLiteral("canceled"), Qt::CaseInsensitive)
            || message.contains(QStringLiteral("已取消"))
        ? FailureCode::Canceled
        : message.contains(QStringLiteral("identity"), Qt::CaseInsensitive)
                || message.contains(QStringLiteral("tamper"), Qt::CaseInsensitive)
                || message.contains(QStringLiteral("hash"), Qt::CaseInsensitive)
                || message.contains(QStringLiteral("changed"), Qt::CaseInsensitive)
        ? FailureCode::ArtifactIncompatible
        : message.contains(QStringLiteral("invalid"), Qt::CaseInsensitive)
        ? FailureCode::InvalidDataset
        : FailureCode::ArtifactIncomplete;
    failure.message = message.isEmpty() ? QStringLiteral("dataset_split_failed") : message;
    failure.suggestedAction = failure.code == FailureCode::Canceled
        ? QStringLiteral("如需继续，请基于同一源快照重新发起拆分。")
        : QStringLiteral("核对源四重身份、Artifact 完整性和拆分 Evidence 后重试。");
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
        if (error) *error = QStringLiteral("dataset_split_artifact_file_unreadable:%1").arg(path);
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("dataset_split_artifact_file_read_failed:%1").arg(path);
            return false;
        }
        digest.addData(block);
    }
    *hash = QString::fromLatin1(digest.result().toHex());
    return true;
}

QString jsonHash(QJsonObject object)
{
    object.remove(QStringLiteral("planHash"));
    return QString::fromLatin1(QCryptographicHash::hash(
        QJsonDocument(object).toJson(QJsonDocument::Compact),
        QCryptographicHash::Sha256).toHex());
}

bool containsAbsolutePath(const QJsonValue& value)
{
    if (value.isString()) {
        const QString text = value.toString();
        return QDir::isAbsolutePath(text)
            || (text.size() > 2 && text.at(1) == QLatin1Char(':')
                && (text.at(2) == QLatin1Char('/') || text.at(2) == QLatin1Char('\\')));
    }
    if (value.isArray()) {
        for (const QJsonValue& item : value.toArray()) {
            if (containsAbsolutePath(item)) return true;
        }
    } else if (value.isObject()) {
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it) {
            if (containsAbsolutePath(it.value())) return true;
        }
    }
    return false;
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    QSaveFile file(path);
    const QByteArray bytes = QJsonDocument(object).toJson(QJsonDocument::Indented);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("dataset_split_plan_write_failed:%1").arg(file.errorString());
        return false;
    }
    return true;
}

bool verifyArtifactInventory(const ArtifactSnapshot& artifact,
    const QString& root,
    QString* error)
{
    if (artifact.files.isEmpty() || !QFileInfo(root).isDir()) {
        if (error) *error = QStringLiteral("dataset_split_source_artifact_incomplete");
        return false;
    }
    for (const ArtifactFileSnapshot& expected : artifact.files) {
        const QString relative = QDir::cleanPath(expected.relativePath);
        const QString path = QDir(root).filePath(relative);
        const QFileInfo info(path);
        QString actualHash;
        if (!safeRelativePath(relative) || !info.isFile() || info.isSymLink()
            || info.size() != expected.byteCount || !isSha256Hex(expected.sha256)
            || !hashFile(path, &actualHash, error) || actualHash != expected.sha256) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("dataset_split_source_artifact_tampered:%1").arg(relative);
            }
            return false;
        }
    }
    return true;
}

bool verifySnapshotManifest(const DatasetSnapshotRecord& snapshot,
    const ArtifactSnapshot& artifact,
    const QString& root,
    QString* error)
{
    QFile file(QDir(root).filePath(QStringLiteral("dataset_snapshot.json")));
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("dataset_split_source_manifest_unreadable");
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    const QJsonObject manifest = document.object();
    if (parseError.error != QJsonParseError::NoError || !document.isObject()
        || manifest.value(QStringLiteral("snapshotId")).toString() != snapshot.id.toString()
        || manifest.value(QStringLiteral("datasetFormat")).toString() != snapshot.datasetFormat
        || manifest.value(QStringLiteral("rootHash")).toString() != snapshot.rootHash) {
        if (error) *error = QStringLiteral("dataset_split_source_manifest_identity_mismatch");
        return false;
    }
    QHash<QString, ArtifactFileSnapshot> inventory;
    for (const ArtifactFileSnapshot& item : artifact.files) inventory.insert(item.relativePath, item);
    const QJsonArray files = manifest.value(QStringLiteral("files")).toArray();
    if (files.size() != snapshot.fileCount) {
        if (error) *error = QStringLiteral("dataset_split_source_manifest_inventory_mismatch");
        return false;
    }
    for (const QJsonValue& value : files) {
        const QJsonObject item = value.toObject();
        const QString relative = item.value(QStringLiteral("relativePath")).toString();
        if (!inventory.contains(relative)
            || inventory.value(relative).sha256 != item.value(QStringLiteral("sha256")).toString()
            || inventory.value(relative).byteCount != item.value(QStringLiteral("bytes")).toString().toLongLong()) {
            if (error) *error = QStringLiteral("dataset_split_source_manifest_inventory_mismatch:%1").arg(relative);
            return false;
        }
    }
    return true;
}

bool copyPureSplitTree(const ArtifactSnapshot& stored,
    const QString& splitRoot,
    const QString& snapshotStaging,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    int copied = 0;
    for (const ArtifactFileSnapshot& expected : stored.files) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (error) *error = QStringLiteral("dataset_split_canceled");
            return false;
        }
        const QString relative = QDir::cleanPath(expected.relativePath);
        if (relative == QStringLiteral("split_plan.json")
            || relative == QStringLiteral("split_plan.json")) continue;
        const QString source = QDir(splitRoot).filePath(relative);
        const QFileInfo info(source);
        QString actualHash;
        if (!safeRelativePath(relative) || relative == QStringLiteral("dataset_snapshot.json")
            || !info.isFile() || info.isSymLink() || info.size() != expected.byteCount
            || !hashFile(source, &actualHash, error) || actualHash != expected.sha256) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("dataset_split_artifact_tampered:%1").arg(relative);
            }
            return false;
        }
        const QString target = QDir(snapshotStaging).filePath(relative);
        if (!QDir().mkpath(QFileInfo(target).absolutePath()) || QFileInfo::exists(target)
            || !QFile::copy(source, target)) {
            if (error) *error = QStringLiteral("dataset_split_snapshot_copy_failed:%1").arg(relative);
            return false;
        }
        ++copied;
    }
    if (copied == 0) {
        if (error) *error = QStringLiteral("dataset_split_snapshot_empty");
        return false;
    }
    return true;
}

} // namespace

bool ProjectWorkspace::runDatasetSplitWorkflow(const TaskId& taskId,
    const DatasetSplitWorkflowRequest& request,
    DatasetSplitWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen() || !taskId.isValid() || !result
        || !request.sourceDatasetId.isValid() || !request.sourceDatasetVersionId.isValid()
        || !request.sourceSnapshotId.isValid() || !request.sourceSnapshotArtifactId.isValid()
        || !request.targetDatasetId.isValid() || request.targetDatasetName.trimmed().isEmpty()
        || containsAbsolutePath(request.options)) {
        if (error) *error = QStringLiteral("dataset_split_invalid_request");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) return false;

    DatasetDriverRegistry drivers;
    if (!registerBuiltinDatasetDrivers(&drivers, error)) return false;
    DatasetSnapshotRecord sourceSnapshot;
    ArtifactSnapshot sourceArtifact;
    const DatasetDriver* driver = nullptr;

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("dataset_split");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    workflow.createdAt = QDateTime::currentDateTimeUtc();
    const QJsonObject parameters{
        {QStringLiteral("sourceDatasetId"), request.sourceDatasetId.toString()},
        {QStringLiteral("sourceDatasetVersionId"), request.sourceDatasetVersionId.toString()},
        {QStringLiteral("sourceSnapshotId"), request.sourceSnapshotId.toString()},
        {QStringLiteral("sourceSnapshotArtifactId"), request.sourceSnapshotArtifactId.toString()},
        {QStringLiteral("targetDatasetId"), request.targetDatasetId.toString()},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("options"), request.options}};
    QVector<WorkflowStepSnapshot> steps;
    const QStringList kinds{QStringLiteral("PlanSplit"), QStringLiteral("MaterializeSplit"),
        QStringLiteral("RegisterSnapshot")};
    for (int ordinal = 0; ordinal < kinds.size(); ++ordinal) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = kinds.at(ordinal);
        step.backend = ordinal == 0 ? QStringLiteral("dataset_snapshot_identity")
            : ordinal == 1 ? QStringLiteral("dataset_split_materializer")
                           : QStringLiteral("dataset_snapshot_registration");
        step.parameterSummary = parameters;
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;

    DatasetSplitPlan driverPlan;
    ArtifactId planArtifactId;
    ArtifactId splitArtifactId;
    DatasetSnapshotArtifactBundle registeredSnapshot;
    QString executionError;
    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult runResult;
    const auto resolveSourceIdentity = [&]() {
        if (!storage_.datasetSnapshot(request.sourceSnapshotId, &sourceSnapshot, &executionError)
            || sourceSnapshot.datasetId != request.sourceDatasetId
            || sourceSnapshot.datasetVersionId != request.sourceDatasetVersionId
            || sourceSnapshot.artifactId != request.sourceSnapshotArtifactId
            || !storage_.artifact(request.sourceSnapshotArtifactId, &sourceArtifact, &executionError)
            || sourceArtifact.kind != QStringLiteral("dataset_snapshot")) {
            if (executionError.isEmpty()) executionError = QStringLiteral("dataset_split_source_identity_mismatch");
            return false;
        }
        QString storedManifestSha256;
        for (const ArtifactFileSnapshot& file : sourceArtifact.files) {
            if (file.relativePath == QStringLiteral("dataset_snapshot.json")) {
                storedManifestSha256 = file.sha256;
                break;
            }
        }
        driver = drivers.driverForFormat(sourceSnapshot.datasetFormat);
        if (storedManifestSha256.isEmpty()
            || storedManifestSha256 != sourceSnapshot.manifestSha256
            || !driver || driver->id() != sourceSnapshot.driverId
            || driver->version() != sourceSnapshot.driverVersion) {
            executionError = QStringLiteral("dataset_split_source_manifest_or_driver_identity_mismatch");
            return false;
        }
        return true;
    };
    const auto executor = [&](const WorkflowStepSnapshot& step,
                              const aitrain::CancellationCallback& stepCancellation) {
        if (aitrain::isCancellationRequested(stepCancellation)) {
            return WorkflowStepExecutionResult{WorkflowStepState::Canceled, {},
                splitFailure(QStringLiteral("dataset_split_canceled"))};
        }
        if (step.kind == QStringLiteral("PlanSplit")) {
            if (!resolveSourceIdentity()) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            DatasetOperationContext context;
            context.isCancellationRequested = stepCancellation;
            DatasetInspection inspection;
    DatasetDriverValidationResult validation;
            const QString sourceRoot = artifactStore_->artifactPath(sourceSnapshot.artifactId);
            if (sourceRoot.isEmpty()
                || !verifyArtifactInventory(sourceArtifact, sourceRoot, &executionError)
                || !verifySnapshotManifest(sourceSnapshot, sourceArtifact, sourceRoot, &executionError)
                || !driver->inspect(sourceRoot, sourceSnapshot.datasetFormat,
                    &inspection, context, &executionError)
                || !driver->validate(inspection, &validation, context, &executionError)
                || !validation.valid
                || !driver->planSplit(inspection, request.options, &driverPlan,
                    context, &executionError)
                || driverPlan.manifest.contains(QStringLiteral("sourceRoot"))
                || containsAbsolutePath(driverPlan.manifest)) {
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_split_source_hash_or_plan_invalid");
                const Failure failure = splitFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            QJsonObject persistedPlan{
                {QStringLiteral("schemaVersion"), 2},
                {QStringLiteral("sourceDatasetId"), request.sourceDatasetId.toString()},
                {QStringLiteral("sourceDatasetVersionId"), request.sourceDatasetVersionId.toString()},
                {QStringLiteral("sourceSnapshotId"), request.sourceSnapshotId.toString()},
                {QStringLiteral("sourceSnapshotArtifactId"), request.sourceSnapshotArtifactId.toString()},
                {QStringLiteral("sourceRootHash"), sourceSnapshot.rootHash},
                {QStringLiteral("format"), sourceSnapshot.datasetFormat},
                {QStringLiteral("driverId"), driver->id()},
                {QStringLiteral("driverVersion"), driver->version()},
                {QStringLiteral("targetDatasetId"), request.targetDatasetId.toString()},
                {QStringLiteral("targetDatasetName"), request.targetDatasetName},
                {QStringLiteral("options"), request.options},
                {QStringLiteral("driverPlan"), driverPlan.manifest}};
            persistedPlan.insert(QStringLiteral("planHash"), jsonHash(persistedPlan));
            QString staging;
            if (!artifactStore_->begin(taskId, QStringLiteral("dataset_split_plan"),
                    &planArtifactId, &staging, &executionError)
                || !writeJson(QDir(staging).filePath(QStringLiteral("dataset_split_plan.json")),
                    persistedPlan, &executionError)) {
                if (!staging.isEmpty()) { QString ignored; artifactStore_->abort(staging, &ignored); }
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            QString ignoredPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(planArtifactId, taskId, QStringLiteral("dataset_split_plan"),
                    staging, &storage_, &ignoredPath, &executionError, stepCancellation, &commitCanceled)) {
                if (QFileInfo::exists(staging)) { QString ignored; artifactStore_->abort(staging, &ignored); }
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, splitFailure(commitCanceled ? QStringLiteral("dataset_split_canceled") : executionError)};
            }
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded, planArtifactId, {}};
        }
        if (step.kind == QStringLiteral("MaterializeSplit")) {
            if (!resolveSourceIdentity()) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            if (!planArtifactId.isValid()) planArtifactId = step.inputArtifactId;
            ArtifactSnapshot planStored;
            if (!storage_.artifact(planArtifactId, &planStored, &executionError)
                || planStored.kind != QStringLiteral("dataset_split_plan")
                || planStored.files.size() != 1
                || planStored.files.first().relativePath != QStringLiteral("dataset_split_plan.json")) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(
                    QStringLiteral("dataset_split_plan_artifact_invalid"))};
            }
            const QString planRoot = artifactStore_->artifactPath(planArtifactId);
            const QString planPath = QDir(planRoot).filePath(QStringLiteral("dataset_split_plan.json"));
            QString planFileHash;
            QFile planFile(planPath);
            if (!hashFile(planPath, &planFileHash, &executionError)
                || planFileHash != planStored.files.first().sha256
                || !planFile.open(QIODevice::ReadOnly)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(planFile.readAll(), &parseError);
            QJsonObject persistedPlan = document.object();
            const QString persistedHash = persistedPlan.value(QStringLiteral("planHash")).toString();
            if (parseError.error != QJsonParseError::NoError || !document.isObject()
                || persistedHash.isEmpty() || persistedHash != jsonHash(persistedPlan)
                || containsAbsolutePath(persistedPlan)
                || persistedPlan.value(QStringLiteral("sourceSnapshotArtifactId")).toString()
                    != request.sourceSnapshotArtifactId.toString()) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(
                    QStringLiteral("dataset_split_plan_tampered"))};
            }
            driverPlan.format = sourceSnapshot.datasetFormat;
            driverPlan.sourceRoot = artifactStore_->artifactPath(sourceSnapshot.artifactId);
            driverPlan.manifest = persistedPlan.value(QStringLiteral("driverPlan")).toObject();
            driverPlan.planHash = driverPlan.manifest.value(QStringLiteral("planHash")).toString();
            QString staging;
            if (!artifactStore_->begin(taskId, QStringLiteral("dataset_split"),
                    &splitArtifactId, &staging, &executionError)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            const auto abort = [&]() { QString ignored; artifactStore_->abort(staging, &ignored); };
            DatasetOperationContext context;
            context.isCancellationRequested = stepCancellation;
            DatasetInspection splitInspection;
    DatasetDriverValidationResult splitValidation;
            if (!driver->materializeSplit(driverPlan, staging, context, &executionError)
                || !driver->inspect(staging, sourceSnapshot.datasetFormat,
                    &splitInspection, context, &executionError)
                || !driver->validate(splitInspection, &splitValidation, context, &executionError)
                || !splitValidation.valid) {
                abort();
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_split_materialized_dataset_invalid");
                const Failure failure = splitFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            QString ignoredPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(splitArtifactId, taskId, QStringLiteral("dataset_split"),
                    staging, &storage_, &ignoredPath, &executionError, stepCancellation, &commitCanceled)) {
                if (QFileInfo::exists(staging)) abort();
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, splitFailure(commitCanceled ? QStringLiteral("dataset_split_canceled") : executionError)};
            }
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded, splitArtifactId, {}};
        }
        if (step.kind == QStringLiteral("RegisterSnapshot")) {
            if (!resolveSourceIdentity()) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
            if (!splitArtifactId.isValid()) splitArtifactId = step.inputArtifactId;
            ArtifactSnapshot splitStored;
            const QString splitRoot = artifactStore_->artifactPath(splitArtifactId);
            ArtifactId snapshotArtifactId;
            QString staging;
            if (!storage_.artifact(splitArtifactId, &splitStored, &executionError)
                || splitStored.kind != QStringLiteral("dataset_split")
                || !artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot"),
                    &snapshotArtifactId, &staging, &executionError)) {
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
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
            if (!copyPureSplitTree(splitStored, splitRoot, staging, stepCancellation, &executionError)
                || !driver->inspect(staging, sourceSnapshot.datasetFormat, &inspection, context, &executionError)
                || !driver->validate(inspection, &validation, context, &executionError)
                || !validation.valid
                || !driver->snapshot(inspection, manifestPath, snapshotOptions,
                    &snapshotResult, &executionError)) {
                abort();
                if (executionError.isEmpty()) executionError = QStringLiteral("dataset_split_snapshot_invalid");
                const Failure failure = splitFailure(executionError);
                return WorkflowStepExecutionResult{
                    failure.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                          : WorkflowStepState::Failed,
                    {}, failure};
            }
            QString finalPath;
            bool commitCanceled = false;
            if (!artifactStore_->commit(snapshotArtifactId, taskId, QStringLiteral("dataset_snapshot"),
                    staging, &storage_, &finalPath, &executionError, stepCancellation, &commitCanceled)) {
                if (QFileInfo::exists(staging)) abort();
                return WorkflowStepExecutionResult{
                    commitCanceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, splitFailure(commitCanceled ? QStringLiteral("dataset_split_canceled") : executionError)};
            }
            ArtifactSnapshot committed;
            QString manifestSha256;
            if (!storage_.artifact(snapshotArtifactId, &committed, &executionError)) {
                QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                return WorkflowStepExecutionResult{WorkflowStepState::Failed, {}, splitFailure(executionError)};
            }
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
            record.datasetFormat = sourceSnapshot.datasetFormat;
            record.driverId = driver->id();
            record.driverVersion = driver->version();
            record.rootHash = snapshotResult.rootHash;
            record.manifestSha256 = manifestSha256;
            record.fileCount = snapshotResult.fileCount;
            record.totalBytes = snapshotResult.totalBytes;
            if (aitrain::isCancellationRequested(stepCancellation)
                || manifestSha256.isEmpty()
                || !storage_.registerDatasetSnapshot(&record, &executionError)) {
                QString ignored; artifactStore_->discardCommitted(snapshotArtifactId, &storage_, &ignored);
                const bool canceled = aitrain::isCancellationRequested(stepCancellation);
                return WorkflowStepExecutionResult{
                    canceled ? WorkflowStepState::Canceled : WorkflowStepState::Failed,
                    {}, splitFailure(canceled ? QStringLiteral("dataset_split_canceled") : executionError)};
            }
            registeredSnapshot.snapshot = record;
            registeredSnapshot.artifactPath = finalPath;
            registeredSnapshot.manifestPath = QDir(finalPath).filePath(QStringLiteral("dataset_snapshot.json"));
            registeredSnapshot.manifest = snapshotResult.manifest;
            return WorkflowStepExecutionResult{WorkflowStepState::Succeeded, snapshotArtifactId, {}};
        }
        return WorkflowStepExecutionResult{WorkflowStepState::Failed, {},
            splitFailure(QStringLiteral("dataset_split_unknown_step"))};
    };
    if (!runner.run(workflow.id, executor, &runResult, error, cancellation)) return false;

    TaskState terminalState = runResult.state == WorkflowStepState::Succeeded
        ? TaskState::Succeeded
        : runResult.state == WorkflowStepState::Canceled ? TaskState::Canceled : TaskState::Failed;
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !requestTaskCancellation(taskId, error)) return false;
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, runResult.failure,
            QDateTime::currentDateTimeUtc(), error)) return false;
    EvidenceBundle evidence;
    EvidenceArtifactBundle committedEvidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)
        || !commitEvidenceBundle(evidence, &committedEvidence, error)
        || !closeWorkflowTerminalization(workflow.id, error)) return false;

    result->workflowRunId = workflow.id;
    result->terminalState = terminalState;
    result->splitPlanArtifactId = planArtifactId;
    result->splitArtifactId = splitArtifactId;
    result->datasetSnapshot = registeredSnapshot.snapshot;
    result->evidenceArtifactId = committedEvidence.artifactId;
    result->failure = runResult.failure;
    result->summary = QJsonObject{
        {QStringLiteral("sourceDatasetId"), request.sourceDatasetId.toString()},
        {QStringLiteral("sourceDatasetVersionId"), request.sourceDatasetVersionId.toString()},
        {QStringLiteral("sourceSnapshotId"), request.sourceSnapshotId.toString()},
        {QStringLiteral("targetDatasetName"), request.targetDatasetName},
        {QStringLiteral("targetDatasetNamePersisted"), false},
        {QStringLiteral("format"), sourceSnapshot.datasetFormat},
        {QStringLiteral("ratios"), QJsonObject{
            {QStringLiteral("train"), request.options.value(QStringLiteral("trainRatio"))},
            {QStringLiteral("val"), request.options.value(QStringLiteral("valRatio"))},
            {QStringLiteral("test"), request.options.value(QStringLiteral("testRatio"))}}}};
    return true;
}

} // namespace aitrain
