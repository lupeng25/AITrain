#include "aitrain/workflow/ProjectWorkspace.h"

#include "aitrain/workflow/EvidenceRenderer.h"
#include "aitrain/workflow/TrainingWorkflowProfile.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>

#include <algorithm>

namespace aitrain {
namespace {

bool isChildPath(const QString& parentPath, const QString& candidatePath);

bool copyDatasetTreeToEmptyStaging(const QString& sourcePath,
    const QString& stagingPath,
    const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    const QDir source(sourcePath);
    const QDir staging(stagingPath);
    if (!source.exists() || source.absolutePath() == staging.absolutePath()
        || isChildPath(source.absolutePath(), staging.absolutePath())
        || isChildPath(staging.absolutePath(), source.absolutePath())) {
        if (error) *error = QStringLiteral("dataset_snapshot_import_path_boundary_invalid");
        return false;
    }
    QDirIterator entries(source.absolutePath(), QDir::AllEntries | QDir::NoDotAndDotDot,
        QDirIterator::Subdirectories);
    while (entries.hasNext()) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (error) *error = QStringLiteral("snapshot_canceled");
            return false;
        }
        const QString absolutePath = entries.next();
        const QFileInfo info(absolutePath);
        if (info.isSymLink()) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_symlink_rejected:%1")
                .arg(source.relativeFilePath(absolutePath));
            return false;
        }
        if (!info.isFile()) continue;
        const QString relativePath = QDir::cleanPath(source.relativeFilePath(absolutePath));
        if (relativePath == QStringLiteral("dataset_snapshot.json")
            || relativePath == QStringLiteral("..")
            || relativePath.startsWith(QStringLiteral("../"))) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_reserved_or_unsafe_path:%1")
                .arg(relativePath);
            return false;
        }
        const QString destination = staging.filePath(relativePath);
        if (!QDir().mkpath(QFileInfo(destination).absolutePath())
            || QFileInfo::exists(destination) || !QFile::copy(absolutePath, destination)) {
            if (error) *error = QStringLiteral("dataset_snapshot_import_copy_failed:%1").arg(relativePath);
            return false;
        }
    }
    return true;
}

bool isChildPath(const QString& parentPath, const QString& candidatePath)
{
    const QString parent = QDir::cleanPath(QDir(parentPath).absolutePath());
    const QString candidate = QDir::cleanPath(QFileInfo(candidatePath).absoluteFilePath());
    return candidate.startsWith(parent + QLatin1Char('/'), Qt::CaseInsensitive);
}

bool writeArtifactFile(const QString& path, const QByteArray& contents, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(contents) != contents.size()
        || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 Evidence 报告文件：%1").arg(path);
        return false;
    }
    return true;
}

QString stableProjectIdentity(const QString& workspacePath)
{
    // Evidence 是可交付报告，不能把工作区绝对路径写入持久化事实。当前
    // schema 尚未提供项目名称/ProjectId 字段，因此以规范工作区路径的
    // SHA-256 生成稳定 opaque identity；同一项目重开仍可关联，报告不会
    // 暴露用户目录、盘符或 UNC 前缀。
    const QString canonical = QDir(workspacePath).absolutePath();
    const QByteArray digest = QCryptographicHash::hash(canonical.toUtf8(), QCryptographicHash::Sha256).toHex();
    return QStringLiteral("project:%1").arg(QString::fromLatin1(digest));
}

bool verifyArtifactFile(const QString& artifactPath,
    const ArtifactFileSnapshot& expected,
    VerifiedWorkflowArtifactFile* result,
    QString* error)
{
    const QString absolutePath = QDir(artifactPath).filePath(expected.relativePath);
    const QFileInfo info(absolutePath);
    if (expected.relativePath.isEmpty() || QDir::isAbsolutePath(expected.relativePath)
        || !isChildPath(artifactPath, absolutePath) || !info.exists() || !info.isFile() || info.isSymLink()
        || info.size() != expected.byteCount) {
        if (error) *error = QStringLiteral("已提交 Artifact 文件无效、越界或已被修改：%1").arg(expected.relativePath);
        return false;
    }
    QFile file(info.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取已提交 Artifact 文件：%1").arg(info.absoluteFilePath());
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("读取已提交 Artifact 文件失败：%1").arg(info.absoluteFilePath());
            return false;
        }
        hash.addData(block);
    }
    if (QString::fromLatin1(hash.result().toHex()) != expected.sha256) {
        if (error) *error = QStringLiteral("已提交 Artifact 文件 SHA-256 不匹配：%1").arg(expected.relativePath);
        return false;
    }
    if (result) {
        *result = {expected.relativePath, info.absoluteFilePath(), expected.sha256, expected.byteCount};
    }
    return true;
}

const VerifiedWorkflowArtifactFile* selectArtifactFile(const VerifiedTrainingWorkflowInput& input,
    const QStringList& preferredRelativePaths)
{
    for (const QString& relativePath : preferredRelativePaths) {
        for (const VerifiedWorkflowArtifactFile& file : input.files) {
            if (file.relativePath == relativePath) return &file;
        }
    }
    return nullptr;
}

bool isDeploymentImageRelativePath(const QString& value)
{
    const QString path = QDir::cleanPath(QDir::fromNativeSeparators(value.trimmed()));
    if (path.isEmpty() || path == QStringLiteral(".") || QDir::isAbsolutePath(path)
        || path == QStringLiteral("..") || path.startsWith(QStringLiteral("../"))) {
        return false;
    }
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (!QStringList{QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("png"),
            QStringLiteral("bmp"), QStringLiteral("tif"), QStringLiteral("tiff"),
            QStringLiteral("webp")}.contains(suffix)) {
        return false;
    }
    const QString normalized = QStringLiteral("/") + path.toLower() + QStringLiteral("/");
    return !normalized.contains(QStringLiteral("/labels/"))
        && !normalized.contains(QStringLiteral("/masks/"))
        && !normalized.contains(QStringLiteral("/annotations/"));
}

const VerifiedWorkflowArtifactFile* selectDeploymentSample(
    const VerifiedTrainingWorkflowInput& snapshotInput,
    const QString& requestedRelativePath,
    QString* error)
{
    const QString requested = QDir::cleanPath(
        QDir::fromNativeSeparators(requestedRelativePath.trimmed()));
    if (!requestedRelativePath.trimmed().isEmpty()) {
        if (!isDeploymentImageRelativePath(requested)) {
            if (error) *error = QStringLiteral("部署样本必须是 Snapshot Artifact 内安全的相对图像路径。");
            return nullptr;
        }
        const auto found = std::find_if(snapshotInput.files.cbegin(), snapshotInput.files.cend(),
            [&requested](const VerifiedWorkflowArtifactFile& file) {
                return QDir::cleanPath(QDir::fromNativeSeparators(file.relativePath)) == requested;
            });
        if (found == snapshotInput.files.cend()) {
            if (error) *error = QStringLiteral("部署样本不属于已绑定的 Snapshot Artifact。");
            return nullptr;
        }
        return &(*found);
    }

    QVector<const VerifiedWorkflowArtifactFile*> candidates;
    for (const VerifiedWorkflowArtifactFile& file : snapshotInput.files) {
        if (isDeploymentImageRelativePath(file.relativePath)) candidates.append(&file);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto* left, const auto* right) {
        const auto score = [](const QString& path) {
            const QString normalized = QStringLiteral("/") + path.toLower() + QStringLiteral("/");
            if (normalized.contains(QStringLiteral("/val/"))) return 0;
            if (normalized.contains(QStringLiteral("/test/"))) return 1;
            if (normalized.contains(QStringLiteral("/images/"))) return 2;
            return 3;
        };
        const int leftScore = score(left->relativePath);
        const int rightScore = score(right->relativePath);
        return leftScore != rightScore ? leftScore < rightScore
            : left->relativePath < right->relativePath;
    });
    if (candidates.isEmpty()) {
        if (error) *error = QStringLiteral("已绑定 Snapshot Artifact 中没有可用于部署验证的图像。");
        return nullptr;
    }
    return candidates.first();
}

const TrainingWorkflowStepProfile* workflowStepProfile(const TrainingWorkflowProfile& profile,
    const QString& kind)
{
    for (const TrainingWorkflowStepProfile& step : profile.steps) {
        if (step.kind == kind) return &step;
    }
    return nullptr;
}

bool datasetSnapshotForWorkflow(const ProjectStore& storage,
    const QVector<WorkflowStepSnapshot>& steps,
    DatasetSnapshotRecord* result,
    QString* error)
{
    if (!result) {
        if (error) *error = QStringLiteral("解析 Workflow 数据集快照需要输出对象。");
        return false;
    }
    const auto snapshotStep = std::find_if(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
        return step.kind == QStringLiteral("CreateSnapshot")
            && step.state == WorkflowStepState::Succeeded
            && step.outputArtifactId.isValid();
    });
    WorkflowInputBinding binding;
    if (snapshotStep == steps.cend()
        || !storage.workflowInput(snapshotStep->workflowRunId, QStringLiteral("dataset_snapshot"),
            &binding, error)
        || binding.sourceArtifactKind != QStringLiteral("dataset_snapshot")
        || snapshotStep->outputArtifactId != binding.sourceArtifactId
        || !storage.datasetSnapshot(binding.datasetSnapshotId, result, error)
        || result->artifactId != binding.sourceArtifactId
        || result->taskId != binding.sourceTaskId
        || result->datasetId != binding.datasetId
        || result->datasetVersionId != binding.datasetVersionId
        || result->manifestSha256 != binding.manifestSha256
        || result->rootHash != binding.rootHash) {
        if (error && error->isEmpty()) *error = QStringLiteral("Workflow 缺少与参数 lineage 一致的可信数据集快照。");
        return false;
    }
    return true;
}

bool projectedTerminalTaskForWorkflow(const TaskSnapshot& persisted,
    const WorkflowTerminalizationSnapshot& terminalization,
    TaskSnapshot* result,
    QString* error)
{
    if (!result || terminalization.taskId != persisted.id
        || terminalization.state == WorkflowTerminalizationState::Closed) {
        if (error) *error = QStringLiteral("推导 Workflow 终态需要同一任务未关闭的终态封存记录。");
        return false;
    }
    TaskSnapshot projected = persisted;
    projected.state = terminalization.terminalState;
    projected.failure = terminalization.failure;
    projected.updatedAt = terminalization.terminalAt;
    *result = projected;
    return true;
}

TaskState taskStateForWorkflowResult(const WorkflowRunExecutionResult& result)
{
    if (result.state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (result.state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

Failure completeWorkflowFailure(TaskState terminalState, Failure failure)
{
    if (terminalState == TaskState::Succeeded) return {};
    if (!failure.isFailure()) {
        failure.code = terminalState == TaskState::Canceled
            ? FailureCode::Canceled : FailureCode::InternalError;
    }
    if (failure.message.trimmed().isEmpty()) {
        failure.message = terminalState == TaskState::Canceled
            ? QStringLiteral("训练 Workflow 已取消。")
            : QStringLiteral("训练 Workflow 失败。");
    }
    if (failure.suggestedAction.trimmed().isEmpty()) {
        failure.suggestedAction = terminalState == TaskState::Canceled
            ? QStringLiteral("如需继续，请重新启动训练任务。")
            : QStringLiteral("检查失败详情、输入数据和运行环境，修复后重新执行。");
    }
    if (!failure.occurredAt.isValid()) failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

bool resolveVerifiedArtifact(const ProjectStore& storage,
    const ArtifactStore* artifactStore,
    const ArtifactId& artifactId,
    VerifiedTrainingWorkflowInput* result,
    QString* error)
{
    if (!artifactStore || !artifactId.isValid() || !result) {
        if (error) *error = QStringLiteral("解析已提交 Artifact 需要有效 Store、ID 和输出对象。");
        return false;
    }
    ArtifactSnapshot artifact;
    if (!storage.artifact(artifactId, &artifact, error)) return false;
    const QString artifactPath = artifactStore->artifactPath(artifact.id);
    if (!QFileInfo(artifactPath).isDir()) {
        if (error) *error = QStringLiteral("已提交 Artifact 目录不存在：%1").arg(artifact.id.toString());
        return false;
    }
    VerifiedTrainingWorkflowInput verified;
    verified.artifactId = artifact.id;
    verified.artifactPath = artifactPath;
    for (const ArtifactFileSnapshot& file : artifact.files) {
        VerifiedWorkflowArtifactFile checked;
        if (!verifyArtifactFile(artifactPath, file, &checked, error)) return false;
        verified.files.append(checked);
    }
    if (verified.files.isEmpty()) {
        if (error) *error = QStringLiteral("已提交 Artifact 不包含可验证文件。");
        return false;
    }
    *result = verified;
    return true;
}

bool parseTensorContracts(const QJsonValue& value, QVector<TensorContract>* result, QString* error)
{
    if (!result || !value.isArray()) {
        if (error) *error = QStringLiteral("官方导出 sidecar 的张量合同必须为数组。");
        return false;
    }
    QVector<TensorContract> tensors;
    for (const QJsonValue& item : value.toArray()) {
        const QJsonObject object = item.toObject();
        if (!item.isObject() || !object.value(QStringLiteral("shape")).isArray()) {
            if (error) *error = QStringLiteral("官方导出 sidecar 的张量合同格式无效。");
            return false;
        }
        TensorContract tensor;
        tensor.name = object.value(QStringLiteral("name")).toString().trimmed();
        tensor.layout = object.value(QStringLiteral("layout")).toString().trimmed();
        for (const QJsonValue& dimension : object.value(QStringLiteral("shape")).toArray()) {
            if (!dimension.isDouble()) {
                if (error) *error = QStringLiteral("官方导出 sidecar 的张量 shape 必须为整数。");
                return false;
            }
            const qint64 parsed = static_cast<qint64>(dimension.toDouble());
            if (parsed == 0 || parsed < -1 || static_cast<double>(parsed) != dimension.toDouble()) {
                if (error) *error = QStringLiteral("官方导出 sidecar 的张量 shape 仅允许正数或 -1。");
                return false;
            }
            tensor.shape.append(parsed);
        }
        if (tensor.name.isEmpty() || tensor.layout.isEmpty() || tensor.shape.isEmpty()) {
            if (error) *error = QStringLiteral("官方导出 sidecar 的张量合同缺少 name、layout 或 shape。");
            return false;
        }
        tensors.append(tensor);
    }
    if (tensors.isEmpty()) {
        if (error) *error = QStringLiteral("官方导出 sidecar 未提供张量合同。");
        return false;
    }
    *result = tensors;
    return true;
}

bool modelManifestFromTrainingExportSidecar(const QJsonObject& sidecar,
    const TrainingWorkflowProfile& profile,
    const TaskId& taskId,
    const SnapshotId& snapshotId,
    const VerifiedWorkflowArtifactFile& exportedModel,
    ModelManifest* result,
    QString* error)
{
    if (!result || sidecar.value(QStringLiteral("backend")).toString() != profile.exportBackend
        || sidecar.value(QStringLiteral("format")).toString() != QStringLiteral("onnx")) {
        if (error) *error = QStringLiteral("RegisterModel 仅接受当前训练 Profile 声明的 ONNX export sidecar。");
        return false;
    }
    const QJsonObject contract = sidecar.value(QStringLiteral("modelContract")).toObject();
    ModelManifest manifest;
    manifest.modelPackageId = ModelPackageId::create();
    manifest.modelFamily = contract.value(QStringLiteral("modelFamily")).toString();
    manifest.taskType = contract.value(QStringLiteral("taskType")).toString();
    manifest.sourceBackend = sidecar.value(QStringLiteral("backend")).toString();
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = snapshotId;
    manifest.sourceArtifactSha256 = exportedModel.sha256;
    manifest.artifactEntryPath = exportedModel.relativePath;
    if (!parseTensorContracts(contract.value(QStringLiteral("inputs")), &manifest.inputs, error)
        || !parseTensorContracts(contract.value(QStringLiteral("outputs")), &manifest.outputs, error)
        || !contract.value(QStringLiteral("preprocessing")).isObject()
        || !contract.value(QStringLiteral("postprocessing")).isObject()
        || !contract.value(QStringLiteral("classNames")).isArray()
        || !contract.value(QStringLiteral("runtimeRoutes")).isArray()) {
        if (error && error->isEmpty()) *error = QStringLiteral("官方导出 sidecar 缺少  Model Manifest 合同。");
        return false;
    }
    manifest.preprocessing = contract.value(QStringLiteral("preprocessing")).toObject();
    manifest.postprocessing = contract.value(QStringLiteral("postprocessing")).toObject();
    manifest.decoder = contract.value(QStringLiteral("decoder")).toString();
    for (const QJsonValue& value : contract.value(QStringLiteral("classNames")).toArray()) manifest.classNames.append(value.toString());
    for (const QJsonValue& value : contract.value(QStringLiteral("runtimeRoutes")).toArray()) manifest.runtimeRoutes.append(value.toString());
    manifest.opset = sidecar.value(QStringLiteral("opset")).toInt(
        profile.modelFamily.startsWith(QStringLiteral("yolo_")) ? 17 : 0);
    manifest.exporterVersion = sidecar.value(QStringLiteral("exporterVersion")).toString();
    if (manifest.exporterVersion.isEmpty()) {
        manifest.exporterVersion = sidecar.value(QStringLiteral("ultralyticsVersion")).toString();
    }
    manifest.verified = true;
    manifest.limitations = profile.limitations;
    if (!validateModelManifest(manifest, error)) return false;
    *result = manifest;
    return true;
}

bool modelManifestFromAnomalibBundleSidecar(const QJsonObject& sidecar,
    const TrainingWorkflowProfile& profile,
    const TaskId& taskId,
    const SnapshotId& snapshotId,
    const VerifiedWorkflowArtifactFile& sidecarFile,
    ModelManifest* result,
    QString* error)
{
    if (!result || sidecar.value(QStringLiteral("kind")).toString() != QStringLiteral("anomalib_bundle")
        || sidecar.value(QStringLiteral("artifactFormat")).toString() != QStringLiteral("anomalib_bundle")
        || sidecar.value(QStringLiteral("modelFamily")).toString() != profile.modelFamily
        || sidecar.value(QStringLiteral("taskType")).toString() != profile.capabilityTaskType
        || sidecar.value(QStringLiteral("checkpointPath")).toString() != QStringLiteral("model.ckpt")) {
        if (error) *error = QStringLiteral("RegisterModel 需要规范化且使用包内相对 checkpoint 的 Anomalib bundle sidecar。");
        return false;
    }
    ModelManifest manifest;
    manifest.modelPackageId = ModelPackageId::create();
    manifest.modelFamily = sidecar.value(QStringLiteral("modelFamily")).toString();
    manifest.taskType = sidecar.value(QStringLiteral("taskType")).toString();
    manifest.sourceBackend = sidecar.value(QStringLiteral("sourceTrainingBackend")).toString();
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = snapshotId;
    manifest.sourceArtifactSha256 = sidecarFile.sha256;
    manifest.artifactEntryPath = sidecarFile.relativePath;
    manifest.artifactFormat = QStringLiteral("anomalib_bundle");
    manifest.preprocessing = sidecar.value(QStringLiteral("preprocessing")).toObject();
    manifest.postprocessing = sidecar.value(QStringLiteral("postprocessing")).toObject();
    manifest.decoder = sidecar.value(QStringLiteral("decoder")).toString();
    for (const QJsonValue& value : sidecar.value(QStringLiteral("classNames")).toArray()) {
        manifest.classNames.append(value.toString());
    }
    for (const QJsonValue& value : sidecar.value(QStringLiteral("runtimeRoutes")).toArray()) {
        manifest.runtimeRoutes.append(value.toString());
    }
    manifest.opset = 0;
    manifest.exporterVersion = sidecar.value(QStringLiteral("exporterVersion")).toString();
    manifest.verified = true;
    manifest.limitations = profile.limitations;
    if (manifest.sourceBackend != profile.trainingBackend || !validateModelManifest(manifest, error)) return false;
    *result = manifest;
    return true;
}

bool modelManifestFromPaddleOcrBundleSidecar(const QJsonObject& sidecar,
    const TrainingWorkflowProfile& profile,
    const TaskId& taskId,
    const SnapshotId& snapshotId,
    const VerifiedWorkflowArtifactFile& sidecarFile,
    const VerifiedWorkflowArtifactFile& bundleFile,
    ModelManifest* result,
    QString* error)
{
    const qint64 declaredBytes = static_cast<qint64>(sidecar.value(QStringLiteral("bundleByteCount")).toDouble(-1));
    if (!result || sidecar.value(QStringLiteral("kind")).toString() != QStringLiteral("paddleocr_bundle")
        || sidecar.value(QStringLiteral("artifactFormat")).toString() != QStringLiteral("paddleocr_inference_bundle")
        || sidecar.value(QStringLiteral("modelFamily")).toString() != profile.modelFamily
        || sidecar.value(QStringLiteral("taskType")).toString() != profile.capabilityTaskType
        || sidecar.value(QStringLiteral("sourceTrainingBackend")).toString() != profile.trainingBackend
        || sidecar.value(QStringLiteral("bundleSha256")).toString() != bundleFile.sha256
        || declaredBytes != bundleFile.byteCount) {
        if (error) *error = QStringLiteral("RegisterModel 需要与已提交 ZIP 哈希和大小一致的 PaddleOCR official bundle sidecar。");
        return false;
    }
    ModelManifest manifest;
    manifest.modelPackageId = ModelPackageId::create();
    manifest.modelFamily = sidecar.value(QStringLiteral("modelFamily")).toString();
    manifest.taskType = sidecar.value(QStringLiteral("taskType")).toString();
    manifest.sourceBackend = sidecar.value(QStringLiteral("sourceTrainingBackend")).toString();
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = snapshotId;
    manifest.sourceArtifactSha256 = sidecarFile.sha256;
    manifest.artifactEntryPath = sidecarFile.relativePath;
    manifest.artifactFormat = QStringLiteral("paddleocr_inference_bundle");
    manifest.preprocessing = sidecar.value(QStringLiteral("preprocessing")).toObject();
    manifest.postprocessing = sidecar.value(QStringLiteral("postprocessing")).toObject();
    manifest.decoder = sidecar.value(QStringLiteral("decoder")).toString();
    for (const QJsonValue& value : sidecar.value(QStringLiteral("classNames")).toArray()) {
        manifest.classNames.append(value.toString());
    }
    for (const QJsonValue& value : sidecar.value(QStringLiteral("runtimeRoutes")).toArray()) {
        manifest.runtimeRoutes.append(value.toString());
    }
    manifest.opset = 0;
    manifest.exporterVersion = sidecar.value(QStringLiteral("exporterVersion")).toString();
    manifest.verified = true;
    manifest.limitations = profile.limitations;
    if (!validateModelManifest(manifest, error)) return false;
    *result = manifest;
    return true;
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0
        || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入训练 Workflow Adapter 请求：%1").arg(path);
        return false;
    }
    return true;
}

QJsonObject artifactFacts(const ArtifactSnapshot& artifact)
{
    QCryptographicHash inventory(QCryptographicHash::Sha256);
    qint64 totalBytes = 0;
    for (const ArtifactFileSnapshot& file : artifact.files) {
        inventory.addData(file.relativePath.toUtf8());
        inventory.addData("\0", 1);
        inventory.addData(file.sha256.toLatin1());
        inventory.addData("\0", 1);
        inventory.addData(QByteArray::number(file.byteCount));
        inventory.addData("\n", 1);
        totalBytes += file.byteCount;
    }
    return {{QStringLiteral("fileCount"), artifact.files.size()},
        {QStringLiteral("totalBytes"), static_cast<double>(totalBytes)},
        {QStringLiteral("inventorySha256"), QString::fromLatin1(inventory.result().toHex())}};
}

WorkflowStepExecutionResult workflowExecutionForAdapterTerminal(const ProtocolEnvelope& event,
    const ArtifactId& outputArtifactId)
{
    WorkflowStepExecutionResult execution;
    if (event.kind == QStringLiteral("event.succeeded")) {
        if (outputArtifactId.isValid()) {
            execution.state = WorkflowStepState::Succeeded;
            execution.outputArtifactId = outputArtifactId;
            return execution;
        }
        execution.state = WorkflowStepState::Failed;
        execution.failure = {FailureCode::ArtifactIncomplete,
            QStringLiteral("官方 Adapter 声明成功，但没有已提交的输出 Artifact。"), {}, event.timestamp};
        return execution;
    }
    if (event.kind == QStringLiteral("event.canceled")) {
        execution.state = WorkflowStepState::Canceled;
        execution.failure = {FailureCode::Canceled,
            event.payload.value(QStringLiteral("message")).toString(), {}, event.timestamp};
        return execution;
    }
    execution.state = WorkflowStepState::Failed;
    FailureCode code = FailureCode::InternalError;
    if (!failureCodeFromString(event.payload.value(QStringLiteral("failureCode")).toString(), &code)
        || code == FailureCode::None) {
        code = FailureCode::InternalError;
    }
    QString message = event.payload.value(QStringLiteral("message")).toString();
    const QString adapterCode = event.payload.value(QStringLiteral("adapterCode")).toString();
    if (!adapterCode.isEmpty()) {
        message = QStringLiteral("[%1] %2").arg(adapterCode, message);
    }
    execution.failure = {code, message, {}, event.timestamp};
    return execution;
}

} // namespace

ProjectWorkspace::ProjectWorkspace() = default;

bool ProjectWorkspace::recoverEvidenceGatedWorkflows(QString* error)
{
    const QVector<WorkflowRunSnapshot> workflows = storage_.pendingEvidenceRequiredWorkflows(1000, error);
    if (error && !error->isEmpty()) return false;
    const QVector<WorkflowTerminalizationSnapshot> pending = storage_.pendingWorkflowTerminalizations(1000, error);
    if (error && !error->isEmpty()) return false;
    QHash<QString, WorkflowTerminalizationSnapshot> terminalizations;
    for (const WorkflowTerminalizationSnapshot& terminalization : pending) {
        terminalizations.insert(terminalization.workflowRunId.toString(), terminalization);
    }

    for (const WorkflowRunSnapshot& workflow : workflows) {
        TaskSnapshot task;
        if (!storage_.task(workflow.taskId, &task, error)) return false;
        WorkflowTerminalizationSnapshot terminalization = terminalizations.value(workflow.id.toString());
        if (!terminalization.workflowRunId.isValid()) {
            WorkflowRunner runner(&storage_);
            WorkflowStepDispatch dispatch;
            const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflow.id, error);
            if (error && !error->isEmpty()) return false;
            const auto running = std::find_if(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
                return step.state == WorkflowStepState::Running;
            });
            if (running != steps.cend()) {
                WorkflowStepExecutionResult interrupted;
                interrupted.state = task.state == TaskState::CancelRequested
                    ? WorkflowStepState::Canceled : WorkflowStepState::Failed;
                interrupted.failure = completeWorkflowFailure(
                    interrupted.state == WorkflowStepState::Canceled ? TaskState::Canceled : TaskState::Failed,
                    {interrupted.state == WorkflowStepState::Canceled ? FailureCode::Canceled : FailureCode::ProcessCrashed,
                        interrupted.state == WorkflowStepState::Canceled
                            ? QStringLiteral("应用关闭前已请求取消训练 Workflow。")
                            : QStringLiteral("应用在训练 Workflow 步骤运行期间关闭。"),
                        {}, QDateTime::currentDateTimeUtc()});
                if (!runner.completeStep(workflow.id, running->id, interrupted, &dispatch, error)) return false;
            } else {
                if (!runner.beginNextStep(workflow.id, &dispatch, error)) return false;
                if (dispatch.hasStep) {
                    WorkflowStepExecutionResult interrupted;
                    interrupted.state = task.state == TaskState::CancelRequested
                        ? WorkflowStepState::Canceled : WorkflowStepState::Failed;
                    interrupted.failure = completeWorkflowFailure(
                        interrupted.state == WorkflowStepState::Canceled ? TaskState::Canceled : TaskState::Failed,
                        {interrupted.state == WorkflowStepState::Canceled ? FailureCode::Canceled : FailureCode::ProcessCrashed,
                            interrupted.state == WorkflowStepState::Canceled
                                ? QStringLiteral("应用关闭前已请求取消训练 Workflow。")
                                : QStringLiteral("应用在训练 Workflow 派发期间关闭。"),
                            {}, QDateTime::currentDateTimeUtc()});
                    if (!runner.completeStep(workflow.id, dispatch.step.id, interrupted, &dispatch, error)) return false;
                }
            }
            if (dispatch.hasStep || !isTerminalWorkflowStepState(dispatch.result.state)) {
                if (error) *error = QStringLiteral("恢复 Evidence 门控工作流时未形成完整步骤终态。");
                return false;
            }
            TaskState terminalState = taskStateForWorkflowResult(dispatch.result);
            Failure terminalFailure = completeWorkflowFailure(terminalState, dispatch.result.failure);
            if (task.state == TaskState::CancelRequested && terminalState == TaskState::Succeeded) {
                terminalState = TaskState::Canceled;
                terminalFailure = completeWorkflowFailure(TaskState::Canceled,
                    {FailureCode::Canceled, QStringLiteral("应用关闭前已请求取消训练 Workflow。"), {},
                        QDateTime::currentDateTimeUtc()});
            }
            if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, terminalFailure,
                    QDateTime::currentDateTimeUtc(), error)
                || !storage_.workflowTerminalization(workflow.id, &terminalization, error)) return false;
        }

        if (terminalization.state == WorkflowTerminalizationState::Sealed) {
            EvidenceBundle evidence;
            EvidenceArtifactBundle committed;
            QString evidenceError;
            if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, &evidenceError)
                || !commitEvidenceBundle(evidence, &committed, &evidenceError)) {
                const Failure failure{FailureCode::ArtifactIncomplete,
                    QStringLiteral("恢复训练 Workflow 时提交 Evidence 失败：%1").arg(evidenceError),
                    QStringLiteral("检查 Artifact Store 与磁盘状态后重新打开项目重试。"),
                    QDateTime::currentDateTimeUtc()};
                QString recordError;
                storage_.recordWorkflowTerminalizationEvidenceFailure(workflow.id, failure, &recordError);
                if (error) *error = recordError.isEmpty() ? failure.message
                    : QStringLiteral("%1；记录失败尝试也失败：%2").arg(failure.message, recordError);
                return false;
            }
            if (!storage_.workflowTerminalization(workflow.id, &terminalization, error)) return false;
        }
        if (terminalization.state == WorkflowTerminalizationState::EvidenceAttached
            && !closeWorkflowTerminalization(workflow.id, error)) return false;
    }
    return true;
}

bool ProjectWorkspace::open(const QString& projectRoot, QString* error)
{
    close();
    const QString normalizedRoot = QDir::cleanPath(QDir::fromNativeSeparators(projectRoot.trimmed()));
    if (normalizedRoot.isEmpty()) {
        if (error) *error = QStringLiteral("打开  项目工作区需要项目根目录。");
        return false;
    }
    const QFileInfo rootInfo(normalizedRoot);
    if (rootInfo.exists() && (!rootInfo.isDir() || rootInfo.isSymLink())) {
        if (error) *error = QStringLiteral(" 项目根目录必须是非符号链接目录。");
        return false;
    }
    const QString candidate = QDir(normalizedRoot).filePath(QStringLiteral(".aitrain"));
    if (!QDir().mkpath(candidate)) {
        if (error) *error = QStringLiteral("无法创建  项目工作区：%1").arg(candidate);
        return false;
    }
    const QString artifactRoot = QDir(candidate).filePath(QStringLiteral("artifacts"));
    const QStringList workspaceDirectories = {
        artifactRoot,
        QDir(artifactRoot).filePath(QStringLiteral(".staging")),
        QDir(artifactRoot).filePath(QStringLiteral(".staging-meta")),
        QDir(artifactRoot).filePath(QStringLiteral("committed")),
        QDir(candidate).filePath(QStringLiteral(".runtime-staging"))
    };
    for (const QString& directory : workspaceDirectories) {
        if (!QDir().mkpath(directory)) {
            if (error) *error = QStringLiteral("无法创建项目工作区目录：%1").arg(directory);
            return false;
        }
    }
    if (!storage_.open(QDir(candidate).filePath(QStringLiteral("project.sqlite")), error)) {
        close();
        return false;
    }
    artifactStore_ = std::make_unique<ArtifactStore>(QDir(candidate).filePath(QStringLiteral("artifacts")));
    storage_.setArtifactStoreRoot(artifactStore_->rootPath());
    workspacePath_ = candidate;
    QStringList diagnostics;
    if (!artifactStore_->recoverStaging(&storage_, &diagnostics, error)
        || !recoverEvidenceGatedWorkflows(error)
        || !storage_.markInterruptedTasksFailed(error)) {
        close();
        return false;
    }
    taskCoordinator_ = std::make_unique<TaskCoordinator>(&storage_);
    trainingAdapterHost_ = std::make_unique<TaskExecutionHost>(taskCoordinator_.get(), artifactStore_.get());
    return true;
}

void ProjectWorkspace::close()
{
    trainingAdapterHost_.reset();
    taskCoordinator_.reset();
    artifactStore_.reset();
    storage_.close();
    workspacePath_.clear();
}

bool ProjectWorkspace::isOpen() const
{
    return storage_.isOpen() && artifactStore_ != nullptr;
}

bool ProjectWorkspace::importModel(const ModelImportRequest& request,
    ModelImportResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    TaskCoordinator coordinator(&storage_);
    ModelImportService importer(&coordinator, artifactStore_.get());
    return importer.importModel(request, result, error, cancellation);
}

bool ProjectWorkspace::startTask(const TaskId& taskId,
    const QString& capabilityId,
    const QString& taskType,
    TaskSnapshot* task,
    QString* error)
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    return taskCoordinator_ && taskCoordinator_->createAndStartTask(taskId, capabilityId, taskType, task, error);
}

bool ProjectWorkspace::requestTaskCancellation(const TaskId& taskId, QString* error)
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    if (!taskCoordinator_) {
        if (error) *error = QStringLiteral(" 项目任务协调器不可用。");
        return false;
    }
    if (trainingAdapterHost_ && trainingAdapterHost_->managesTask(taskId)) {
        return trainingAdapterHost_->requestCancellation(taskId, error);
    }
    return taskCoordinator_->requestCancellation(taskId, error);
}

bool ProjectWorkspace::finalizeTask(const TaskId& taskId,
    TaskState terminalState,
    const Failure& failure,
    QString* error)
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    return taskCoordinator_ && taskCoordinator_->finalizeTask(taskId, terminalState, failure, error);
}

bool ProjectWorkspace::commitDatasetSnapshot(const TaskId& taskId,
    const DatasetSnapshotCommitRequest& request,
    DatasetSnapshotArtifactBundle* result,
    QString* error)
{
    if (!isOpen() || !taskId.isValid() || !result || request.datasetRoot.trimmed().isEmpty()
        || request.datasetFormat.trimmed().isEmpty() || request.driverId.trimmed().isEmpty()
        || request.driverVersion.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("提交数据集快照需要运行中任务、完整数据集驱动信息和输出对象。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("仅运行中的任务可以提交数据集快照。");
        return false;
    }
    ArtifactId artifactId;
    QString artifactStaging;
    if (!artifactStore_->begin(taskId, QStringLiteral("dataset_snapshot"), &artifactId, &artifactStaging, error)) {
        return false;
    }
    const auto abort = [&]() {
        QString ignored;
        artifactStore_->abort(artifactStaging, &ignored);
    };
    DatasetSnapshotResult snapshotResult;
    const QString stagingManifestPath = QDir(artifactStaging).filePath(QStringLiteral("dataset_snapshot.json"));
    const aitrain::CancellationCallback cancellation = request.options.isCancellationRequested;
    if (!copyDatasetTreeToEmptyStaging(request.datasetRoot, artifactStaging, cancellation, error)
        || !createDatasetSnapshot(artifactStaging, stagingManifestPath, request.datasetFormat.trimmed(),
            request.driverId.trimmed(), request.driverVersion.trimmed(), request.options, &snapshotResult, error)) {
        abort();
        return false;
    }
    QString artifactPath;
    if (!artifactStore_->commit(artifactId, taskId, QStringLiteral("dataset_snapshot"), artifactStaging,
            &storage_, &artifactPath, error, cancellation)) {
        if (QFileInfo::exists(artifactStaging)) abort();
        return false;
    }
    ArtifactSnapshot artifact;
    if (!storage_.artifact(artifactId, &artifact, error)) {
        QString ignored;
        artifactStore_->discardCommitted(artifactId, &storage_, &ignored);
        return false;
    }
    QString manifestSha256;
    for (const ArtifactFileSnapshot& file : artifact.files) {
        if (file.relativePath == QStringLiteral("dataset_snapshot.json")) {
            manifestSha256 = file.sha256;
            break;
        }
    }
    DatasetSnapshotRecord record;
    record.id = snapshotResult.snapshotId;
    record.datasetId = request.datasetId;
    record.taskId = taskId;
    record.artifactId = artifactId;
    record.rootPath = artifactPath;
    record.datasetFormat = request.datasetFormat.trimmed();
    record.driverId = request.driverId.trimmed();
    record.driverVersion = request.driverVersion.trimmed();
    record.rootHash = snapshotResult.rootHash;
    record.manifestSha256 = manifestSha256;
    record.fileCount = snapshotResult.fileCount;
    record.totalBytes = snapshotResult.totalBytes;
    if (manifestSha256.isEmpty() || !storage_.registerDatasetSnapshot(&record, error)) {
        QString ignored;
        artifactStore_->discardCommitted(artifactId, &storage_, &ignored);
        return false;
    }
    result->snapshot = record;
    result->artifactPath = artifactPath;
    result->manifestPath = QDir(artifactPath).filePath(QStringLiteral("dataset_snapshot.json"));
    result->manifest = snapshotResult.manifest;
    return true;
}

bool ProjectWorkspace::beginTrainingWorkflow(const TaskId& taskId,
    const TrainingWorkflowRequest& request,
    TrainingWorkflowDispatch* result,
    QString* error)
{
    if (!isOpen() || !taskId.isValid() || !result
        || request.templateId.trimmed().isEmpty() || request.trainingBackend.trimmed().isEmpty()
        || request.evaluationBackend.trimmed().isEmpty() || request.exportBackend.trimmed().isEmpty()
        || request.deploymentBackend.trimmed().isEmpty()) {
        if (error) *error = QStringLiteral("创建训练 Workflow 需要运行中任务、完整后端和输出对象。");
        return false;
    }
    TrainingWorkflowProfile profile;
    if (!resolveTrainingWorkflowProfile(request.trainingBackend, &profile, error)
        || request.templateId.trimmed() != profile.templateId
        || request.evaluationBackend.trimmed() != profile.evaluationBackend
        || request.exportBackend.trimmed() != profile.exportBackend
        || request.deploymentBackend.trimmed() != profile.deploymentBackend) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("训练 Workflow 的模板或步骤后端与已注册 Profile 不一致。");
        }
        return false;
    }
    if (!request.datasetId.isValid() || !request.datasetVersionId.isValid()
        || !request.snapshotId.isValid() || !request.snapshotArtifactId.isValid()) {
        if (error) *error = QStringLiteral("训练 Workflow 必须引用完整的 Dataset/Version/Snapshot/Artifact 身份。");
        return false;
    }
    TaskSnapshot task;
    DatasetSnapshotRecord snapshot;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 必须依附运行中任务。");
        return false;
    }
    if (task.taskType != profile.capabilityTaskType) {
        if (error) *error = QStringLiteral("训练根任务类型与已注册 Workflow Profile 不一致。");
        return false;
    }
    if (!storage_.datasetSnapshot(request.snapshotId, &snapshot, error)
        || snapshot.datasetId != request.datasetId
        || snapshot.datasetVersionId != request.datasetVersionId
        || snapshot.artifactId != request.snapshotArtifactId
        || snapshot.datasetFormat != profile.datasetFormat) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("训练 Workflow 引用的数据集身份、格式或 Snapshot Artifact 不一致。");
        }
        return false;
    }
    ArtifactSnapshot snapshotArtifact;
    if (!storage_.artifact(request.snapshotArtifactId, &snapshotArtifact, error)
        || snapshotArtifact.taskId != snapshot.taskId
        || snapshotArtifact.kind != QStringLiteral("dataset_snapshot")) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 必须引用已提交的 Dataset Snapshot Artifact。");
        return false;
    }
    const auto manifestIt = std::find_if(snapshotArtifact.files.cbegin(), snapshotArtifact.files.cend(),
        [](const ArtifactFileSnapshot& file) { return file.relativePath == QStringLiteral("dataset_snapshot.json"); });
    VerifiedTrainingWorkflowInput verifiedSnapshot;
    if (manifestIt == snapshotArtifact.files.cend()
        || manifestIt->sha256 != snapshot.manifestSha256
        || !resolveVerifiedArtifact(storage_, artifactStore_.get(), request.snapshotArtifactId,
            &verifiedSnapshot, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Snapshot Artifact 的 manifest 或文件完整性校验失败。");
        return false;
    }
    const VerifiedWorkflowArtifactFile* verifiedManifest = selectArtifactFile(
        verifiedSnapshot, {QStringLiteral("dataset_snapshot.json")});
    QFile manifestFile(verifiedManifest ? verifiedManifest->absolutePath : QString());
    QJsonParseError manifestParseError;
    const QJsonDocument manifestDocument = manifestFile.open(QIODevice::ReadOnly)
        ? QJsonDocument::fromJson(manifestFile.readAll(), &manifestParseError)
        : QJsonDocument{};
    const QJsonObject manifest = manifestDocument.object();
    if (!verifiedManifest || manifestParseError.error != QJsonParseError::NoError
        || manifest.value(QStringLiteral("schemaVersion")).toInt() != 2
        || !manifest.value(QStringLiteral("complete")).toBool()
        || manifest.value(QStringLiteral("snapshotId")).toString() != snapshot.id.toString()
        || manifest.value(QStringLiteral("datasetFormat")).toString() != snapshot.datasetFormat
        || manifest.value(QStringLiteral("rootHash")).toString() != snapshot.rootHash) {
        if (error) *error = QStringLiteral("训练 Snapshot manifest 与已登记身份不一致。");
        return false;
    }

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = profile.templateId;
    workflow.terminalPolicy = request.requireEvidenceBeforeTerminal
        ? WorkflowTerminalPolicy::EvidenceRequired
        : WorkflowTerminalPolicy::Immediate;
    QJsonObject parameterSummary = request.parameterSummary;
    parameterSummary.insert(QStringLiteral("datasetId"), request.datasetId.toString());
    parameterSummary.insert(QStringLiteral("datasetVersionId"), request.datasetVersionId.toString());
    parameterSummary.insert(QStringLiteral("datasetSnapshotId"), request.snapshotId.toString());
    parameterSummary.insert(QStringLiteral("datasetSnapshotArtifactId"), request.snapshotArtifactId.toString());
    parameterSummary.insert(QStringLiteral("datasetSnapshotSourceTaskId"), snapshot.taskId.toString());
    parameterSummary.insert(QStringLiteral("trainingBackend"), request.trainingBackend.trimmed());
    parameterSummary.insert(QStringLiteral("evaluationBackend"), request.evaluationBackend.trimmed());
    parameterSummary.insert(QStringLiteral("exportBackend"), request.exportBackend.trimmed());
    parameterSummary.insert(QStringLiteral("deploymentBackend"), request.deploymentBackend.trimmed());
    const QString validationRoot = QDir(runtimeStagingPath(taskId)).filePath(
        QStringLiteral("training-input-validation"));
    const QString validationPath = QDir(validationRoot).filePath(
        QStringLiteral("dataset_validation_report.json"));
    const QJsonObject validationReport{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("status"), QStringLiteral("validated")},
        {QStringLiteral("validator"), QStringLiteral("snapshot_integrity")},
        {QStringLiteral("datasetId"), snapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), snapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), snapshot.id.toString()},
        {QStringLiteral("snapshotArtifactId"), snapshot.artifactId.toString()},
        {QStringLiteral("snapshotSourceTaskId"), snapshot.taskId.toString()},
        {QStringLiteral("datasetFormat"), snapshot.datasetFormat},
        {QStringLiteral("manifestSha256"), snapshot.manifestSha256},
        {QStringLiteral("rootHash"), snapshot.rootHash}};
    RuntimeArtifactBundle validationBundle;
    if (!QDir().mkpath(validationRoot)
        || !writeJsonFile(validationPath, validationReport, error)
        || !commitRuntimeArtifacts(taskId, QStringLiteral("dataset_validation"),
            {{QStringLiteral("dataset_validation_report"), validationPath}},
            &validationBundle, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("无法提交训练 Snapshot 完整性校验 Artifact。");
        return false;
    }
    parameterSummary.insert(QStringLiteral("datasetValidationArtifactId"),
        validationBundle.artifactId.toString());
    const auto makeStep = [&workflow, &parameterSummary](int ordinal, const QString& kind, const QString& backend) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = kind;
        step.backend = backend;
        step.parameterSummary = parameterSummary;
        return step;
    };
    QVector<WorkflowStepSnapshot> steps;
    for (int ordinal = 0; ordinal < profile.steps.size(); ++ordinal) {
        const TrainingWorkflowStepProfile& stepProfile = profile.steps.at(ordinal);
        steps.append(makeStep(ordinal, stepProfile.kind, stepProfile.backend));
    }
    if (steps.size() != 8) {
        if (error) *error = QStringLiteral("训练 Workflow Profile 必须定义完整八步合同。");
        return false;
    }
    WorkflowInputBinding input;
    input.workflowRunId = workflow.id;
    input.role = QStringLiteral("dataset_snapshot");
    input.sourceArtifactId = snapshot.artifactId;
    input.sourceTaskId = snapshot.taskId;
    input.sourceArtifactKind = QStringLiteral("dataset_snapshot");
    input.datasetId = snapshot.datasetId;
    input.datasetSnapshotId = snapshot.id;
    input.datasetVersionId = snapshot.datasetVersionId;
    input.manifestSha256 = snapshot.manifestSha256;
    input.rootHash = snapshot.rootHash;
    input.boundAt = QDateTime::currentDateTimeUtc();
    if (!storage_.createWorkflowRunWithInput(workflow, steps, input, error)) {
        QString ignored;
        artifactStore_->discardCommitted(validationBundle.artifactId, &storage_, &ignored);
        return false;
    }
    WorkflowRunner runner(&storage_);
    TrainingWorkflowDispatch dispatch;
    dispatch.workflowRunId = workflow.id;
    if (!runner.beginNextStep(workflow.id, &dispatch.dispatch, error)
        || !dispatch.dispatch.hasStep || dispatch.dispatch.step.id != steps.at(0).id) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 未能派发 ValidateDataset 步骤。");
        return false;
    }
    WorkflowStepExecutionResult preflight;
    preflight.state = WorkflowStepState::Succeeded;
    preflight.outputArtifactId = validationBundle.artifactId;
    if (!runner.completeStep(workflow.id, steps.at(0).id, preflight, &dispatch.dispatch, error)
        || !dispatch.dispatch.hasStep || dispatch.dispatch.step.id != steps.at(1).id) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 未能收口 ValidateDataset 或派发 CreateSnapshot 步骤。");
        return false;
    }
    preflight.outputArtifactId = request.snapshotArtifactId;
    if (!runner.completeStep(workflow.id, steps.at(1).id, preflight, &dispatch.dispatch, error)
        || !dispatch.dispatch.hasStep || dispatch.dispatch.step.id != steps.at(2).id) {
        if (error) *error = QStringLiteral("训练 Workflow 未能派发 Train 步骤。");
        return false;
    }
    *result = dispatch;
    return true;
}

bool ProjectWorkspace::completeTrainingWorkflowStep(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    const WorkflowStepExecutionResult& execution,
    TrainingWorkflowDispatch* result,
    QString* error)
{
    if (!isOpen() || !workflowRunId.isValid() || !workflowStepId.isValid() || !result) {
        if (error) *error = QStringLiteral("收口训练 Workflow 需要已打开工作区、有效运行/步骤 ID 和输出对象。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    WorkflowRunner runner(&storage_);
    TrainingWorkflowDispatch dispatch;
    dispatch.workflowRunId = workflowRunId;
    if (!runner.completeStep(workflowRunId, workflowStepId, execution, &dispatch.dispatch, error)) return false;
    if (!dispatch.dispatch.hasStep && isTerminalWorkflowStepState(dispatch.dispatch.result.state)) {
        const TaskState taskState = taskStateForWorkflowResult(dispatch.dispatch.result);
        const Failure failure = completeWorkflowFailure(taskState, dispatch.dispatch.result.failure);
        if (workflow.terminalPolicy == WorkflowTerminalPolicy::EvidenceRequired) {
            if (!storage_.sealWorkflowTerminalization(workflowRunId, taskState, failure,
                    QDateTime::currentDateTimeUtc(), error)) return false;
        } else {
            TaskCoordinator coordinator(&storage_);
            if (!coordinator.finalizeTask(workflow.taskId, taskState, failure, error)) return false;
        }
    }
    *result = dispatch;
    return true;
}

bool ProjectWorkspace::startTrainingWorkflowAdapterStep(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    const PythonAdapterLaunch& launch,
    TrainingWorkflowDispatchHandler nextStepHandler,
    QString* error,
    TrainingWorkflowAdapterEventHandler eventHandler)
{
    if (!isOpen() || !taskCoordinator_ || !trainingAdapterHost_ || !workflowRunId.isValid()
        || !workflowStepId.isValid()) {
        if (error) *error = QStringLiteral("启动训练 Workflow Adapter 步骤需要已打开工作区、协调器和有效步骤。" );
        return false;
    }
    if (trainingAdapterHost_->isRunning()) {
        if (error) *error = QStringLiteral("已有训练 Workflow Adapter 正在运行。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto stepIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (stepIt == steps.cend() || stepIt->state != WorkflowStepState::Running) {
        if (error) *error = QStringLiteral("仅能托管已由 Workflow Runner 派发的 Running 步骤。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(workflow.taskId, &task, error) || task.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 的根任务未处于 Running 状态。");
        return false;
    }
    auto completedDispatch = std::make_shared<TrainingWorkflowDispatch>();
    auto completed = std::make_shared<bool>(false);
    return trainingAdapterHost_->startExistingTask(task, launch,
        [this, workflowRunId, workflowStepId, completedDispatch, completed](const ProtocolEnvelope& terminal,
            const ArtifactId& outputArtifactId, QString* terminalError) {
            const WorkflowStepExecutionResult execution = workflowExecutionForAdapterTerminal(terminal, outputArtifactId);
            if (!completeTrainingWorkflowStep(workflowRunId, workflowStepId, execution, completedDispatch.get(), terminalError)) {
                return false;
            }
            *completed = true;
            return true;
        }, error, [completedDispatch, completed, nextStepHandler] {
            if (*completed && nextStepHandler) {
                nextStepHandler(*completedDispatch);
            }
        }, std::move(eventHandler));
}

bool ProjectWorkspace::requestTrainingWorkflowAdapterCancellation(const TaskId& taskId, QString* error)
{
    if (!isOpen() || !trainingAdapterHost_ || !trainingAdapterHost_->managesTask(taskId)) {
        if (error) *error = QStringLiteral("没有运行中的训练 Workflow Adapter 管理指定任务。");
        return false;
    }
    return trainingAdapterHost_->requestCancellation(taskId, error);
}

bool ProjectWorkspace::isTrainingWorkflowAdapterRunning() const
{
    return trainingAdapterHost_ && trainingAdapterHost_->isRunning();
}

AdapterEventEndpoint ProjectWorkspace::trainingWorkflowAdapterEndpoint() const
{
    return trainingAdapterHost_ ? trainingAdapterHost_->adapterEndpoint() : AdapterEventEndpoint{};
}

bool ProjectWorkspace::resolveTrainingWorkflowStepInput(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    VerifiedTrainingWorkflowInput* result,
    QString* error) const
{
    if (!isOpen() || !workflowRunId.isValid() || !workflowStepId.isValid() || !result) {
        if (error) *error = QStringLiteral("解析训练 Workflow 输入需要已打开工作区、有效步骤和输出对象。");
        return false;
    }
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto stepIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (stepIt == steps.cend() || stepIt->state != WorkflowStepState::Running || !stepIt->inputArtifactId.isValid()) {
        if (error) *error = QStringLiteral("只能解析已派发 Running 训练步骤的有效输入 Artifact。");
        return false;
    }
    ArtifactSnapshot artifact;
    if (!storage_.artifact(stepIt->inputArtifactId, &artifact, error)) return false;
    const QString artifactPath = artifactStore_->artifactPath(artifact.id);
    if (!QFileInfo(artifactPath).isDir()) {
        if (error) *error = QStringLiteral("已提交训练输入 Artifact 目录不存在：%1").arg(artifact.id.toString());
        return false;
    }
    VerifiedTrainingWorkflowInput resolved;
    resolved.artifactId = artifact.id;
    resolved.artifactPath = artifactPath;
    for (const ArtifactFileSnapshot& file : artifact.files) {
        VerifiedWorkflowArtifactFile verified;
        if (!verifyArtifactFile(artifactPath, file, &verified, error)) return false;
        resolved.files.append(verified);
    }
    if (resolved.files.isEmpty()) {
        if (error) *error = QStringLiteral("已提交训练输入 Artifact 不包含可验证文件。");
        return false;
    }
    *result = resolved;
    return true;
}

bool ProjectWorkspace::prepareTrainingWorkflowAdapterLaunch(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    const TrainingWorkflowAdapterConfig& config,
    TrainingWorkflowAdapterLaunch* result,
    QString* error) const
{
    if (!isOpen() || !workflowRunId.isValid() || !workflowStepId.isValid() || !result
        || config.pythonProgram.trimmed().isEmpty() || config.trainersRoot.trimmed().isEmpty()
        || config.cancellationGraceMs < 1) {
        if (error) *error = QStringLiteral("构建训练 Workflow Adapter 请求需要有效工作区、步骤、Python 与训练器目录。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto stepIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (stepIt == steps.cend() || stepIt->state != WorkflowStepState::Running) {
        if (error) *error = QStringLiteral("只能为已派发 Running 训练步骤构建 Adapter 请求。");
        return false;
    }
    const QString trainingBackend = stepIt->parameterSummary.value(QStringLiteral("trainingBackend")).toString();
    TrainingWorkflowProfile profile;
    if (!resolveTrainingWorkflowProfile(trainingBackend, &profile, error)
        || workflow.templateId != profile.templateId) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练 Workflow 缺少匹配的已注册 Profile。");
        return false;
    }
    const TrainingWorkflowStepProfile* route = workflowStepProfile(profile, stepIt->kind);
    if (!route || route->backend != stepIt->backend || route->script.isEmpty()) {
        if (error) *error = QStringLiteral("训练步骤未在 Profile 中声明可执行 Adapter 路由：%1").arg(stepIt->kind);
        return false;
    }
    DatasetSnapshotRecord snapshot;
    if (!datasetSnapshotForWorkflow(storage_, steps, &snapshot, error)) return false;
    VerifiedTrainingWorkflowInput snapshotInput;
    if (!resolveVerifiedArtifact(storage_, artifactStore_.get(), snapshot.artifactId,
            &snapshotInput, error)) return false;
    ArtifactSnapshot snapshotArtifact;
    if (!storage_.artifact(snapshot.artifactId, &snapshotArtifact, error)) return false;
    const auto manifestIt = std::find_if(snapshotArtifact.files.cbegin(), snapshotArtifact.files.cend(),
        [](const ArtifactFileSnapshot& file) { return file.relativePath == QStringLiteral("dataset_snapshot.json"); });
    VerifiedWorkflowArtifactFile verifiedSnapshotManifest;
    const QString snapshotArtifactPath = artifactStore_->artifactPath(snapshot.artifactId);
    if (manifestIt == snapshotArtifact.files.cend() || manifestIt->sha256 != snapshot.manifestSha256
        || !verifyArtifactFile(snapshotArtifactPath, *manifestIt, &verifiedSnapshotManifest, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("登记的数据集快照 Artifact 缺少可信 manifest。" );
        return false;
    }
    VerifiedTrainingWorkflowInput input;
    if (!resolveTrainingWorkflowStepInput(workflowRunId, workflowStepId, &input, error)) return false;
    const VerifiedWorkflowArtifactFile* snapshotManifest = selectArtifactFile(input, {QStringLiteral("dataset_snapshot.json")});
    if (stepIt->kind == QStringLiteral("Train") && (!snapshotManifest || input.artifactId != snapshot.artifactId
        || snapshotManifest->absolutePath != verifiedSnapshotManifest.absolutePath)) {
        if (error) *error = QStringLiteral("Train 步骤必须直接消费与登记快照一致的 dataset_snapshot.json Artifact。");
        return false;
    }
    const QString outputRoot = QDir(runtimeStagingPath(workflow.taskId)).filePath(
        QStringLiteral("training-adapter/%1").arg(workflowStepId.toString()));
    if (QFileInfo::exists(outputRoot) || !QDir().mkpath(outputRoot)) {
        if (error) *error = QStringLiteral("训练 Adapter 输出暂存目录不可用：%1").arg(outputRoot);
        return false;
    }
    const QString snapshotStaging = QDir(outputRoot).filePath(QStringLiteral("snapshot-input"));
    QJsonObject request;
    request.insert(QStringLiteral("taskId"), workflow.taskId.toString());
    request.insert(QStringLiteral("datasetPath"), artifactStore_->artifactPath(snapshot.artifactId));
    request.insert(QStringLiteral("datasetSnapshotManifest"), verifiedSnapshotManifest.absolutePath);
    request.insert(QStringLiteral("datasetSnapshotStagingPath"), snapshotStaging);
    request.insert(QStringLiteral("outputPath"), outputRoot);
    request.insert(QStringLiteral("taskType"), profile.adapterTaskType);
    request.insert(QStringLiteral("backend"), stepIt->backend);
    request.insert(QStringLiteral("trainingBackend"), profile.trainingBackend);

    const QString scriptRelative = route->script;
    if (stepIt->kind == QStringLiteral("Train")) {
        request.insert(QStringLiteral("mode"), QStringLiteral("train"));
        QJsonObject parameters = stepIt->parameterSummary;
        parameters.insert(QStringLiteral("trainingBackend"), stepIt->backend);
        parameters.insert(QStringLiteral("exportOnnx"), profile.modelFamily == QStringLiteral("semantic_segmentation"));
        parameters.insert(QStringLiteral("datasetSnapshotManifest"), request.value(QStringLiteral("datasetSnapshotManifest")));
        parameters.insert(QStringLiteral("datasetSnapshotStagingPath"), snapshotStaging);
        request.insert(QStringLiteral("parameters"), parameters);
    } else if (stepIt->kind == QStringLiteral("Evaluate")) {
        request.insert(QStringLiteral("mode"), QStringLiteral("evaluate"));
        const VerifiedWorkflowArtifactFile* modelInput = selectArtifactFile(input, route->artifactCandidates);
        if (!modelInput) {
            if (error) *error = QStringLiteral("Evaluate 步骤输入缺少 Profile 声明的模型 Artifact。" );
            return false;
        }
        request.insert(QStringLiteral("modelPath"), modelInput->absolutePath);
        const VerifiedWorkflowArtifactFile* sidecar = selectArtifactFile(input,
            {QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json")});
        if (sidecar) request.insert(QStringLiteral("sidecarPath"), sidecar->absolutePath);
        const VerifiedWorkflowArtifactFile* checkpoint = selectArtifactFile(input,
            {QStringLiteral("checkpoint/best.pt"), QStringLiteral("checkpoint/last.pt"),
                QStringLiteral("checkpoint/model.ckpt")});
        if (checkpoint) request.insert(QStringLiteral("checkpointPath"), checkpoint->absolutePath);
        const VerifiedWorkflowArtifactFile* configFile = selectArtifactFile(input,
            {QStringLiteral("config/train.yml"), QStringLiteral("config/train.yaml")});
        if (configFile) request.insert(QStringLiteral("configPath"), configFile->absolutePath);
        const VerifiedWorkflowArtifactFile* dictionary = selectArtifactFile(input,
            {QStringLiteral("dictionary/dict.txt")});
        if (dictionary) request.insert(QStringLiteral("dictionaryPath"), dictionary->absolutePath);
        QJsonObject options = stepIt->parameterSummary;
        options.insert(QStringLiteral("datasetSnapshotId"), snapshot.id.toString());
        options.insert(QStringLiteral("datasetSnapshotManifest"), request.value(QStringLiteral("datasetSnapshotManifest")));
        options.insert(QStringLiteral("datasetSnapshotStagingPath"), snapshotStaging);
        request.insert(QStringLiteral("options"), options);
    } else if (stepIt->kind == QStringLiteral("Export")) {
        request.insert(QStringLiteral("mode"), QStringLiteral("export"));
        const VerifiedWorkflowArtifactFile* modelInput = selectArtifactFile(input, route->artifactCandidates);
        if (!modelInput) {
            if (error) *error = QStringLiteral("Export 步骤输入缺少 Profile 声明的模型 Artifact。" );
            return false;
        }
        request.insert(QStringLiteral("modelPath"), modelInput->absolutePath);
        const VerifiedWorkflowArtifactFile* sidecar = selectArtifactFile(input,
            {QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json")});
        if (sidecar) request.insert(QStringLiteral("sidecarPath"), sidecar->absolutePath);
        const VerifiedWorkflowArtifactFile* evaluationReport = selectArtifactFile(input,
            {QStringLiteral("evaluation_report/evaluation_report.json")});
        if (evaluationReport) {
            request.insert(QStringLiteral("evaluationReportPath"), evaluationReport->absolutePath);
        }
        request.insert(QStringLiteral("format"), stepIt->parameterSummary.value(QStringLiteral("exportFormat")).toString(QStringLiteral("onnx")));
        if (profile.artifactFormat == QStringLiteral("anomalib_bundle")) {
            const VerifiedWorkflowArtifactFile* anomalySidecar = selectArtifactFile(input,
                {QStringLiteral("anomaly_sidecar/anomaly_sidecar.json")});
            const VerifiedWorkflowArtifactFile* anomalyCheckpoint = selectArtifactFile(input,
                {QStringLiteral("checkpoint/model.ckpt")});
            if (!anomalySidecar || !anomalyCheckpoint) {
                if (error) *error = QStringLiteral("Anomalib Export 缺少 sidecar 或 checkpoint。");
                return false;
            }
            request.insert(QStringLiteral("sidecarPath"), anomalySidecar->absolutePath);
            request.insert(QStringLiteral("checkpointPath"), anomalyCheckpoint->absolutePath);
        } else if (profile.artifactFormat == QStringLiteral("onnx")) {
            request.insert(QStringLiteral("outputPath"), QDir(outputRoot).filePath(QStringLiteral("model.onnx")));
        }
        const VerifiedWorkflowArtifactFile* configFile = selectArtifactFile(input,
            {QStringLiteral("config/train.yml"), QStringLiteral("config/train.yaml")});
        if (configFile) request.insert(QStringLiteral("configPath"), configFile->absolutePath);
        const VerifiedWorkflowArtifactFile* dictionary = selectArtifactFile(input,
            {QStringLiteral("dictionary/dict.txt")});
        if (dictionary) request.insert(QStringLiteral("dictionaryPath"), dictionary->absolutePath);
        request.insert(QStringLiteral("parameters"), stepIt->parameterSummary);
    } else if (stepIt->kind == QStringLiteral("DeploymentValidate")) {
        const VerifiedWorkflowArtifactFile* sidecar = selectArtifactFile(input,
            {route->artifactCandidates.value(0)});
        const VerifiedWorkflowArtifactFile* bundle = selectArtifactFile(input,
            {route->artifactCandidates.value(1)});
        const VerifiedWorkflowArtifactFile* sample = selectDeploymentSample(
            snapshotInput, config.deploymentSampleRelativePath, error);
        if (route->artifactCandidates.size() < 2 || !sidecar || !bundle || !sample) {
            if (error) *error = QStringLiteral("外部 Python bundle 部署验证缺少可信 sidecar、模型包或样本图像。");
            return false;
        }
        request.insert(QStringLiteral("mode"), QStringLiteral("infer"));
        request.insert(QStringLiteral("modelPath"), sidecar->absolutePath);
        request.insert(QStringLiteral("sidecarPath"), sidecar->absolutePath);
        if (profile.artifactFormat == QStringLiteral("anomalib_bundle")) {
            request.insert(QStringLiteral("checkpointPath"), bundle->absolutePath);
        } else {
            request.insert(QStringLiteral("bundlePath"), bundle->absolutePath);
        }
        request.insert(QStringLiteral("imagePath"), sample->absolutePath);
        request.insert(QStringLiteral("options"), stepIt->parameterSummary);
    } else {
        if (error) *error = QStringLiteral("训练 Adapter 不支持当前 Workflow 步骤。" );
        return false;
    }

    const QString scriptPath = QDir(config.trainersRoot).filePath(scriptRelative);
    if (!QFileInfo(scriptPath).isFile()) {
        if (error) *error = QStringLiteral("训练 Adapter 脚本不存在：%1").arg(scriptPath);
        return false;
    }
    const QString requestPath = QDir(outputRoot).filePath(QStringLiteral("adapter_request.json"));
    if (!writeJsonFile(requestPath, request, error)) return false;
    result->requestPath = requestPath;
    result->request = request;
    result->launch.program = config.pythonProgram;
    result->launch.arguments = QStringList{scriptPath, QStringLiteral("--request"), requestPath};
    result->launch.workingDirectory = config.trainersRoot;
    result->launch.artifactCandidateRoots = QStringList{outputRoot, input.artifactPath, snapshotInput.artifactPath};
    result->launch.artifactCandidateRoots.removeDuplicates();
    result->launch.environment = config.environment;
    result->launch.cancellationGraceMs = config.cancellationGraceMs;
    return true;
}

bool ProjectWorkspace::registerTrainingWorkflowModel(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    TrainingModelRegistration* result,
    QString* error)
{
    if (!isOpen() || !artifactStore_ || !workflowRunId.isValid() || !workflowStepId.isValid() || !result) {
        if (error) *error = QStringLiteral("登记训练 Workflow 模型需要已打开工作区、运行中 RegisterModel 步骤和输出对象。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto registerIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (registerIt == steps.cend() || registerIt->state != WorkflowStepState::Running
        || registerIt->kind != QStringLiteral("RegisterModel")) {
        if (error) *error = QStringLiteral("仅能为已派发的 Running RegisterModel 步骤登记模型。");
        return false;
    }
    const QString trainingBackend = registerIt->parameterSummary.value(QStringLiteral("trainingBackend")).toString();
    TrainingWorkflowProfile profile;
    if (!resolveTrainingWorkflowProfile(trainingBackend, &profile, error)
        || workflow.templateId != profile.templateId) return false;
    const TrainingWorkflowStepProfile* registerRoute = workflowStepProfile(profile, QStringLiteral("RegisterModel"));
    if (!registerRoute || registerRoute->artifactCandidates.size() < 2) {
        if (error) *error = QStringLiteral("训练 Profile 缺少 RegisterModel 合同。");
        return false;
    }
    const auto exportIt = std::find_if(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
        return step.kind == QStringLiteral("Export") && step.state == WorkflowStepState::Succeeded && step.outputArtifactId.isValid();
    });
    if (exportIt == steps.cend()) {
        if (error) *error = QStringLiteral("RegisterModel 缺少已成功 Export 步骤的 Artifact。");
        return false;
    }
    DatasetSnapshotRecord snapshot;
    if (!datasetSnapshotForWorkflow(storage_, steps, &snapshot, error)) return false;
    const SnapshotId snapshotId = snapshot.id;
    VerifiedTrainingWorkflowInput exported;
    if (!resolveVerifiedArtifact(storage_, artifactStore_.get(), exportIt->outputArtifactId, &exported, error)) return false;
    const VerifiedWorkflowArtifactFile* entry = selectArtifactFile(exported,
        {registerRoute->artifactCandidates.value(0)});
    const VerifiedWorkflowArtifactFile* companion = selectArtifactFile(exported,
        {registerRoute->artifactCandidates.value(1)});
    if (!entry || !companion) {
        if (error) *error = QStringLiteral("RegisterModel 缺少 Profile 声明的同一 Export Artifact 模型入口或伴随文件。");
        return false;
    }
    const bool anomalibBundle = profile.artifactFormat == QStringLiteral("anomalib_bundle");
    const bool paddleOcrBundle = profile.artifactFormat == QStringLiteral("paddleocr_inference_bundle");
    const VerifiedWorkflowArtifactFile* sidecar = (anomalibBundle || paddleOcrBundle) ? entry : companion;
    QFile sidecarFile(sidecar->absolutePath);
    QJsonParseError parseError;
    if (!sidecarFile.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取已验证的 export sidecar：%1").arg(sidecarFile.errorString());
        return false;
    }
    const QJsonDocument sidecarDocument = QJsonDocument::fromJson(sidecarFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !sidecarDocument.isObject()) {
        if (error) *error = QStringLiteral("export sidecar 不是有效 JSON：%1").arg(parseError.errorString());
        return false;
    }
    ModelManifest manifest;
    bool manifestReady = false;
    if (anomalibBundle) {
        manifestReady = modelManifestFromAnomalibBundleSidecar(sidecarDocument.object(), profile,
            workflow.taskId, snapshotId, *sidecar, &manifest, error);
    } else if (paddleOcrBundle) {
        manifestReady = modelManifestFromPaddleOcrBundleSidecar(sidecarDocument.object(), profile,
            workflow.taskId, snapshotId, *sidecar, *companion, &manifest, error);
    } else {
        manifestReady = modelManifestFromTrainingExportSidecar(sidecarDocument.object(), profile,
            workflow.taskId, snapshotId, *entry, &manifest, error);
    }
    if (!manifestReady) return false;
    if (snapshot.datasetFormat != profile.datasetFormat
        || manifest.taskType != profile.adapterTaskType
        || manifest.modelFamily != profile.modelFamily
        || manifest.decoder != profile.decoder
        || manifest.runtimeRoutes != profile.runtimeRoutes) {
        if (error) {
            *error = QStringLiteral("export sidecar 与 Workflow Profile 的任务、数据格式或运行时合同不一致。");
        }
        return false;
    }

    ArtifactId registrationArtifactId;
    QString registrationStagingPath;
    if (!artifactStore_->begin(workflow.taskId, QStringLiteral("model_registration"), &registrationArtifactId, &registrationStagingPath, error)) return false;
    const QString manifestPath = QDir(registrationStagingPath).filePath(QStringLiteral("model_manifest.json"));
    const QJsonObject encoded = encodeModelManifest(manifest, error);
    if (encoded.isEmpty() || !writeArtifactFile(manifestPath, QJsonDocument(encoded).toJson(QJsonDocument::Indented), error)) {
        QString ignored;
        artifactStore_->abort(registrationStagingPath, &ignored);
        return false;
    }
    QString registrationArtifactPath;
    if (!artifactStore_->commit(registrationArtifactId, workflow.taskId, QStringLiteral("model_registration"), registrationStagingPath,
            &storage_, &registrationArtifactPath, error)) {
        QString ignored;
        artifactStore_->abort(registrationStagingPath, &ignored);
        return false;
    }
    ModelPackageSnapshot package{manifest, exportIt->outputArtifactId, QDateTime::currentDateTimeUtc()};
    if (!storage_.registerModelPackage(package, error)) {
        const QString registrationError = error ? *error : QStringLiteral("无法登记 Model Package。");
        QString cleanupError;
        if (!artifactStore_->discardCommitted(registrationArtifactId, &storage_, &cleanupError)) {
            if (error) *error = QStringLiteral("%1；同时无法清理模型登记 Artifact：%2").arg(registrationError, cleanupError);
        }
        return false;
    }
    result->modelPackage = package;
    result->registrationArtifact.artifactId = registrationArtifactId;
    result->registrationArtifact.artifactPath = registrationArtifactPath;
    result->registrationArtifact.pathsByKind.insert(QStringLiteral("model_manifest"),
        QDir(registrationArtifactPath).filePath(QStringLiteral("model_manifest.json")));
    return true;
}

bool ProjectWorkspace::prepareTrainingWorkflowDeploymentInvocation(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    const QString& deploymentSampleRelativePath,
    TrainingDeploymentInvocation* result,
    QString* error) const
{
    if (!isOpen() || !artifactStore_ || !workflowRunId.isValid() || !workflowStepId.isValid() || !result) {
        if (error) *error = QStringLiteral("准备训练部署验证需要已打开工作区、运行中步骤和输出对象。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto deploymentIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (deploymentIt == steps.cend() || deploymentIt->state != WorkflowStepState::Running
        || deploymentIt->kind != QStringLiteral("DeploymentValidate")
        || deploymentIt->backend != QStringLiteral("aitrain_onnxruntime")) {
        if (error) *error = QStringLiteral("仅支持由 aitrain_onnxruntime 执行的 Running DeploymentValidate 步骤。");
        return false;
    }
    const QString trainingBackend = deploymentIt->parameterSummary.value(QStringLiteral("trainingBackend")).toString();
    TrainingWorkflowProfile profile;
    if (!resolveTrainingWorkflowProfile(trainingBackend, &profile, error)
        || workflow.templateId != profile.templateId) return false;
    const TrainingWorkflowStepProfile* deploymentRoute = workflowStepProfile(profile, QStringLiteral("DeploymentValidate"));
    if (!deploymentRoute || deploymentRoute->artifactCandidates.size() < 2) {
        if (error) *error = QStringLiteral("训练 Profile 缺少 DeploymentValidate Artifact 合同。");
        return false;
    }
    const auto exportIt = std::find_if(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
        return step.kind == QStringLiteral("Export") && step.state == WorkflowStepState::Succeeded && step.outputArtifactId.isValid();
    });
    if (exportIt == steps.cend()) {
        if (error) *error = QStringLiteral("DeploymentValidate 缺少已成功 Export 步骤的 Artifact。");
        return false;
    }
    DatasetSnapshotRecord snapshot;
    if (!datasetSnapshotForWorkflow(storage_, steps, &snapshot, error)) return false;
    const SnapshotId snapshotId = snapshot.id;
    VerifiedTrainingWorkflowInput snapshotInput;
    if (!resolveVerifiedArtifact(storage_, artifactStore_.get(), snapshot.artifactId, &snapshotInput, error)) return false;
    const VerifiedWorkflowArtifactFile* sample = selectDeploymentSample(
        snapshotInput, deploymentSampleRelativePath, error);
    if (!sample) return false;
    VerifiedTrainingWorkflowInput exported;
    if (!resolveVerifiedArtifact(storage_, artifactStore_.get(), exportIt->outputArtifactId, &exported, error)) return false;
    const VerifiedWorkflowArtifactFile* model = selectArtifactFile(exported,
        {deploymentRoute->artifactCandidates.at(0)});
    const VerifiedWorkflowArtifactFile* sidecar = selectArtifactFile(exported,
        {deploymentRoute->artifactCandidates.at(1)});
    if (!model || !sidecar) {
        if (error) *error = QStringLiteral("DeploymentValidate 需要同一 Export Artifact 中的 model.onnx 与官方 export sidecar。");
        return false;
    }
    QFile sidecarFile(sidecar->absolutePath);
    QJsonParseError parseError;
    if (!sidecarFile.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取已验证的官方 export sidecar：%1").arg(sidecarFile.errorString());
        return false;
    }
    const QJsonDocument sidecarDocument = QJsonDocument::fromJson(sidecarFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !sidecarDocument.isObject()) {
        if (error) *error = QStringLiteral("官方 export sidecar 不是有效 JSON：%1").arg(parseError.errorString());
        return false;
    }
    ModelManifest manifest;
    if (!modelManifestFromTrainingExportSidecar(sidecarDocument.object(), profile,
            workflow.taskId, snapshotId, *model, &manifest, error)) return false;
    const QString outputPath = QDir(runtimeStagingPath(workflow.taskId)).filePath(
        QStringLiteral("deployment/%1").arg(workflowStepId.toString()));
    if (!QDir().mkpath(outputPath)) {
        if (error) *error = QStringLiteral("无法创建部署验证暂存目录：%1").arg(outputPath);
        return false;
    }
    RuntimeInvocation invocation;
    invocation.model = {manifest, exported.artifactPath};
    invocation.runtimeRoute = QStringLiteral("aitrain_onnxruntime");
    invocation.imagePath = sample->absolutePath;
    invocation.outputPath = outputPath;
    invocation.options = deploymentIt->parameterSummary.value(QStringLiteral("runtimeOptions")).toObject();
    const QJsonObject encoded = encodeRuntimeInvocation(invocation, error);
    if (encoded.isEmpty()) return false;
    result->invocation = encoded;
    result->sourceArtifactId = exportIt->outputArtifactId;
    return true;
}

bool ProjectWorkspace::renderTrainingWorkflowDeliveryReport(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    RuntimeArtifactBundle* result,
    QString* error)
{
    if (!isOpen() || !artifactStore_ || !workflowRunId.isValid() || !workflowStepId.isValid() || !result) {
        if (error) *error = QStringLiteral("渲染训练交付报告需要已打开工作区、运行中步骤和输出对象。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    const auto renderIt = std::find_if(steps.cbegin(), steps.cend(), [&workflowStepId](const WorkflowStepSnapshot& step) {
        return step.id == workflowStepId;
    });
    if (renderIt == steps.cend() || renderIt->state != WorkflowStepState::Running
        || renderIt->kind != QStringLiteral("RenderDeliveryReport")) {
        if (error) *error = QStringLiteral("仅能渲染已派发的 Running RenderDeliveryReport 步骤。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(workflow.taskId, &task, error) || task.state != TaskState::Running) return false;
    DatasetSnapshotRecord snapshot;
    WorkflowInputBinding externalInput;
    if (!datasetSnapshotForWorkflow(storage_, steps, &snapshot, error)
        || !storage_.workflowInput(workflow.id, QStringLiteral("dataset_snapshot"),
            &externalInput, error)
        || externalInput.datasetId != snapshot.datasetId
        || externalInput.datasetVersionId != snapshot.datasetVersionId
        || externalInput.datasetSnapshotId != snapshot.id
        || externalInput.sourceArtifactId != snapshot.artifactId
        || externalInput.sourceTaskId != snapshot.taskId) {
        if (error && error->isEmpty()) *error = QStringLiteral("训练交付报告缺少可信的外部 Snapshot lineage。");
        return false;
    }
    QJsonArray stepFacts;
    for (const WorkflowStepSnapshot& step : steps) {
        stepFacts.append(QJsonObject{{QStringLiteral("ordinal"), step.ordinal}, {QStringLiteral("kind"), step.kind},
            {QStringLiteral("state"), workflowStepStateToString(step.state)}, {QStringLiteral("backend"), step.backend},
            {QStringLiteral("inputArtifactId"), step.inputArtifactId.toString()}, {QStringLiteral("outputArtifactId"), step.outputArtifactId.toString()},
            {QStringLiteral("failureCode"), failureCodeToString(step.failure.code)}, {QStringLiteral("failureDetails"), step.failure.message}});
    }
    const QJsonObject externalInputFact{{QStringLiteral("role"), externalInput.role},
        {QStringLiteral("producerTaskId"), externalInput.sourceTaskId.toString()},
        {QStringLiteral("artifactId"), externalInput.sourceArtifactId.toString()},
        {QStringLiteral("datasetId"), externalInput.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), externalInput.datasetVersionId.toString()},
        {QStringLiteral("datasetSnapshotId"), externalInput.datasetSnapshotId.toString()},
        {QStringLiteral("manifestSha256"), externalInput.manifestSha256},
        {QStringLiteral("rootHash"), externalInput.rootHash}};
    const QJsonObject report{{QStringLiteral("schemaVersion"), 3}, {QStringLiteral("kind"), QStringLiteral("training_delivery_report")},
        {QStringLiteral("workflowRunId"), workflow.id.toString()}, {QStringLiteral("taskId"), workflow.taskId.toString()},
        {QStringLiteral("taskStateAtRender"), taskStateToString(task.state)},
        {QStringLiteral("externalInput"), externalInputFact}, {QStringLiteral("steps"), stepFacts},
        {QStringLiteral("limitations"), QStringLiteral("此报告是根任务结束前由 RenderDeliveryReport 步骤生成的工作流事实摘要；根任务终态后的完整 Evidence Bundle 会另行提交。")}};
    QStringList markdownLines;
    markdownLines << QStringLiteral("# AITrain  训练交付摘要")
        << QStringLiteral("") << QStringLiteral("- Workflow：%1").arg(workflow.id.toString())
        << QStringLiteral("- 根任务：%1").arg(workflow.taskId.toString())
        << QStringLiteral("- Snapshot 生产任务：%1").arg(externalInput.sourceTaskId.toString())
        << QStringLiteral("- Snapshot Artifact：%1").arg(externalInput.sourceArtifactId.toString())
        << QStringLiteral("- Dataset / Version / Snapshot：%1 / %2 / %3")
            .arg(externalInput.datasetId.toString(), externalInput.datasetVersionId.toString(),
                externalInput.datasetSnapshotId.toString())
        << QStringLiteral("- 渲染时任务状态：%1").arg(taskStateToString(task.state)) << QStringLiteral("")
        << QStringLiteral("| 顺序 | 步骤 | 状态 | 后端 | 输出 Artifact |")
        << QStringLiteral("| --- | --- | --- | --- | --- |");
    for (const WorkflowStepSnapshot& step : steps) {
        markdownLines.append(QStringLiteral("| %1 | %2 | %3 | %4 | %5 |")
            .arg(step.ordinal).arg(step.kind, workflowStepStateToString(step.state), step.backend, step.outputArtifactId.toString()));
    }
    ArtifactId artifactId;
    QString stagingPath;
    if (!artifactStore_->begin(workflow.taskId, QStringLiteral("training_delivery_report"), &artifactId, &stagingPath, error)) return false;
    const auto abort = [&]() { QString ignored; artifactStore_->abort(stagingPath, &ignored); };
    if (!writeArtifactFile(QDir(stagingPath).filePath(QStringLiteral("delivery_report.json")),
            QJsonDocument(report).toJson(QJsonDocument::Indented), error)
        || !writeArtifactFile(QDir(stagingPath).filePath(QStringLiteral("delivery_report.md")), markdownLines.join(QStringLiteral("\n")).toUtf8(), error)) {
        abort();
        return false;
    }
    QString artifactPath;
    if (!artifactStore_->commit(artifactId, workflow.taskId, QStringLiteral("training_delivery_report"), stagingPath,
            &storage_, &artifactPath, error)) {
        abort();
        return false;
    }
    result->artifactId = artifactId;
    result->artifactPath = artifactPath;
    result->pathsByKind = {{QStringLiteral("delivery_report_json"), QDir(artifactPath).filePath(QStringLiteral("delivery_report.json"))},
        {QStringLiteral("delivery_report_markdown"), QDir(artifactPath).filePath(QStringLiteral("delivery_report.md"))}};
    return true;
}

bool ProjectWorkspace::commitRuntimeArtifacts(const TaskId& taskId,
    const QString& bundleKind,
    const QVector<RuntimeArtifactCandidate>& candidates,
    RuntimeArtifactBundle* result,
    QString* error)
{
    if (!isOpen() || !taskId.isValid() || bundleKind.trimmed().isEmpty() || candidates.isEmpty() || !result) {
        if (error) *error = QStringLiteral("提交运行产物需要已打开工作区、有效任务和非空候选集合。");
        return false;
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error)) {
        return false;
    }
    if (task.state != TaskState::Running) {
        if (error) *error = QStringLiteral("仅运行中的任务可以提交运行产物。");
        return false;
    }
    const QString stagingRoot = runtimeStagingPath(taskId);
    if (stagingRoot.isEmpty() || !QDir(stagingRoot).exists()) {
        if (error) *error = QStringLiteral("运行产物暂存目录不存在：%1").arg(stagingRoot);
        return false;
    }

    ArtifactId artifactId;
    QString artifactStaging;
    if (!artifactStore_->begin(taskId, bundleKind, &artifactId, &artifactStaging, error)) {
        return false;
    }
    const auto abort = [&]() {
        QString abortError;
        artifactStore_->abort(artifactStaging, &abortError);
    };
    const QRegularExpression safeKind(QStringLiteral("^[A-Za-z0-9._-]+$"));
    QSet<QString> seenKinds;
    QHash<QString, QString> relativePaths;
    for (const RuntimeArtifactCandidate& candidate : candidates) {
        const QString kind = candidate.kind.trimmed();
        const QFileInfo source(candidate.sourcePath);
        if (!safeKind.match(kind).hasMatch() || seenKinds.contains(kind)
            || !source.exists() || !source.isFile() || source.isSymLink()
            || !isChildPath(stagingRoot, source.absoluteFilePath())) {
            if (error) *error = QStringLiteral("运行产物候选无效：%1").arg(candidate.sourcePath);
            abort();
            return false;
        }
        seenKinds.insert(kind);
        const QString fileName = source.fileName();
        const QString relativePath = QStringLiteral("%1/%2").arg(kind, fileName);
        const QString destination = QDir(artifactStaging).filePath(relativePath);
        if (!QDir().mkpath(QFileInfo(destination).absolutePath()) || !QFile::copy(source.absoluteFilePath(), destination)) {
            if (error) *error = QStringLiteral("无法暂存运行产物：%1").arg(source.absoluteFilePath());
            abort();
            return false;
        }
        relativePaths.insert(kind, relativePath);
    }
    QString artifactPath;
    if (!artifactStore_->commit(artifactId, taskId, bundleKind, artifactStaging, &storage_, &artifactPath, error)) {
        abort();
        return false;
    }
    result->artifactId = artifactId;
    result->artifactPath = artifactPath;
    result->pathsByKind.clear();
    for (auto it = relativePaths.cbegin(); it != relativePaths.cend(); ++it) {
        result->pathsByKind.insert(it.key(), QDir(artifactPath).filePath(it.value()));
    }
    return true;
}

bool ProjectWorkspace::buildWorkflowEvidenceBundle(const WorkflowRunId& workflowRunId,
    EvidenceBundle* result,
    QString* error) const
{
    if (!isOpen() || !workflowRunId.isValid() || !result) {
        if (error) *error = QStringLiteral("构建运行时 Evidence 需要已打开工作区、有效 Workflow ID 和输出对象。");
        return false;
    }
    WorkflowRunSnapshot workflow;
    TaskSnapshot task;
    if (!storage_.workflowRun(workflowRunId, &workflow, error)
        || !storage_.task(workflow.taskId, &task, error)) {
        return false;
    }
    const QVector<WorkflowStepSnapshot> steps = storage_.workflowSteps(workflowRunId, error);
    if (error && !error->isEmpty()) return false;
    if (!isTerminalTaskState(task.state)) {
        WorkflowTerminalizationSnapshot terminalization;
        if (workflow.terminalPolicy != WorkflowTerminalPolicy::EvidenceRequired
            || !storage_.workflowTerminalization(workflowRunId, &terminalization, error)
            || !projectedTerminalTaskForWorkflow(task, terminalization, &task, error)) {
            if (error && error->isEmpty()) *error = QStringLiteral("只有终态或 Evidence 门控中的已完成 Workflow 可以生成 Evidence。");
            return false;
        }
    }
    const QVector<MetricSnapshot> metrics = storage_.metricsForTask(task.id, error);
    if (error && !error->isEmpty()) return false;

    EvidenceBundle bundle;
    bundle.projectIdentity = stableProjectIdentity(workspacePath_);
    bundle.task = task;
    bundle.workflowRunId = workflow.id;
    bundle.createdAt = QDateTime::currentDateTimeUtc();
    const bool hasSucceededSnapshotStep = std::any_of(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
        return step.kind == QStringLiteral("CreateSnapshot") && step.state == WorkflowStepState::Succeeded;
    });
    if (hasSucceededSnapshotStep) {
        DatasetSnapshotRecord snapshot;
        if (!datasetSnapshotForWorkflow(storage_, steps, &snapshot, error)) return false;
        bundle.datasetSnapshotId = snapshot.id;
        WorkflowInputBinding binding;
        if (!storage_.workflowInput(workflow.id, QStringLiteral("dataset_snapshot"), &binding, error)
            || binding.datasetId != snapshot.datasetId
            || binding.datasetVersionId != snapshot.datasetVersionId
            || binding.datasetSnapshotId != snapshot.id
            || binding.sourceArtifactId != snapshot.artifactId
            || binding.sourceTaskId != snapshot.taskId
            || binding.manifestSha256 != snapshot.manifestSha256
            || binding.rootHash != snapshot.rootHash) {
            if (error && error->isEmpty()) {
                *error = QStringLiteral("Evidence 外部输入与训练 Workflow 的 Snapshot 绑定不一致。");
            }
            return false;
        }
        bundle.externalInputs.append(EvidenceExternalInput{
            binding.role, binding.sourceTaskId, binding.sourceArtifactId, binding.datasetId,
            binding.datasetVersionId, binding.datasetSnapshotId, binding.manifestSha256,
            binding.rootHash});
    } else {
        // 数据质量等工作流消费既有 Snapshot；Snapshot ID 必须来自已持久化步骤参数，
        // 并重新解析到 Storage 记录，不能由裸目录或 GUI 状态推断。
        for (const WorkflowStepSnapshot& step : steps) {
            const QString snapshotText = step.parameterSummary.value(QStringLiteral("datasetSnapshotId")).toString();
            if (snapshotText.isEmpty()) continue;
            SnapshotId snapshotId;
            DatasetSnapshotRecord snapshot;
            if (!SnapshotId::parse(snapshotText, &snapshotId, error)
                || !storage_.datasetSnapshot(snapshotId, &snapshot, error)) return false;
            bundle.datasetSnapshotId = snapshot.id;
            WorkflowInputBinding binding;
            QString bindingError;
            if (storage_.workflowInput(workflow.id, QStringLiteral("dataset_snapshot"),
                    &binding, &bindingError)) {
                if (binding.workflowRunId != workflow.id
                    || binding.role != QStringLiteral("dataset_snapshot")
                    || binding.sourceArtifactKind != QStringLiteral("dataset_snapshot")
                    || binding.sourceArtifactId != snapshot.artifactId
                    || binding.sourceTaskId != snapshot.taskId
                    || binding.datasetId != snapshot.datasetId
                    || binding.datasetVersionId != snapshot.datasetVersionId
                    || binding.datasetSnapshotId != snapshot.id
                    || binding.manifestSha256 != snapshot.manifestSha256
                    || binding.rootHash != snapshot.rootHash) {
                    if (error) *error = QStringLiteral("Evidence 外部输入与消费 Workflow 的 Snapshot 绑定不一致。");
                    return false;
                }
                bundle.externalInputs.append(EvidenceExternalInput{
                    binding.role, binding.sourceTaskId, binding.sourceArtifactId, binding.datasetId,
                    binding.datasetVersionId, binding.datasetSnapshotId, binding.manifestSha256,
                    binding.rootHash});
            } else if (workflow.templateId == QStringLiteral("dataset_quality")) {
                if (error) {
                    *error = bindingError.isEmpty()
                        ? QStringLiteral("Data Quality Evidence 缺少 dataset_snapshot 外部输入绑定。")
                        : bindingError;
                }
                return false;
            }
            break;
        }
    }

    QSet<QString> backends;
    QJsonArray backendList;
    QJsonArray parameterSteps;
    QJsonArray statusSteps;
    QSet<QString> artifactIds;
    QHash<QString, int> artifactIndexes;
    const auto appendArtifact = [&](const ArtifactId& artifactId, const QString& role,
                                    const WorkflowStepSnapshot* step) -> bool {
        if (!artifactId.isValid()) return true;
        const QString id = artifactId.toString();
        int index = artifactIndexes.value(id, -1);
        if (index < 0) {
            ArtifactSnapshot artifact;
            if (!storage_.artifact(artifactId, &artifact, error)) return false;
            EvidenceArtifact evidenceArtifact;
            evidenceArtifact.artifactId = artifact.id;
            evidenceArtifact.kind = artifact.kind;
            evidenceArtifact.facts = artifactFacts(artifact);
            bundle.artifacts.append(evidenceArtifact);
            index = bundle.artifacts.size() - 1;
            artifactIndexes.insert(id, index);
            artifactIds.insert(id);
        }
        if (step) {
            QJsonArray references = bundle.artifacts[index].facts.value(QStringLiteral("workflowReferences")).toArray();
            references.append(QJsonObject{{QStringLiteral("stepId"), step->id.toString()},
                {QStringLiteral("ordinal"), step->ordinal},
                {QStringLiteral("role"), role}});
            bundle.artifacts[index].facts.insert(QStringLiteral("workflowReferences"), references);
        }
        return true;
    };

    const QVector<ArtifactSnapshot> taskArtifacts = storage_.artifactsForTask(task.id, error);
    if (error && !error->isEmpty()) return false;
    for (const ArtifactSnapshot& artifact : taskArtifacts) {
        if (!appendArtifact(artifact.id, QStringLiteral("task_artifact"), nullptr)) return false;
    }
    for (const WorkflowStepSnapshot& step : steps) {
        if (!backends.contains(step.backend)) {
            backends.insert(step.backend);
            backendList.append(step.backend);
        }
        parameterSteps.append(QJsonObject{{QStringLiteral("stepId"), step.id.toString()},
            {QStringLiteral("ordinal"), step.ordinal}, {QStringLiteral("kind"), step.kind},
            {QStringLiteral("backend"), step.backend}, {QStringLiteral("parameters"), step.parameterSummary},
            {QStringLiteral("retryCount"), step.retryCount}});
        statusSteps.append(QJsonObject{{QStringLiteral("stepId"), step.id.toString()},
            {QStringLiteral("ordinal"), step.ordinal}, {QStringLiteral("kind"), step.kind},
            {QStringLiteral("state"), workflowStepStateToString(step.state)},
            {QStringLiteral("startedAt"), step.startedAt.isValid() ? step.startedAt.toUTC().toString(Qt::ISODateWithMs) : QString()},
            {QStringLiteral("finishedAt"), step.finishedAt.isValid() ? step.finishedAt.toUTC().toString(Qt::ISODateWithMs) : QString()},
            {QStringLiteral("inputArtifactId"), step.inputArtifactId.toString()},
            {QStringLiteral("outputArtifactId"), step.outputArtifactId.toString()},
            {QStringLiteral("failureCode"), failureCodeToString(step.failure.code)},
            {QStringLiteral("failureMessage"), step.failure.message}});
        if (!appendArtifact(step.inputArtifactId, QStringLiteral("input"), &step)
            || !appendArtifact(step.outputArtifactId, QStringLiteral("output"), &step)) return false;
    }
    QJsonArray metricValues;
    for (const MetricSnapshot& metric : metrics) {
        metricValues.append(QJsonObject{{QStringLiteral("name"), metric.name}, {QStringLiteral("value"), metric.value},
            {QStringLiteral("occurredAt"), metric.occurredAt.toUTC().toString(Qt::ISODateWithMs)}});
    }
    bundle.backendEnvironment = {{QStringLiteral("backends"), backendList},
        {QStringLiteral("evidenceSource"), QStringLiteral("aitrain_persisted_storage")}};
    bundle.parameters = {{QStringLiteral("workflowTemplateId"), workflow.templateId},
        {QStringLiteral("steps"), parameterSteps}};
    bundle.metrics = {{QStringLiteral("observations"), metricValues}};
    bundle.runtimeStatus = {{QStringLiteral("workflowRunId"), workflow.id.toString()},
        {QStringLiteral("workflowTemplateId"), workflow.templateId},
        {QStringLiteral("steps"), statusSteps}};
    bundle.limitations.clear();
    bundle.limitations.append(QStringLiteral("仅陈述已持久化的任务、Workflow、指标和 Artifact 事实；不代表客户域精度或外部部署验收。"));
    const auto trainStep = std::find_if(steps.cbegin(), steps.cend(), [](const WorkflowStepSnapshot& step) {
        return step.kind == QStringLiteral("Train");
    });
    TrainingWorkflowProfile profile;
    if (trainStep != steps.cend() && resolveTrainingWorkflowProfile(trainStep->backend, &profile, nullptr)) {
        for (const QString& limitation : profile.limitations) {
            if (!bundle.limitations.contains(limitation)) bundle.limitations.append(limitation);
        }
    }
    if (!validateEvidenceBundle(bundle, error)) return false;
    *result = bundle;
    return true;
}

bool ProjectWorkspace::commitEvidenceBundle(const EvidenceBundle& bundle,
    EvidenceArtifactBundle* result,
    QString* error)
{
    if (!isOpen() || !result || !validateEvidenceBundle(bundle, error)) {
        if (error && error->isEmpty()) *error = QStringLiteral("提交 Evidence Bundle 需要有效输出对象。");
        return false;
    }
    TaskSnapshot persisted;
    WorkflowRunSnapshot workflow;
    if (!storage_.task(bundle.task.id, &persisted, error)
        || !storage_.workflowRun(bundle.workflowRunId, &workflow, error)
        || workflow.taskId != persisted.id) {
        if (error && error->isEmpty()) *error = QStringLiteral("Evidence Bundle 的 Workflow 与根任务不一致。");
        return false;
    }
    bool taskFactsMatch = isTerminalTaskState(persisted.state)
        && persisted.requestId == bundle.task.requestId
        && persisted.state == bundle.task.state
        && persisted.failure.code == bundle.task.failure.code
        && persisted.failure.message == bundle.task.failure.message
        && persisted.failure.suggestedAction == bundle.task.failure.suggestedAction
        && persisted.failure.occurredAt.toUTC() == bundle.task.failure.occurredAt.toUTC();
    if (!taskFactsMatch && (persisted.state == TaskState::Running || persisted.state == TaskState::CancelRequested)
        && workflow.terminalPolicy == WorkflowTerminalPolicy::EvidenceRequired) {
        WorkflowTerminalizationSnapshot terminalization;
        TaskSnapshot projected;
        taskFactsMatch = storage_.workflowTerminalization(bundle.workflowRunId, &terminalization, error)
            && projectedTerminalTaskForWorkflow(persisted, terminalization, &projected, error)
            && projected.requestId == bundle.task.requestId
            && projected.state == bundle.task.state
            && projected.failure.code == bundle.task.failure.code
            && projected.failure.message == bundle.task.failure.message
            && projected.failure.suggestedAction == bundle.task.failure.suggestedAction
            && projected.failure.occurredAt.toUTC() == bundle.task.failure.occurredAt.toUTC();
    }
    if (!taskFactsMatch) {
        if (error) *error = QStringLiteral("Evidence Bundle 的终态任务事实与已持久化记录不一致。");
        return false;
    }
    const QByteArray json = EvidenceRenderer::renderJson(bundle, error);
    if (json.isEmpty()) return false;
    const QString markdown = EvidenceRenderer::renderMarkdown(bundle, error);
    if (markdown.isEmpty()) return false;
    const QString html = EvidenceRenderer::renderHtml(bundle, error);
    if (html.isEmpty()) return false;
    const QByteArray modelCard = EvidenceRenderer::renderModelCard(bundle, error);
    if (modelCard.isEmpty()) return false;

    ArtifactId artifactId;
    QString stagingPath;
    if (!artifactStore_->begin(bundle.task.id, QStringLiteral("evidence_bundle"), &artifactId, &stagingPath, error)) return false;
    const auto abort = [&]() {
        QString ignored;
        artifactStore_->abort(stagingPath, &ignored);
    };
    const QVector<QPair<QString, QByteArray>> files = {
        {QStringLiteral("evidence.json"), json},
        {QStringLiteral("evidence.md"), markdown.toUtf8()},
        {QStringLiteral("evidence.html"), html.toUtf8()},
        {QStringLiteral("model_card.json"), modelCard}};
    for (const auto& file : files) {
        if (!writeArtifactFile(QDir(stagingPath).filePath(file.first), file.second, error)) {
            abort();
            return false;
        }
    }
    QString artifactPath;
    const WorkflowRunId terminalizationWorkflow = workflow.terminalPolicy == WorkflowTerminalPolicy::EvidenceRequired
        ? bundle.workflowRunId : WorkflowRunId{};
    if (!artifactStore_->commit(artifactId, bundle.task.id, QStringLiteral("evidence_bundle"),
            stagingPath, &storage_, &artifactPath, error, {}, nullptr, terminalizationWorkflow)) {
        abort();
        return false;
    }
    result->artifactId = artifactId;
    result->artifactPath = artifactPath;
    result->pathsByKind.clear();
    for (const auto& file : files) {
        result->pathsByKind.insert(file.first, QDir(artifactPath).filePath(file.first));
    }
    return true;
}

bool ProjectWorkspace::recordWorkflowEvidenceFailure(const WorkflowRunId& workflowRunId,
    const Failure& failure,
    QString* error)
{
    if (!isOpen() || !workflowRunId.isValid()) {
        if (error) *error = QStringLiteral("记录 Workflow Evidence 失败需要已打开工作区和有效 Workflow ID。");
        return false;
    }
    return storage_.recordWorkflowTerminalizationEvidenceFailure(workflowRunId, failure, error);
}

bool ProjectWorkspace::closeWorkflowTerminalization(const WorkflowRunId& workflowRunId,
    QString* error)
{
    if (!isOpen() || !workflowRunId.isValid()) {
        if (error) *error = QStringLiteral("关闭 Workflow 终态需要已打开工作区和有效 Workflow ID。");
        return false;
    }
    WorkflowTerminalizationSnapshot terminalization;
    TaskSnapshot task;
    if (!storage_.workflowTerminalization(workflowRunId, &terminalization, error)
        || !storage_.task(terminalization.taskId, &task, error)) return false;
    return storage_.closeWorkflowTerminalization(workflowRunId, task.state, error);
}

QString ProjectWorkspace::runtimeStagingPath(const TaskId& taskId) const
{
    if (workspacePath_.isEmpty() || !taskId.isValid()) {
        return {};
    }
    return QDir(workspacePath_).filePath(
        QStringLiteral(".runtime-staging/%1").arg(taskId.toString()));
}

bool ProjectWorkspace::cleanupRuntimeStaging(const TaskId& taskId, QString* error)
{
    const QString path = runtimeStagingPath(taskId);
    if (path.isEmpty()) {
        if (error) *error = QStringLiteral("清理运行暂存目录需要有效任务 ID。");
        return false;
    }
    if (!QFileInfo::exists(path)) {
        return true;
    }
    if (!isChildPath(QDir(workspacePath_).filePath(QStringLiteral(".runtime-staging")), path)
        || !QDir(path).removeRecursively()) {
        if (error) *error = QStringLiteral("无法清理运行产物暂存目录：%1").arg(path);
        return false;
    }
    return true;
}

QVector<TaskSnapshot> ProjectWorkspace::tasks(int limit, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.tasks(limit, error);
}

QVector<DatasetCatalogItem> ProjectWorkspace::datasets(int limit, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.datasets(limit, error);
}

bool ProjectWorkspace::task(const TaskId& taskId, TaskSnapshot* result, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    return storage_.task(taskId, result, error);
}

bool ProjectWorkspace::artifact(const ArtifactId& artifactId, ArtifactSnapshot* result, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。已提交 Artifact 查询被拒绝。");
        return false;
    }
    return storage_.artifact(artifactId, result, error);
}

QVector<ArtifactSnapshot> ProjectWorkspace::artifactsForTask(const TaskId& taskId, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.artifactsForTask(taskId, error);
}

QVector<DeliveryEvidenceCandidate> ProjectWorkspace::deliveryEvidenceCandidates(
    int limit, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.deliveryEvidenceCandidates(limit, error);
}

bool ProjectWorkspace::readCommittedArtifactFile(const ArtifactId& artifactId,
    const QString& relativePath,
    ArtifactFilePreview* result,
    qint64 maxBytes,
    QString* error) const
{
    if (error) error->clear();
    if (!isOpen() || !artifactId.isValid() || !result || maxBytes <= 0 || maxBytes > 4 * 1024 * 1024) {
        if (error) *error = QStringLiteral("读取 Artifact 预览需要有效工作区、Artifact ID、输出对象和合法大小限制。");
        return false;
    }
    ArtifactSnapshot snapshot;
    if (!storage_.artifact(artifactId, &snapshot, error)) return false;
    return readCommittedArtifactFile(snapshot, relativePath, result, maxBytes, error);
}

bool ProjectWorkspace::readCommittedArtifactFile(const ArtifactSnapshot& snapshot,
    const QString& relativePath,
    ArtifactFilePreview* result,
    qint64 maxBytes,
    QString* error) const
{
    if (error) error->clear();
    if (!isOpen() || !snapshot.id.isValid() || !result || maxBytes <= 0 || maxBytes > 4 * 1024 * 1024) {
        if (error) *error = QStringLiteral("读取 Artifact 预览需要有效工作区、Artifact ID、输出对象和合法大小限制。");
        return false;
    }

    const QString normalizedPath = QDir::cleanPath(QDir::fromNativeSeparators(relativePath.trimmed()));
    if (normalizedPath.isEmpty() || normalizedPath == QStringLiteral(".")
        || QDir::isAbsolutePath(normalizedPath) || normalizedPath == QStringLiteral("..")
        || normalizedPath.startsWith(QStringLiteral("../"))) {
        if (error) *error = QStringLiteral("Artifact 预览相对路径无效。");
        return false;
    }

    const auto fileIt = std::find_if(snapshot.files.cbegin(), snapshot.files.cend(),
        [&normalizedPath](const ArtifactFileSnapshot& file) {
            return QDir::cleanPath(QDir::fromNativeSeparators(file.relativePath)) == normalizedPath;
        });
    if (fileIt == snapshot.files.cend()) {
        if (error) *error = QStringLiteral("Artifact 清单中不存在请求的文件。");
        return false;
    }

    VerifiedWorkflowArtifactFile verified;
    if (!verifyArtifactFile(artifactStore_->artifactPath(snapshot.id), *fileIt, &verified, error)) {
        return false;
    }
    QFile file(verified.absolutePath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取已提交 Artifact 文件。");
        return false;
    }
    const QByteArray content = file.read(maxBytes + 1);
    if (file.error() != QFileDevice::NoError) {
        if (error) *error = QStringLiteral("读取已提交 Artifact 文件失败。");
        return false;
    }
    *result = {verified.relativePath, verified.sha256, verified.byteCount,
        content.left(maxBytes), content.size() > maxBytes};
    return true;
}

QVector<MetricSnapshot> ProjectWorkspace::metricsForTask(const TaskId& taskId, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.metricsForTask(taskId, error);
}

QVector<WorkflowRunSnapshot> ProjectWorkspace::workflowRunsForTask(const TaskId& taskId, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.workflowRunsForTask(taskId, error);
}

QVector<WorkflowStepSnapshot> ProjectWorkspace::workflowSteps(const WorkflowRunId& workflowRunId, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.workflowSteps(workflowRunId, error);
}

QVector<ModelPackageSnapshot> ProjectWorkspace::modelPackages(int limit, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return {};
    }
    return storage_.modelPackages(limit, error);
}

bool ProjectWorkspace::projectSummary(ProjectSummarySnapshot* result, QString* error) const
{
    if (!isOpen()) {
        if (error) *error = QStringLiteral(" 项目工作区未打开。");
        return false;
    }
    return storage_.projectSummary(result, error);
}

QString ProjectWorkspace::workspacePath() const
{
    return workspacePath_;
}

} // namespace aitrain
