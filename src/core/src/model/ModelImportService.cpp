#include "aitrain/model/ModelImportService.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace aitrain {
namespace {

bool copyAndHash(const QString& sourcePath,
    const QString& destinationPath,
    QString* sha256,
    QString* error,
    const aitrain::CancellationCallback& cancellation,
    bool* canceled)
{
    if (canceled) {
        *canceled = false;
    }
    QFile source(sourcePath);
    if (!source.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取导入模型：%1").arg(sourcePath);
        return false;
    }
    if (!QDir().mkpath(QFileInfo(destinationPath).absolutePath())) {
        if (error) *error = QStringLiteral("无法创建导入 Artifact 目录：%1").arg(destinationPath);
        return false;
    }
    QFile destination(destinationPath);
    if (!destination.open(QIODevice::WriteOnly)) {
        if (error) *error = QStringLiteral("无法写入导入 Artifact：%1").arg(destinationPath);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!source.atEnd()) {
        if (aitrain::isCancellationRequested(cancellation)) {
            if (canceled) {
                *canceled = true;
            }
            if (error) *error = QStringLiteral("模型导入已取消。");
            return false;
        }
        const QByteArray block = source.read(1024 * 1024);
        if (block.isEmpty() && source.error() != QFile::NoError) {
            if (error) *error = QStringLiteral("读取导入模型时出错。 ");
            return false;
        }
        if (destination.write(block) != block.size()) {
            if (error) *error = QStringLiteral("写入导入模型时出错。 ");
            return false;
        }
        hash.addData(block);
    }
    if (sha256) *sha256 = QString::fromLatin1(hash.result().toHex());
    return true;
}

} // namespace

ModelImportService::ModelImportService(TaskCoordinator* coordinator, ArtifactStore* artifactStore)
    : coordinator_(coordinator)
    , artifactStore_(artifactStore)
{
}

bool ModelImportService::importModel(const ModelImportRequest& request,
    ModelImportResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!coordinator_ || !coordinator_->storage() || !artifactStore_ || !result) {
        if (error) *error = QStringLiteral("模型导入服务未完成初始化。" );
        return false;
    }
    const QFileInfo source(request.sourceFilePath);
    if (!source.exists() || !source.isFile() || source.isSymLink()) {
        if (error) *error = QStringLiteral("只允许导入存在的常规模型文件，且不允许符号链接。" );
        return false;
    }
    ModelManifest manifest = request.manifest;
    if (!manifest.modelPackageId.isValid() || !manifest.sourceSnapshotId.isValid() || manifest.artifactEntryPath.isEmpty()
        || QDir::isAbsolutePath(manifest.artifactEntryPath)
        || QDir::cleanPath(manifest.artifactEntryPath).startsWith(QStringLiteral(".."))) {
        if (error) *error = QStringLiteral("导入模型需要用户确认的模型包 ID、来源 Snapshot 和产物入口路径。" );
        return false;
    }
    *result = {};
    TaskSnapshot task;
    const bool taskCreated = request.taskId.isValid()
        ? coordinator_->createAndStartTask(request.taskId, QStringLiteral("model.import"), QStringLiteral("model_import"), &task, error)
        : coordinator_->createAndStartTask(QStringLiteral("model.import"), QStringLiteral("model_import"), &task, error);
    if (!taskCreated) return false;
    result->task = task;
    ArtifactId artifactId;
    QString stagingPath;
    QString artifactPath;
    const auto finalizeFailed = [&](const QString& message) {
        QString finalizationError;
        if (!coordinator_->finalizeTask(task.id, TaskState::Failed,
                {FailureCode::ArtifactIncomplete, message, {}, QDateTime::currentDateTimeUtc()}, &finalizationError)) {
            if (error) *error = finalizationError;
            return false;
        }
        result->task.state = TaskState::Failed;
        result->task.failure = {FailureCode::ArtifactIncomplete, message, {}, QDateTime::currentDateTimeUtc()};
        if (error) *error = message;
        return false;
    };
    const auto fail = [&](QString message) {
        if (!stagingPath.isEmpty()) {
            QString cleanupError;
            if (!artifactStore_->abort(stagingPath, &cleanupError)) {
                message += QStringLiteral("；并且无法清理 Artifact staging：%1").arg(cleanupError);
            } else {
                stagingPath.clear();
            }
        }
        return finalizeFailed(message);
    };
    const auto cancel = [&](const QString& message) {
        if (!stagingPath.isEmpty()) {
            QString cleanupError;
            if (!artifactStore_->abort(stagingPath, &cleanupError)) {
                return finalizeFailed(QStringLiteral("模型导入取消后无法清理 Artifact staging：%1").arg(cleanupError));
            }
            stagingPath.clear();
        }
        QString finalizationError;
        if (!coordinator_->requestCancellation(task.id, &finalizationError)) {
            if (error) *error = finalizationError;
            return false;
        }
        if (!coordinator_->finalizeTask(task.id, TaskState::Canceled,
                {FailureCode::Canceled, message, {}, QDateTime::currentDateTimeUtc()}, &finalizationError)) {
            if (error) *error = finalizationError;
            return false;
        }
        result->task.state = TaskState::Canceled;
        result->task.failure = {FailureCode::Canceled, message, {}, QDateTime::currentDateTimeUtc()};
        if (error) *error = message;
        return false;
    };
    if (aitrain::isCancellationRequested(cancellation)) {
        return cancel(QStringLiteral("模型导入在开始前已取消。"));
    }
    manifest.sourceTaskId = task.id;
    if (!artifactStore_->begin(task.id, QStringLiteral("model_import_bundle"), &artifactId, &stagingPath, error)) return fail(error ? *error : QStringLiteral("无法创建导入 Artifact staging。"));
    QString hash;
    const QString destination = QDir(stagingPath).filePath(manifest.artifactEntryPath);
    bool canceled = false;
    if (!copyAndHash(source.absoluteFilePath(), destination, &hash, error, cancellation, &canceled)) {
        return canceled
            ? cancel(error ? *error : QStringLiteral("模型导入已取消。"))
            : fail(error ? *error : QStringLiteral("无法复制导入模型。"));
    }
    if (aitrain::isCancellationRequested(cancellation)) {
        return cancel(QStringLiteral("模型导入在提交前已取消。"));
    }
    manifest.sourceArtifactSha256 = hash;
    if (!validateModelManifest(manifest, error)) return fail(error ? *error : QStringLiteral("导入 Manifest 无效。"));
    bool commitCanceled = false;
    if (!artifactStore_->commit(artifactId, task.id, QStringLiteral("model_import_bundle"), stagingPath,
            coordinator_->storage(), &artifactPath, error, cancellation, &commitCanceled)) {
        return commitCanceled
            ? cancel(error ? *error : QStringLiteral("模型导入在 Artifact 提交时已取消。"))
            : fail(error ? *error : QStringLiteral("无法提交导入 Artifact。"));
    }
    stagingPath.clear();
    if (aitrain::isCancellationRequested(cancellation)) {
        QString cleanupError;
        if (!artifactStore_->discardCommitted(artifactId, coordinator_->storage(), &cleanupError)) {
            return finalizeFailed(QStringLiteral("模型导入取消后无法清理已提交 Artifact：%1").arg(cleanupError));
        }
        return cancel(QStringLiteral("模型导入在登记前已取消。"));
    }
    ModelPackageSnapshot modelPackage{manifest, artifactId, QDateTime::currentDateTimeUtc()};
    if (!coordinator_->storage()->registerModelPackage(modelPackage, error)) {
        const QString registrationError = error ? *error : QStringLiteral("无法登记导入模型包。" );
        QString cleanupError;
        if (!artifactStore_->discardCommitted(artifactId, coordinator_->storage(), &cleanupError)) {
            return finalizeFailed(QStringLiteral("模型包登记失败：%1；并且无法清理已提交 Artifact：%2")
                    .arg(registrationError, cleanupError));
        }
        return fail(registrationError);
    }
    if (!coordinator_->finalizeTask(task.id, TaskState::Succeeded, {}, error)) return false;
    result->task = task;
    result->task.state = TaskState::Succeeded;
    result->modelPackage = modelPackage;
    result->artifactPath = artifactPath;
    return true;
}

} // namespace aitrain
