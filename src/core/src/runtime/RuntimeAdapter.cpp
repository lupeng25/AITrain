#include "aitrain/runtime/RuntimeAdapter.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace aitrain {
namespace {

RuntimeOperationResult incompatible(const QString& message)
{
    return {RuntimeStatus::ArtifactIncompatible, message, {}};
}

bool isChildOf(const QString& path, const QString& root)
{
    const QString normalizedRoot = QDir::cleanPath(root);
    const QString normalizedPath = QDir::cleanPath(path);
    return normalizedPath == normalizedRoot || normalizedPath.startsWith(normalizedRoot + QLatin1Char('/'));
}

} // namespace

QString runtimeStatusToString(RuntimeStatus status)
{
    switch (status) {
    case RuntimeStatus::Available: return QStringLiteral("available");
    case RuntimeStatus::RuntimeNotImplemented: return QStringLiteral("runtime_not_implemented");
    case RuntimeStatus::SdkMissing: return QStringLiteral("sdk_missing");
    case RuntimeStatus::DependencyMissing: return QStringLiteral("dependency_missing");
    case RuntimeStatus::HardwareUnsupported: return QStringLiteral("hardware_unsupported");
    case RuntimeStatus::ArtifactIncompatible: return QStringLiteral("artifact_incompatible");
    }
    return QStringLiteral("runtime_not_implemented");
}

RuntimeOperationResult validateRuntimeModel(const RuntimeModelLocation& model, const QString& runtimeRoute)
{
    QString error;
    const QJsonObject manifest = encodeModelManifest(model.manifest, &error);
    if (manifest.isEmpty() || !canUseModelManifestForRuntime(&manifest, runtimeRoute, &error)) {
        return incompatible(error.isEmpty() ? QStringLiteral("模型缺少可用于目标运行时的有效 Manifest。") : error);
    }
    const QFileInfo rootInfo(model.artifactDirectory);
    if (!rootInfo.exists() || !rootInfo.isDir() || rootInfo.isSymLink()) {
        return incompatible(QStringLiteral("模型 Artifact 根目录不存在、不是目录或为符号链接。"));
    }
    const QString root = rootInfo.canonicalFilePath();
    const QFileInfo entryInfo(QDir(root).filePath(model.manifest.artifactEntryPath));
    if (!entryInfo.exists() || !entryInfo.isFile() || entryInfo.isSymLink() || !isChildOf(entryInfo.canonicalFilePath(), root)) {
        return incompatible(QStringLiteral("Model Manifest 的产物入口不存在、不安全或越出 Artifact 根目录。"));
    }
    QFile entry(entryInfo.canonicalFilePath());
    if (!entry.open(QIODevice::ReadOnly)) {
        return incompatible(QStringLiteral("无法读取 Model Manifest 指定的产物入口。"));
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!entry.atEnd()) {
        const QByteArray bytes = entry.read(1024 * 1024);
        if (bytes.isEmpty() && entry.error() != QFile::NoError) {
            return incompatible(QStringLiteral("读取模型产物时发生错误。"));
        }
        hash.addData(bytes);
    }
    if (QString::fromLatin1(hash.result().toHex()) != model.manifest.sourceArtifactSha256) {
        return incompatible(QStringLiteral("模型产物 SHA-256 与 Model Manifest 不一致。"));
    }
    return {RuntimeStatus::Available, QStringLiteral("模型通过 Manifest 和产物完整性准入。"), {}};
}

} // namespace aitrain
