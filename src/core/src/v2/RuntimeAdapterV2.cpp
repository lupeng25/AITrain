#include "aitrain/v2/RuntimeAdapterV2.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace aitrain::v2 {
namespace {

RuntimeOperationResultV2 incompatible(const QString& message)
{
    return {RuntimeStatusV2::ArtifactIncompatible, message, {}};
}

bool isChildOf(const QString& path, const QString& root)
{
    const QString normalizedRoot = QDir::cleanPath(root);
    const QString normalizedPath = QDir::cleanPath(path);
    return normalizedPath == normalizedRoot || normalizedPath.startsWith(normalizedRoot + QLatin1Char('/'));
}

} // namespace

QString runtimeStatusV2ToString(RuntimeStatusV2 status)
{
    switch (status) {
    case RuntimeStatusV2::Available: return QStringLiteral("available");
    case RuntimeStatusV2::RuntimeNotImplemented: return QStringLiteral("runtime_not_implemented");
    case RuntimeStatusV2::SdkMissing: return QStringLiteral("sdk_missing");
    case RuntimeStatusV2::DependencyMissing: return QStringLiteral("dependency_missing");
    case RuntimeStatusV2::HardwareUnsupported: return QStringLiteral("hardware_unsupported");
    case RuntimeStatusV2::ArtifactIncompatible: return QStringLiteral("artifact_incompatible");
    }
    return QStringLiteral("runtime_not_implemented");
}

RuntimeOperationResultV2 validateRuntimeModelV2(const RuntimeModelLocationV2& model, const QString& runtimeRoute)
{
    QString error;
    const QJsonObject manifest = encodeModelManifestV2(model.manifest, &error);
    if (manifest.isEmpty() || !canUseModelManifestV2ForRuntime(&manifest, runtimeRoute, &error)) {
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
    return {RuntimeStatusV2::Available, QStringLiteral("模型通过 Manifest 和产物完整性准入。"), {}};
}

} // namespace aitrain::v2
