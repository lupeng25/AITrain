#include "aitrain/v2/DatasetSnapshotV2.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>
#include <QVector>

#include <algorithm>

namespace aitrain::v2 {
namespace {

struct SnapshotFile final {
    QString relativePath;
    qint64 bytes = 0;
    QString sha256;
};

bool canceled(const DatasetSnapshotOptions& options)
{
    return options.isCancellationRequested && options.isCancellationRequested();
}

bool hashFile(const QString& path, QString* hash, QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = QStringLiteral("无法读取快照文件：%1").arg(path);
        }
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray bytes = file.read(1024 * 1024);
        if (bytes.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) {
                *error = QStringLiteral("读取快照文件失败：%1").arg(path);
            }
            return false;
        }
        digest.addData(bytes);
    }
    if (hash) {
        *hash = QString::fromLatin1(digest.result().toHex());
    }
    return true;
}

} // namespace

bool createDatasetSnapshotV2(const QString& datasetRoot,
    const QString& manifestPath,
    const QString& datasetFormat,
    const QString& driverId,
    const QString& driverVersion,
    const DatasetSnapshotOptions& options,
    DatasetSnapshotResult* result,
    QString* error)
{
    const QDir root(datasetRoot);
    if (!root.exists() || manifestPath.isEmpty() || datasetFormat.isEmpty() || driverId.isEmpty() || driverVersion.isEmpty() || !result) {
        if (error) {
            *error = QStringLiteral("Dataset Snapshot V2 参数无效。");
        }
        return false;
    }
    if (QFileInfo::exists(manifestPath)) {
        if (error) {
            *error = QStringLiteral("快照 manifest 已存在，禁止覆盖：%1").arg(manifestPath);
        }
        return false;
    }

    QVector<SnapshotFile> files;
    QDirIterator iterator(root.absolutePath(), QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        if (canceled(options)) {
            if (error) {
                *error = QStringLiteral("snapshot_canceled");
            }
            return false;
        }
        const QString absolutePath = iterator.next();
        const QFileInfo info(absolutePath);
        const QString relativePath = QDir::cleanPath(root.relativeFilePath(absolutePath));
        if (relativePath.startsWith(QStringLiteral("../")) || relativePath == QStringLiteral("..")) {
            if (error) {
                *error = QStringLiteral("快照文件路径越出数据集根目录：%1").arg(absolutePath);
            }
            return false;
        }
        if (files.size() >= options.maxFileCount) {
            if (error) {
                *error = QStringLiteral("file_limit_exceeded：文件数超过安全上限 %1。").arg(options.maxFileCount);
            }
            return false;
        }
        files.append({relativePath, info.size(), {}});
        if (options.progress) {
            options.progress(files.size());
        }
    }
    std::sort(files.begin(), files.end(), [](const SnapshotFile& left, const SnapshotFile& right) {
        return left.relativePath < right.relativePath;
    });

    QCryptographicHash rootHash(QCryptographicHash::Sha256);
    QJsonArray fileArray;
    qint64 totalBytes = 0;
    for (SnapshotFile& entry : files) {
        if (canceled(options)) {
            if (error) {
                *error = QStringLiteral("snapshot_canceled");
            }
            return false;
        }
        if (!hashFile(root.filePath(entry.relativePath), &entry.sha256, error)) {
            return false;
        }
        totalBytes += entry.bytes;
        rootHash.addData(entry.relativePath.toUtf8());
        rootHash.addData("\0", 1);
        rootHash.addData(QByteArray::number(entry.bytes));
        rootHash.addData("\0", 1);
        rootHash.addData(entry.sha256.toLatin1());
        rootHash.addData("\0", 1);
        fileArray.append(QJsonObject{{QStringLiteral("relativePath"), entry.relativePath},
            {QStringLiteral("bytes"), QString::number(entry.bytes)},
            {QStringLiteral("sha256"), entry.sha256}});
    }

    const SnapshotId snapshotId = SnapshotId::create();
    QJsonObject manifest;
    manifest.insert(QStringLiteral("schemaVersion"), 2);
    manifest.insert(QStringLiteral("complete"), true);
    manifest.insert(QStringLiteral("snapshotId"), snapshotId.toString());
    manifest.insert(QStringLiteral("driver"), QJsonObject{{QStringLiteral("id"), driverId}, {QStringLiteral("version"), driverVersion}});
    manifest.insert(QStringLiteral("datasetFormat"), datasetFormat);
    manifest.insert(QStringLiteral("classDefinitions"), options.classDefinitions);
    manifest.insert(QStringLiteral("fileCount"), QString::number(files.size()));
    manifest.insert(QStringLiteral("totalBytes"), QString::number(totalBytes));
    manifest.insert(QStringLiteral("rootHash"), QString::fromLatin1(rootHash.result().toHex()));
    manifest.insert(QStringLiteral("files"), fileArray);

    QDir outputDirectory(QFileInfo(manifestPath).absolutePath());
    if (!outputDirectory.exists() && !QDir().mkpath(outputDirectory.absolutePath())) {
        if (error) {
            *error = QStringLiteral("无法创建快照输出目录：%1").arg(outputDirectory.absolutePath());
        }
        return false;
    }
    if (canceled(options)) {
        if (error) {
            *error = QStringLiteral("snapshot_canceled");
        }
        return false;
    }
    QSaveFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::WriteOnly)
        || manifestFile.write(QJsonDocument(manifest).toJson(QJsonDocument::Indented)) < 0
        || !manifestFile.commit()) {
        if (error) {
            *error = QStringLiteral("无法写入完整快照 manifest：%1").arg(manifestFile.errorString());
        }
        return false;
    }
    result->snapshotId = snapshotId;
    result->manifestPath = manifestPath;
    result->rootHash = manifest.value(QStringLiteral("rootHash")).toString();
    result->fileCount = files.size();
    result->totalBytes = totalBytes;
    result->manifest = manifest;
    return true;
}

} // namespace aitrain::v2
